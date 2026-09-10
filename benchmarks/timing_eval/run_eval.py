"""Run the CODA timing eval over the real recordings with a configurable inference model.

Transcription (faster-whisper) is cached per transcriber configuration under
transcripts/<transcriber>-<whisper-model>/ and reused across inference models, so
running a second model does not re-transcribe. Grounding is skipped: the CHAMPS agent
infers from text only, so gilda annotations do not affect the result.

Each (case, phase, mode) is transcribed once (whole-file and 20s-chunked) and inferred
with the chosen model. Phase 1 is the VA narrative; phase 2 is VA followed by the
clinical narrative (combined audio), run so the agent carries VA context into it.

    python run_eval.py --model qwen2.5:7b-instruct
    python run_eval.py --model gpt-oss:20b
    python run_eval.py --agent embedding --model-path ../../../champs_statsML/output/coda_dialogue_model.joblib

Results are written to real_cases_results/<model>/, one subfolder per model, so the
report can compare every model that has been run.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from coda.cli import SAMPLE_RATE, load_audio_int16, build_agent, write_outputs
from coda.dialogue import AudioProcessor, create_transcriber

from make_combined import COMBINED_DIR, concat_va_clinical, load_cases

TRANSCRIBER = "faster-whisper"
WHISPER_MODEL = "medium"
CHUNK_SECONDS = 20.0
NO_SPEECH_THRESHOLD = 1.0
LANGUAGE = "en"
TASK = "transcribe"
PROVIDER = "ollama"
PHASE_DIRS = {"va": "phase1_va", "combined": "phase2_va_clinical"}
MODES = ["whole", "chunked"]


def slug(s):
    return s.replace(":", "-").replace("/", "-")


# Transcription

async def transcribe(transcriber, audio_i16, mode):
    """Transcribe audio into cache segments: whole-file is one segment, chunked is one
    per 20s chunk plus the trailing remainder. Each records its own transcription time."""
    async def one(chunk_id, audio):
        t = time.perf_counter()
        text = await transcriber.transcribe_audio(
            audio, sample_rate=SAMPLE_RATE, language=LANGUAGE, task=TASK)
        return {"chunk_id": chunk_id, "audio_s": round(len(audio) / SAMPLE_RATE, 3),
                "text": text, "transcription_s": round(time.perf_counter() - t, 3)}

    if mode == "whole":
        return [await one("chunk-0", audio_i16)]
    segs = []
    processor = AudioProcessor(sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_SECONDS)
    processor.add_audio(audio_i16.tobytes())
    while True:
        chunk = processor.get_chunk()
        if chunk is None:
            break
        chunk_id, _ts, audio = chunk
        segs.append(await one(chunk_id, audio))
    tail = processor.audio_buffer
    if tail.size > 0:
        segs.append(await one("chunk-tail", tail))
    return segs


async def get_segments(cache_file, input_path, mode, retranscribe, transcriber_holder):
    """Cached transcript segments for one (phase, mode); transcribe on a miss."""
    if cache_file.exists() and not retranscribe:
        print("    transcript: cached")
        return json.loads(cache_file.read_text())
    print("    transcript: transcribing")
    if transcriber_holder[0] is None:  # load the whisper model once, only if needed
        transcriber_holder[0] = create_transcriber(
            TRANSCRIBER, model=WHISPER_MODEL, no_speech_threshold=NO_SPEECH_THRESHOLD)
    segs = await transcribe(transcriber_holder[0], load_audio_int16(str(input_path)), mode)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(segs))
    return segs


# Inference

async def infer(segments, agent_name, provider, model, model_path):
    """Replay transcript segments through the inference agent, timing each inference."""
    agent = build_agent(agent_name, provider=provider, model=model, model_path=model_path)
    per_chunk = []
    for s in segments:
        t = time.perf_counter()
        inference = await agent.process_chunk(s["chunk_id"], s["text"], [], time.time())
        per_chunk.append((s["chunk_id"], s["text"], [], inference,
                          {"audio_s": s["audio_s"], "transcription_s": s["transcription_s"],
                           "grounding_s": 0.0,
                           "inference_s": round(time.perf_counter() - t, 3)}))
    return agent.all_text.strip(), per_chunk


# Main

async def run(args):
    cache_root = HERE / "transcripts" / f"{slug(TRANSCRIBER)}-{slug(WHISPER_MODEL)}"
    run_name = args.tag or args.model or ("embedding" if args.agent == "embedding" else None)
    out_root = Path(args.out) / slug(run_name) if run_name else None
    cases = load_cases()
    case_ids = args.only or sorted(cases, key=int)
    phases = [p for p in args.phases.split(",") if p]
    modes = [m for m in args.modes.split(",") if m]
    transcriber_holder = [None]

    total = failed = 0
    for cid in case_ids:
        parts = cases.get(cid, {})
        inputs = {}
        if parts.get("va") and parts["va"].exists():
            inputs["va"] = parts["va"]
        if parts.get("clinical") and parts["clinical"].exists() and "va" in inputs:
            combined = COMBINED_DIR / f"case{cid}_combined.wav"
            if not combined.exists():
                COMBINED_DIR.mkdir(parents=True, exist_ok=True)
                concat_va_clinical(inputs["va"], parts["clinical"], combined)
            inputs["combined"] = combined

        for phase in phases:
            if phase not in inputs:
                continue
            for mode in modes:
                total += 1
                print(f"=== [{total}] case{cid} {PHASE_DIRS[phase]} {mode}  "
                      f"({run_name or 'transcribe-only'})")
                out_dir = out_root / f"case{cid}" / PHASE_DIRS[phase] / mode if out_root else None
                if out_dir and (out_dir / "inference.json").exists() and not args.force:
                    print("    skip (already done)")
                    continue
                cache_file = cache_root / f"case{cid}" / PHASE_DIRS[phase] / f"{mode}.json"
                try:
                    segs = await get_segments(cache_file, inputs[phase], mode,
                                              args.retranscribe, transcriber_holder)
                    if args.transcribe_only:
                        continue
                    full_text, per_chunk = await infer(
                        segs, args.agent, args.provider, args.model, args.model_path)
                except Exception as e:
                    failed += 1
                    print(f"!! FAILED: {e}", file=sys.stderr)
                    continue
                meta = {"input": str(inputs[phase]), "input_type": "audio", "mode": mode,
                        "agent": args.agent, "provider": args.provider, "model": args.model,
                        "model_path": args.model_path,
                        "transcriber": TRANSCRIBER, "whisper_model": WHISPER_MODEL,
                        "language": LANGUAGE, "task": TASK,
                        "audio_duration_s": round(sum(s["audio_s"] for s in segs), 1),
                        "transcript_cache": str(cache_file)}
                write_outputs(out_dir, full_text, per_chunk, meta)
                top = per_chunk[-1][3].get("causes", {})
                best = max(top.values(), key=lambda c: c["score"])["name"] if top else "-"
                print(f"    -> top={best}")

    print(f"\nDone: {total} run(s), {failed} failure(s). Output in {out_root}")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=["champs", "embedding"], default="champs",
                    help="Inference agent (default: champs LLM agent)")
    ap.add_argument("--model", help="LLM model, --agent champs (e.g. qwen2.5:7b-instruct)")
    ap.add_argument("--model-path", help="Joblib bundle path, --agent embedding")
    ap.add_argument("--tag", default=None,
                    help="Results subfolder name (default: --model, or 'embedding')")
    ap.add_argument("--transcribe-only", action="store_true",
                    help="Populate the transcript cache and skip inference")
    ap.add_argument("--provider", default=PROVIDER)
    ap.add_argument("--out", default="real_cases_results",
                    help="Results root, each model gets its own <out>/<model>/ subfolder")
    ap.add_argument("--only", nargs="*", default=None, help="Case ids (default: all)")
    ap.add_argument("--phases", default="va,combined")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--force", action="store_true",
                    help="Re-run inference even if output exists (transcript cache still reused)")
    ap.add_argument("--retranscribe", action="store_true",
                    help="Ignore the transcript cache and transcribe again")
    args = ap.parse_args()
    if not args.transcribe_only:
        if args.agent == "champs" and not args.model:
            ap.error("--model is required for --agent champs unless --transcribe-only")
        if args.agent == "embedding" and not args.model_path:
            ap.error("--model-path is required for --agent embedding unless --transcribe-only")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
