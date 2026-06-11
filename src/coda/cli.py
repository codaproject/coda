"""CLI for running CODA offline: audio file -> cause-of-death inference.

Wires the same in-process components the web app uses (Whisper transcription,
grounding, inference agent) without the server/WebSocket layer.

Example
-------
    python -m coda.cli --input recording.mp3 --output results/
"""
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np
import whisper

from coda.dialogue import AudioProcessor, DEFAULT_CHUNK_DURATION
from coda.dialogue.whisper import WhisperTranscriber, DEFAULT_MODEL_SIZE
from coda.grounding.gilda_grounder import GildaGrounder
from coda.inference.agent import CodaToyInferenceAgent

logger = logging.getLogger("coda.cli")

SAMPLE_RATE = 16000


def build_grounder(name: str, provider: str = None, model: str = None):
    """Construct a grounder from a flag value ("gilda" or "rag")."""
    if name == "rag":
        from coda.grounding.rag_grounder import RagGrounder
        grounder = RagGrounder()
        cfg = {k: v for k, v in (("provider", provider), ("model", model)) if v}
        if cfg:
            grounder.update_config(**cfg)
        return grounder
    return GildaGrounder()


def build_agent(name: str, provider: str = None, model: str = None):
    """Construct an inference agent from a flag value ("champs" or "toy")."""
    if name == "toy":
        return CodaToyInferenceAgent()
    from coda.inference.champs_llm_agent import create_champs_agent
    kwargs = {k: v for k, v in (("provider", provider), ("model", model)) if v}
    return create_champs_agent(**kwargs)


def load_audio_int16(input_path: str) -> np.ndarray:
    """Load any ffmpeg-readable audio as 16kHz mono int16 (Whisper's expected rate)."""
    audio_f32 = whisper.load_audio(input_path, sr=SAMPLE_RATE)
    return (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)


def _ground(grounder, text):
    """Ground text to annotations; returns (annotations, elapsed_seconds)."""
    t = time.perf_counter()
    annotations = grounder.annotate(text) if text else []
    return annotations, round(time.perf_counter() - t, 3)


async def _infer(agent, chunk_id, text, annotations, ts):
    """Run inference for one chunk; returns (inference, elapsed_seconds)."""
    t = time.perf_counter()
    inference = await agent.process_chunk(chunk_id, text, annotations, ts)
    return inference, round(time.perf_counter() - t, 3)


async def _transcribe_and_infer(transcriber, grounder, agent, chunk_id, audio,
                                language, task, ts):
    """Transcribe, ground, and infer for one audio chunk, timing each step."""
    t = time.perf_counter()
    text = await transcriber.transcribe_audio(
        audio, sample_rate=SAMPLE_RATE, language=language, task=task
    )
    transcription_s = round(time.perf_counter() - t, 3)
    annotations, grounding_s = _ground(grounder, text)
    inference, inference_s = await _infer(agent, chunk_id, text, annotations, ts)
    timings = {"transcription_s": transcription_s, "grounding_s": grounding_s,
               "inference_s": inference_s}
    return chunk_id, text, annotations, inference, timings


async def run_whole_file(transcriber, grounder, agent, audio_i16, language, task):
    """Transcribe the entire recording in one pass, then a single inference call."""
    row = await _transcribe_and_infer(
        transcriber, grounder, agent, "chunk-0", audio_i16, language, task, time.time()
    )
    return row[1], [row]


async def run_chunked(transcriber, grounder, agent, audio_i16, language, task,
                      chunk_duration):
    """Simulate the live pipeline: feed fixed-duration chunks and infer per chunk."""
    processor = AudioProcessor(sample_rate=SAMPLE_RATE, chunk_duration=chunk_duration)
    processor.add_audio(audio_i16.tobytes())

    per_chunk = []
    while True:
        chunk = processor.get_chunk()
        if chunk is None:
            break
        chunk_id, ts, audio = chunk
        per_chunk.append(await _transcribe_and_infer(
            transcriber, grounder, agent, chunk_id, audio, language, task, ts
        ))

    # Process the trailing remainder (< chunk_size) that get_chunk() leaves behind
    tail = processor.audio_buffer
    if tail.size > 0:
        per_chunk.append(await _transcribe_and_infer(
            transcriber, grounder, agent, "chunk-tail", tail, language, task, time.time()
        ))

    return agent.all_text.strip(), per_chunk


async def run_text(grounder, agent, text):
    """Ground a pre-existing transcript and run a single inference call."""
    annotations, grounding_s = _ground(grounder, text)
    inference, inference_s = await _infer(agent, "chunk-0", text, annotations, time.time())
    timings = {"grounding_s": grounding_s, "inference_s": inference_s}
    return text, [("chunk-0", text, annotations, inference, timings)]


def write_outputs(output_dir: Path, full_text: str, per_chunk: list, meta: dict):
    """Write transcript, annotations, per-chunk trace, and final inference to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "transcript.txt").write_text(full_text + "\n")

    # All annotations across chunks, serialized in gilda's JSON form
    annotations = []
    for _, _, anns, _, _ in per_chunk:
        annotations.extend(a.to_json() for a in anns)
    (output_dir / "annotations.json").write_text(json.dumps(annotations, indent=2))

    # Per-chunk trace (transcript + causes + timing), useful mainly in chunked mode
    with (output_dir / "chunks.jsonl").open("w") as fh:
        for chunk_id, text, anns, inf, timings in per_chunk:
            fh.write(json.dumps({
                "chunk_id": chunk_id,
                "timestamp": inf.get("timestamp"),
                "text": text,
                "annotations": [a.to_json() for a in anns],
                "causes": inf.get("causes", {}),
                "reasoning": inf.get("reasoning"),
                "timing": timings,
            }) + "\n")

    # Roll up timing across all chunks (steps timed separately; the text path
    # has no transcription step, the audio path has all three).
    transcription_s = round(sum(t.get("transcription_s", 0) for *_, t in per_chunk), 3)
    grounding_s = round(sum(t.get("grounding_s", 0) for *_, t in per_chunk), 3)
    inference_s = round(sum(t.get("inference_s", 0) for *_, t in per_chunk), 3)
    audio_s = meta.get("audio_duration_s") or 0
    timing = {
        "transcription_s": transcription_s,
        "grounding_s": grounding_s,
        "inference_s": inference_s,
        "total_s": round(transcription_s + grounding_s + inference_s, 3),
        # real-time factor: <1 means faster than real time
        "transcription_rtf": round(transcription_s / audio_s, 3) if audio_s else None,
    }

    # Final inference is the last chunk's result (agents accumulate dialogue history)
    final = per_chunk[-1][3] if per_chunk else {}
    (output_dir / "inference.json").write_text(json.dumps({
        **meta,
        "timing": timing,
        "causes": final.get("causes", {}),
        "reasoning": final.get("reasoning"),
        "chunks_processed": final.get("chunks_processed"),
    }, indent=2))


def print_summary(full_text: str, per_chunk: list):
    """Print a short human-readable summary of the top causes to stdout."""
    final = per_chunk[-1][3] if per_chunk else {}
    causes = final.get("causes", {})
    transcription_s = sum(t.get("transcription_s", 0) for *_, t in per_chunk)
    grounding_s = sum(t.get("grounding_s", 0) for *_, t in per_chunk)
    inference_s = sum(t.get("inference_s", 0) for *_, t in per_chunk)
    parts = []
    if transcription_s:
        parts.append(f"transcription {transcription_s:.1f}s")
    if grounding_s:
        parts.append(f"grounding {grounding_s:.1f}s")
    parts.append(f"inference {inference_s:.1f}s")
    print("\nTiming: " + ", ".join(parts))
    print(f"\nTranscript ({len(full_text)} chars):")
    print(f"  {full_text[:500]}{'...' if len(full_text) > 500 else ''}\n")
    if not causes:
        print("No causes inferred.")
        return
    print("Top causes:")
    for curie, c in sorted(causes.items(), key=lambda kv: kv[1]["score"], reverse=True):
        print(f"  {c['score']:.3f}  {curie}  {c['name']}")
    if final.get("reasoning"):
        print(f"\nReasoning: {final['reasoning']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run CODA on an audio file: transcribe, ground, infer cause of death."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to input audio file (mp3/wav/...)")
    source.add_argument("--text", help="Path to a text/transcript file")
    parser.add_argument("--output", required=True, help="Output folder for results")
    parser.add_argument("--chunking", type=float, nargs="?", const=DEFAULT_CHUNK_DURATION,
                        default=None, metavar="SECONDS",
                        help="Simulate the live app's chunked pipeline. Bare --chunking uses the "
                             f"production chunk length ({DEFAULT_CHUNK_DURATION}s); pass a number to "
                             "override. Omit entirely for one-pass whole-file transcription (default).")
    parser.add_argument("--agent", choices=["champs", "toy"], default="champs",
                        help="Inference agent (default: champs LLM agent)")
    parser.add_argument("--grounder", choices=["gilda", "rag"], default="gilda",
                        help="Grounder for entity/code annotation (default: gilda)")
    parser.add_argument("--provider", default=None,
                        help="LLM provider for the agent / RAG grounder (e.g. openai, ollama)")
    parser.add_argument("--model", default=None,
                        help="LLM model name (e.g. gpt-4o-mini, gpt-oss:20b)")
    parser.add_argument("--whisper-model", default=DEFAULT_MODEL_SIZE,
                        help=f"Whisper model size (default: {DEFAULT_MODEL_SIZE})")
    parser.add_argument("--language", default="en", help="Spoken language (default: en)")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe",
                        help="transcribe in language, or translate speech to English")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    grounder = build_grounder(args.grounder, args.provider, args.model)
    agent = build_agent(args.agent, args.provider, args.model)

    common_meta = {
        "agent": args.agent,
        "grounder": args.grounder,
        "provider": args.provider,
        "model": args.model,
    }

    if args.text:
        if args.chunking is not None:
            print("Note: --chunking is ignored for text input.")
        text_in = Path(args.text).read_text().strip()
        print(f"Loaded text: {args.text} ({len(text_in)} chars). Mode=text, "
              f"agent={args.agent}, grounder={args.grounder}")
        full_text, per_chunk = asyncio.run(run_text(grounder, agent, text_in))
        meta = {
            "input": str(Path(args.text).resolve()),
            "input_type": "text",
            "mode": "text",
            **common_meta,
        }
    else:
        transcriber = WhisperTranscriber(model_size=args.whisper_model)
        print(f"Loading audio: {args.input}")
        audio_i16 = load_audio_int16(args.input)
        duration_s = len(audio_i16) / SAMPLE_RATE
        mode = "whole" if args.chunking is None else f"chunked@{args.chunking}s"
        print(f"Loaded {duration_s:.1f}s of audio. Mode={mode}, "
              f"agent={args.agent}, grounder={args.grounder}")

        if args.chunking is None:
            full_text, per_chunk = asyncio.run(
                run_whole_file(transcriber, grounder, agent, audio_i16,
                               args.language, args.task)
            )
        else:
            full_text, per_chunk = asyncio.run(
                run_chunked(transcriber, grounder, agent, audio_i16,
                            args.language, args.task, args.chunking)
            )
        meta = {
            "input": str(Path(args.input).resolve()),
            "input_type": "audio",
            "mode": mode,
            **common_meta,
            "whisper_model": args.whisper_model,
            "language": args.language,
            "task": args.task,
            "audio_duration_s": round(duration_s, 1),
        }

    output_dir = Path(args.output)
    write_outputs(output_dir, full_text, per_chunk, meta)
    print_summary(full_text, per_chunk)
    print(f"\nResults written to {output_dir.resolve()}/")


if __name__ == "__main__":
    main()
