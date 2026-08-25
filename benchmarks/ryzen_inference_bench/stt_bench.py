"""Benchmark WhisperLiveKit speech-to-text, matching CODA's transcriber path.

Ryzen/Windows variant of the Mac stt_bench: same WhisperLiveKit streaming path,
model size, and localagreement policy. faster-whisper (CTranslate2) has no
Vulkan/ROCm support and falls back to CPU on this hardware, which can't keep up
with real time (keep_up ~2.9x). "lemonade-whisper" instead routes through
Lemonade's whispercpp/Vulkan backend (iGPU-accelerated), which keeps up with
real time (keep_up ~1.0x). keep_up stays directly comparable to the Mac runs
since it is a real-time factor. The whisper weights differ across backends, so
this compares throughput, not word error rate.

Prints a single JSON object to stdout. Logs go to stderr.
"""
import argparse
import asyncio
import contextlib
import json
import os
import time
import wave

LEMONADE_BASE = "http://localhost:13305/api/v1"


def _patch_lemonade_whisper():
    """Adapt WhisperLiveKit's OpenaiApiASR to Lemonade's whisper.cpp response shape.

    WhisperLiveKit hardcodes model="whisper-1" (register that alias in Lemonade
    to point at your loaded whisper model, e.g. `lemonade alias add whisper-1
    Whisper-Small`). Lemonade also nests word timestamps per-segment as plain
    dicts (segments[i].words[j]['word']) rather than OpenAI's flat top-level
    resp.words with object attributes, so the real OpenAI API would not need
    (and would break under) this patch.
    """
    from whisperlivekit.local_agreement.backends import OpenaiApiASR
    from whisperlivekit.timed_objects import ASRToken

    def ts_words(self, resp):
        tokens = []
        for seg in (resp.segments or []):
            for w in (seg.words or []):
                tokens.append(ASRToken(w["start"], w["end"], w["word"],
                    probability=w.get("probability")))
        return tokens

    def segments_end_ts(self, resp):
        return [seg.end for seg in (resp.segments or [])]

    OpenaiApiASR.ts_words = ts_words
    OpenaiApiASR.segments_end_ts = segments_end_ts


def peak_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None) or mi.rss
        return peak / 1e9
    except Exception:
        return None


def read_pcm(path):
    with contextlib.closing(wave.open(str(path), "rb")) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("clip must be 16-bit mono PCM WAV")
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    return pcm, rate


def read_float32(path):
    import numpy as np
    pcm, rate = read_pcm(path)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, rate


async def run_stream(audio, model_size, backend, policy, language):
    if backend == "lemonade-whisper":
        os.environ["OPENAI_API_KEY"] = "local"
        os.environ["OPENAI_BASE_URL"] = LEMONADE_BASE
        _patch_lemonade_whisper()
        backend = "openai-api"

    from whisperlivekit import TranscriptionEngine, AudioProcessor
    pcm, rate = read_pcm(audio)
    clip = len(pcm) / 2 / rate
    engine = TranscriptionEngine(model_size=model_size, lan=language,
        backend=backend, backend_policy=policy, pcm_input=True)
    processor = AudioProcessor(transcription_engine=engine)
    results = await processor.create_tasks()

    async def feed():
        step = int(rate * 0.5) * 2
        for i in range(0, len(pcm), step):
            await processor.process_audio(pcm[i:i + step])
            await asyncio.sleep(0.5)
        await processor.process_audio(b"")

    lines = {}
    start = time.time()
    feeder = asyncio.create_task(feed())

    async def consume():
        async for response in results:
            msg = response.to_dict() if hasattr(response, "to_dict") else response
            for line in msg.get("lines", []):
                text = (line.get("text") or "").strip()
                if text:
                    lines[line.get("start")] = text

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(consume(), timeout=clip * 3 + 60)
    wall = time.time() - start
    feeder.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await feeder
    with contextlib.suppress(Exception):
        await processor.cleanup()

    committed = sum(len(t.split()) for t in lines.values())
    return {
        "mode": "stream",
        "backend": backend,
        "model": model_size,
        "clip_sec": round(clip, 2),
        "wall_sec": round(wall, 2),
        "keep_up": round(wall / clip, 3) if clip else None,
        "committed_words": committed,
        "peak_gb": round(peak_gb() or 0, 2),
    }


def run_batch(audio, model_size, language, loop_seconds):
    from faster_whisper import WhisperModel
    samples, rate = read_float32(audio)
    clip = len(samples) / rate
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe():
        segments, _ = model.transcribe(samples, language=language)
        for _ in segments:
            pass

    transcribe()
    audio_sec = 0.0
    wall = 0.0
    iters = 0
    deadline = time.time() + loop_seconds
    while time.time() < deadline:
        t0 = time.time()
        transcribe()
        wall += time.time() - t0
        audio_sec += clip
        iters += 1
    return {
        "mode": "batch",
        "backend": "faster-whisper",
        "model": model_size,
        "clip_sec": round(clip, 2),
        "iterations": iters,
        "rtf": round(wall / audio_sec, 3) if audio_sec else None,
        "speedup_x": round(audio_sec / wall, 2) if wall else None,
        "peak_gb": round(peak_gb() or 0, 2),
    }


def prepull(model_size, backend, policy, language):
    from whisperlivekit import TranscriptionEngine
    TranscriptionEngine(model_size=model_size, lan=language,
        backend=backend, backend_policy=policy, pcm_input=True)
    return {"mode": "prepull", "backend": backend, "model": model_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--mode", choices=("stream", "batch", "prepull"),
        default="stream")
    ap.add_argument("--model", default="small",
        help="whisper model size (small matches the Mac run)")
    ap.add_argument("--backend", default="lemonade-whisper")
    ap.add_argument("--policy", default="localagreement")
    ap.add_argument("--language", default="en")
    ap.add_argument("--loop-seconds", type=float, default=20.0)
    args = ap.parse_args()
    if args.mode == "batch":
        result = run_batch(args.audio, args.model, args.language, args.loop_seconds)
    elif args.mode == "prepull":
        result = prepull(args.model, args.backend, args.policy, args.language)
    else:
        result = asyncio.run(run_stream(args.audio, args.model, args.backend,
            args.policy, args.language))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
