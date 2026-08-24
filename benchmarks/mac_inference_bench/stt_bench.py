"""Benchmark WhisperLiveKit speech-to-text, matching CODA's transcriber path.

Prints a single JSON object to stdout. Logs go to stderr.

The shipped clip is a 16 kHz mono PCM WAV read with the standard library, so no
ffmpeg is needed. stream mode feeds that PCM to WhisperLiveKit in real time the
same way CODA feeds browser audio (pcm_input=True), so keep_up = wall / clip
reports whether the machine transcribes in real time (<= 1.0) or fell behind.

batch mode hands the decoded samples to the underlying mlx-whisper model as fast
as possible and reports its raw throughput (speedup over real time).
"""
import argparse
import asyncio
import contextlib
import json
import time
import wave


def peak_gb():
    import mlx.core as mx
    fn = getattr(mx, "get_peak_memory", None)
    if fn is None:
        fn = getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    if fn is None:
        return None
    try:
        return fn() / 1e9
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


def run_batch(audio, model_repo, language, loop_seconds):
    import mlx_whisper
    samples, rate = read_float32(audio)
    clip = len(samples) / rate
    kw = dict(path_or_hf_repo=model_repo, language=language)
    mlx_whisper.transcribe(samples, **kw)
    audio_sec = 0.0
    wall = 0.0
    iters = 0
    deadline = time.time() + loop_seconds
    while time.time() < deadline:
        t0 = time.time()
        mlx_whisper.transcribe(samples, **kw)
        wall += time.time() - t0
        audio_sec += clip
        iters += 1
    return {
        "mode": "batch",
        "backend": "mlx-whisper",
        "model": model_repo,
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
        help="WLK model size for stream mode, or hf repo for batch mode")
    ap.add_argument("--backend", default="mlx-whisper")
    ap.add_argument("--policy", default="localagreement")
    ap.add_argument("--language", default="en")
    ap.add_argument("--loop-seconds", type=float, default=20.0)
    args = ap.parse_args()
    if args.mode == "batch":
        repo = args.model if "/" in args.model else "mlx-community/whisper-small-mlx"
        result = run_batch(args.audio, repo, args.language, args.loop_seconds)
    elif args.mode == "prepull":
        result = prepull(args.model, args.backend, args.policy, args.language)
    else:
        result = asyncio.run(run_stream(args.audio, args.model, args.backend,
            args.policy, args.language))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
