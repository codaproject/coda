"""Runner: stream one benchmark composite through CODA's transcriber.

Feeds the composite to CODA's whisper-livekit transcriber as 16 kHz PCM in
the pieces the browser sends, and writes one JSON with every committed word
and its audio time, the text CODA would have emitted, and the configuration
that was active. The scorer reads that JSON together with the composite's
region map.

Any --set KEY=VALUE becomes a CODA_DIALOGUE__WHISPER_LIVEKIT__KEY environment
override before CODA loads its config, so a run tests exactly what the app
would do with that setting. WhisperLiveKit's engine is a process-wide
singleton, so one configuration per process. Results are written under
results/<feed>/<config>/<variant>/<language>/<case>.json, where feed is
lockstep or realtime (below).

How audio is fed decides what the result measures. WhisperLiveKit accepts a
word only when two consecutive passes agree on it and trims audio it has not
caught up with, so on a slow machine fed at speaking speed it drops words,
which is a property of the machine, not of the model. The default here is
lock-step: the next chunk is fed only once everything before it has been
processed, so the transcriber keeps up by construction and the result is the
same on any hardware. --realtime feeds at live speed instead, to measure
whether a given machine keeps pace.

    python benchmarks/synthetic_narrative/run.py \\
        benchmarks/synthetic_narrative/data/benchmark/controlled/en/lri_1.wav \\
        --config ns_0.4 --set DECODE_OPTIONS__NO_SPEECH_THRESHOLD=0.4
"""
import argparse
import asyncio
import contextlib
import importlib
import json
import os
import time
from pathlib import Path

from common import HERE, SAMPLE_RATE
import audio

RESULTS_DIR = HERE / "results"
ENV_PREFIX = "CODA_DIALOGUE__WHISPER_LIVEKIT__"
# Audio fed per step: the app's page sends 4096 samples per message (the
# ScriptProcessor buffer in templates/index.html), 0.256 s at 16 kHz.
CHUNK_SAMPLES = 4096
MAX_RUN_S = 6 * 3600.0  # safety limit on one run; lock-step on CPU can be very slow


def asr_language(language):
    """Whisper language code, as the lang_support adapters define it."""
    if language == "en":
        return "en"
    return importlib.import_module(f"languages.{language}").ASR_LANGUAGE


async def stream(engine, pcm, realtime):
    """Feed pcm through a WhisperLiveKit session; return (words, lines, emitted, complete).

    words: every committed token with audio-time start/end, from the alignment
    the response lines are built from. lines: the final response lines.
    emitted: the committed text CODA's own event parser would have produced.
    """
    from whisperlivekit import AudioProcessor
    from coda.dialogue.whisper_livekit import _events_from_response
    processor = AudioProcessor(transcription_engine=engine)
    results = await processor.create_tasks()
    raw = pcm.tobytes()
    step = CHUNK_SAMPLES * 2  # s16le bytes per chunk

    # Count decodes in flight, so lock-step feeding can tell "caught up" from
    # "queue empty but still decoding the last chunk". The backend's decode
    # runs in a worker thread; wrapping it on this engine object is local to
    # this process.
    busy = [0]
    decode = engine.asr.transcribe

    def counted_decode(*args, **kwargs):
        busy[0] += 1
        try:
            return decode(*args, **kwargs)
        finally:
            busy[0] -= 1
    engine.asr.transcribe = counted_decode

    async def caught_up():
        stable = 0
        while stable < 2:  # hold for two polls to cover the moment between dequeue and decode
            await asyncio.sleep(0.03)
            stable = stable + 1 if processor.transcription_queue.empty() \
                and busy[0] == 0 else 0

    async def feed():
        for i in range(0, len(raw), step):
            await processor.process_audio(raw[i:i + step])
            if realtime:
                await asyncio.sleep(CHUNK_SAMPLES / SAMPLE_RATE)
            else:
                await caught_up()
        await processor.process_audio(b"")

    words, lines, emitted = {}, {}, []
    state = {"emitted": {}, "preview": ""}

    async def consume():
        async for response in results:
            msg = response.to_dict() if hasattr(response, "to_dict") else response
            # The alignment keeps a rolling window, so collect as we go.
            for tok in processor.tokens_alignment.all_tokens:
                text = getattr(tok, "text", "") or ""
                if text.strip() and not tok.is_silence():
                    words[(round(tok.start, 3), text)] = {
                        "start": round(float(tok.start), 3),
                        "end": round(float(tok.end), 3), "text": text}
            for line in msg.get("lines", []):
                if (line.get("text") or "").strip():
                    lines[line["start"]] = {"start": line["start"],
                                            "end": line["end"],
                                            "text": line["text"].strip()}
            emitted.extend(e.text for e in _events_from_response(msg, state)
                           if e.committed)

    feeder = asyncio.create_task(feed())
    complete = True
    try:
        await asyncio.wait_for(consume(), timeout=MAX_RUN_S)
    except asyncio.TimeoutError:
        complete = False  # WLK never finished; the result is partial
    feeder.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await feeder
    with contextlib.suppress(Exception):
        await processor.cleanup()
    engine.asr.transcribe = decode
    return (sorted(words.values(), key=lambda w: (w["start"], w["end"])),
            list(lines.values()), emitted, complete)


def run(composite, model, overrides, realtime, out):
    composite = Path(composite)
    variant, language = composite.parent.parent.name, composite.parent.name
    os.environ["CODA_DIALOGUE__TRANSCRIBER_BACKEND"] = "whisper-livekit"
    os.environ["CODA_DIALOGUE__LANGUAGE"] = asr_language(language)
    for key, value in overrides.items():
        os.environ[ENV_PREFIX + key.upper()] = value
    from coda.dialogue import create_transcriber  # after the env is set
    from coda.config import settings

    transcriber = create_transcriber("whisper-livekit", model=model)
    pcm = audio.read_wav(composite)
    started, started_awake = time.time(), time.monotonic()
    words, lines, emitted, complete = asyncio.run(
        stream(transcriber._engine, pcm, realtime))
    wall = time.time() - started
    # The monotonic clock stops while the machine sleeps and the wall clock
    # does not, so a gap between them means the laptop slept mid-run: the
    # timing is void, so mark the run incomplete to be redone on resume.
    slept = wall - (time.monotonic() - started_awake)
    if slept > 2.0:
        complete = False
    result = {
        "composite": str(composite.relative_to(HERE)),
        "map": str(composite.with_suffix(".json").relative_to(HERE)),
        "complete": complete,
        # Not complete and the machine did not sleep: the transcriber fell
        # behind the audio and never finished (real time) or hit the limit.
        "timed_out": not complete and slept <= 2.0,
        "variant": variant, "language": language,
        "asr_language": os.environ["CODA_DIALOGUE__LANGUAGE"], "model": model,
        "config": {"vac": settings.dialogue.whisper_livekit.vac,
                   "decode_options": transcriber.decode_options},
        "overrides": overrides,
        "feed": "realtime" if realtime else "lockstep",
        "wall_seconds": round(wall, 1),
        "slept_seconds": round(max(slept, 0.0), 1),
        "audio_seconds": round(audio.seconds(pcm), 3),
        "words": words, "lines": lines, "emitted": " ".join(emitted),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("composite", help="data/benchmark/<variant>/<lang>/<case>.wav")
    parser.add_argument("--config", default="baseline",
                        help="name for this configuration; results go under it")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override under dialogue.whisper_livekit, e.g. "
                             "VAC=false or DECODE_OPTIONS__NO_SPEECH_THRESHOLD=0.4; "
                             "@none disables a threshold")
    parser.add_argument("--model", default="small", help="the app default")
    parser.add_argument("--realtime", action="store_true",
                        help="feed at live speed to measure whether this machine "
                             "keeps pace, instead of lock-step with processing")
    parser.add_argument("--out", default=None, help="result JSON path")
    args = parser.parse_args()

    composite = Path(args.composite).resolve()
    out = Path(args.out) if args.out else (
        RESULTS_DIR / ("realtime" if args.realtime else "lockstep") / args.config
        / composite.parent.parent.name / composite.parent.name
        / composite.with_suffix(".json").name)
    overrides = dict(item.split("=", 1) for item in args.set)
    result = run(composite, args.model, overrides, args.realtime, out)
    print(f"{result['variant']}/{result['language']}/{composite.stem}: "
          f"{len(result['words'])} words in {result['wall_seconds']}s wall "
          f"for {result['audio_seconds']}s audio -> {out}")


if __name__ == "__main__":
    main()
