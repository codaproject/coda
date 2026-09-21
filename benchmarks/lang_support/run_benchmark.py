"""Run one language-support ASR benchmark with shared scoring and reporting."""
import argparse
import importlib
import json
import time
from pathlib import Path

from dataset_io import clip_duration, load_samples
from metrics import cer, wer_details
from reporting import hardware

# Dataset tag to the module holding its normalization and ASR language code
LANGUAGES = {"bn": "languages.bn", "pt_br": "languages.pt_br",
             "ts": "languages.ts"}

# Engines backed by a rate-limited remote API reject requests intermittently.
# Retry here rather than inside an engine so each attempt is timed separately.
ATTEMPTS = 4
RETRY_WAIT = 5


def transcribe_clip(transcribe, path):
    """Return (text, seconds) for one clip, retrying transient empty results.

    An engine that fails internally returns empty text rather than raising, which
    would otherwise score as a complete mis-transcription. Only the successful
    attempt is timed, so a retry never inflates RTF.
    """
    last = None
    for attempt in range(ATTEMPTS):
        started = time.time()
        try:
            text = transcribe(str(path))
        except Exception as exc:
            last = exc
        else:
            elapsed = time.time() - started
            if text.strip():
                return text, elapsed
            last = RuntimeError("empty transcript")
        if attempt < ATTEMPTS - 1:
            time.sleep(RETRY_WAIT * (attempt + 1))
    raise RuntimeError(f"no transcript after {ATTEMPTS} attempts: {last}")


def run(language, registry, engines=None, *, strip_accents=False, **options):
    """Run selected engines and write results under results/<language>."""
    try:
        adapter = importlib.import_module(LANGUAGES[language])
    except KeyError as exc:
        raise ValueError(
            f"Unknown language {language!r}; choose {list(LANGUAGES)}") from exc

    selected = engines or list(registry)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"Unknown engines {unknown}; available: {list(registry)}")

    def normalize(text):
        return adapter.normalize(text, strip_accents)

    samples = load_samples(language)
    durations = {s.case_id: clip_duration(s.audio_path) for s in samples}
    host = hardware()
    compute_type = options.get("compute_type")
    output_dir = Path(__file__).parent / "results" / language
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hardware: {host['chip']} {host['ram_gb']}GB  language={language}  "
          f"clips={len(samples)}", flush=True)
    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        started = time.time()
        try:
            transcribe = registry[name]()
        except Exception as exc:
            print(f"  engine load failed: {str(exc)[:150]}", flush=True)
            continue
        load_sec = round(time.time() - started, 1)

        clips = []
        failed = []
        for sample in samples:
            duration = durations[sample.case_id]
            try:
                hypothesis, elapsed = transcribe_clip(
                    transcribe, sample.audio_path)
            except Exception as exc:
                failed.append(sample.case_id)
                print(f"  {sample.case_id:<14} FAILED {str(exc)[:90]}", flush=True)
                continue
            w, s, d, i, n, accuracy = wer_details(
                sample.reference, hypothesis, normalize)
            c = cer(sample.reference, hypothesis, normalize)
            rtf = elapsed / duration if duration else None
            clips.append({"case_id": sample.case_id, "wer": round(w, 3),
                          "cer": round(c, 3), "S": s, "D": d, "I": i, "N": n,
                          "accuracy": round(accuracy, 3),
                          "audio_sec": round(duration, 1) if duration else None,
                          "time_sec": round(elapsed, 2),
                          "rtf": round(rtf, 3) if rtf else None,
                          "ref": sample.reference, "hyp": hypothesis,
                          "ref_norm": normalize(sample.reference),
                          "hyp_norm": normalize(hypothesis)})
            suffix = f"  {elapsed:5.1f}s  RTF={rtf:.2f}" if rtf else ""
            print(f"  {sample.case_id:<14} WER={w:.3f} CER={c:.3f}{suffix}",
                  flush=True)

        if not clips:
            continue
        mean_wer = sum(clip["wer"] for clip in clips) / len(clips)
        mean_cer = sum(clip["cer"] for clip in clips) / len(clips)
        rtfs = [clip["rtf"] for clip in clips if clip["rtf"] is not None]
        mean_rtf = sum(rtfs) / len(rtfs) if rtfs else None
        tail = f"  mean_RTF={mean_rtf:.2f}" if mean_rtf else ""
        note = f"  FAILED={len(failed)}" if failed else ""
        print(f"  MEAN WER={mean_wer:.3f}  MEAN CER={mean_cer:.3f}  "
              f"load={load_sec}s{tail}  (n={len(clips)}){note}", flush=True)

        path = output_dir / f"transcripts_{name}.json"
        path.write_text(json.dumps(
            {"engine": name, "language": language, "hardware": host,
             "compute_type": compute_type, "strip_accents": strip_accents,
             "load_sec": load_sec, "failed": failed, "clips": clips},
            ensure_ascii=False, indent=2) + "\n")
        print(f"  transcripts -> {path}", flush=True)


def build_parser(description=None):
    """Build the argument parser shared by every language entry point."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("engines", nargs="*",
                        help="engine names to run (default: all)")
    parser.add_argument("--strip-accents", action="store_true",
                        help="ignore diacritics when scoring")
    parser.add_argument("--device", default="cpu",
                        help="torch device for openai-whisper")
    parser.add_argument("--fw-device", default="cpu",
                        help="ctranslate2 device for faster-whisper")
    # float32 keeps faster-whisper comparable to openai-whisper, "auto" would
    # silently pick int8 on CPU
    parser.add_argument("--compute-type", default="float32",
                        help="faster-whisper compute type, e.g. int8 or float32")
    return parser


def main_for(language, build_engines):
    """Entry point for a language runner that supplies its own engine registry."""
    args = build_parser().parse_args()
    args.language = importlib.import_module(LANGUAGES[language]).ASR_LANGUAGE
    return run(language, build_engines(args), args.engines or None,
               strip_accents=args.strip_accents, device=args.device,
               fw_device=args.fw_device, compute_type=args.compute_type)


RUNNERS = {"bn": "bn_asr_bench", "pt_br": "pt_br_asr_bench",
           "ts": "ts_asr_bench"}


def main():
    parser = build_parser()
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    args = parser.parse_args()
    language = args.language
    build_engines = importlib.import_module(RUNNERS[language]).build_engines
    args.language = importlib.import_module(LANGUAGES[language]).ASR_LANGUAGE
    return run(language, build_engines(args), args.engines or None,
               strip_accents=args.strip_accents, device=args.device,
               fw_device=args.fw_device, compute_type=args.compute_type)


if __name__ == "__main__":
    main()
