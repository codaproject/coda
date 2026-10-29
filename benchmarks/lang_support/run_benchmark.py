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
             "ts": "languages.ts", "zu": "languages.zu", "st": "languages.st",
             "syl": "languages.syl"}

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


def load_adapter(language):
    try:
        return importlib.import_module(LANGUAGES[language])
    except KeyError as exc:
        raise ValueError(
            f"Unknown language {language!r}; choose {list(LANGUAGES)}") from exc


def score(adapter, case_id, reference, hypothesis, strip_accents):
    """Return the scored fields of one clip.

    Words before the first reference word are not counted for a recording that
    opens with a spoken title missing from its reference. A recording read from
    a different translation is scored but marked excluded, keeping it out of the
    means.
    """
    def normalize(text):
        return adapter.normalize(text, strip_accents)

    lead_in = case_id in getattr(adapter, "SPOKEN_TITLE", ())
    w, s, d, i, n, accuracy = wer_details(reference, hypothesis, normalize,
                                          lead_in)
    c = cer(reference, hypothesis, normalize, lead_in)
    return {"wer": round(w, 3), "cer": round(c, 3), "S": s, "D": d, "I": i,
            "N": n, "accuracy": round(accuracy, 3),
            "excluded": case_id in getattr(adapter, "MISMATCHED", ())}


def label(sample):
    return f"{sample.case_id}/{sample.recording_set}" \
        if sample.recording_set else sample.case_id


def means(clips):
    """Return mean WER, mean CER, and the number of scored clips not excluded."""
    scored = [clip for clip in clips
              if "wer" in clip and not clip.get("excluded")]
    if not scored:
        return float("nan"), float("nan"), 0
    return (sum(clip["wer"] for clip in scored) / len(scored),
            sum(clip["cer"] for clip in scored) / len(scored), len(scored))


def run(language, registry, engines=None, *, strip_accents=False, **options):
    """Run selected engines and write results under results/<language>."""
    adapter = load_adapter(language)

    selected = engines or list(registry)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"Unknown engines {unknown}; available: {list(registry)}")

    def normalize(text):
        return adapter.normalize(text, strip_accents)

    samples = load_samples(language)
    durations = {s.audio_path: clip_duration(s.audio_path) for s in samples}
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
            duration = durations[sample.audio_path]
            try:
                hypothesis, elapsed = transcribe_clip(
                    transcribe, sample.audio_path)
            except Exception as exc:
                failed.append(label(sample))
                print(f"  {label(sample):<16} FAILED {str(exc)[:90]}", flush=True)
                continue
            rtf = elapsed / duration if duration else None
            clip = {"case_id": sample.case_id}
            if sample.recording_set:
                clip["recording_set"] = sample.recording_set
            if sample.reference is not None:
                clip.update(score(adapter, sample.case_id, sample.reference,
                                  hypothesis, strip_accents))
            clip.update({"audio_sec": round(duration, 1) if duration else None,
                         "time_sec": round(elapsed, 2),
                         "rtf": round(rtf, 3) if rtf else None})
            if sample.reference is not None:
                clip["ref"] = sample.reference
            clip["hyp"] = hypothesis
            if sample.reference is not None:
                clip["ref_norm"] = normalize(sample.reference)
            clip["hyp_norm"] = normalize(hypothesis)
            clips.append(clip)

            prefix = f"WER={clip['wer']:.3f} CER={clip['cer']:.3f}" \
                if "wer" in clip else "unscored"
            suffix = f"  {elapsed:5.1f}s  RTF={rtf:.2f}" if rtf else ""
            if clip.get("excluded"):
                suffix += "  (excluded)"
            print(f"  {label(sample):<16} {prefix}{suffix}", flush=True)

        if not clips:
            continue
        rtfs = [clip["rtf"] for clip in clips if clip["rtf"] is not None]
        mean_rtf = sum(rtfs) / len(rtfs) if rtfs else None
        tail = f"  mean_RTF={mean_rtf:.2f}" if mean_rtf else ""
        note = f"  FAILED={len(failed)}" if failed else ""
        print(f"  {summary(clips)}  load={load_sec}s{tail}{note}", flush=True)

        path = output_dir / f"transcripts_{name}.json"
        path.write_text(json.dumps(
            {"engine": name, "language": language, "hardware": host,
             "compute_type": compute_type, "strip_accents": strip_accents,
             "load_sec": load_sec, "failed": failed, "clips": clips},
            ensure_ascii=False, indent=2) + "\n")
        print(f"  transcripts -> {path}", flush=True)


def summary(clips):
    """Describe mean WER and CER, and per recording set when there are several."""
    mean_wer, mean_cer, scored = means(clips)
    if not scored:
        return f"unscored, no references (n={len(clips)})"
    text = f"MEAN WER={mean_wer:.3f}  MEAN CER={mean_cer:.3f}  (n={scored})"
    sets = sorted({clip["recording_set"] for clip in clips
                   if clip.get("recording_set")})
    for set_name in sets if len(sets) > 1 else ():
        set_wer, set_cer, set_n = means(
            [clip for clip in clips if clip.get("recording_set") == set_name])
        text += f"\n    {set_name:<6} WER={set_wer:.3f}  CER={set_cer:.3f}  " \
            f"(n={set_n})"
    return text


def rescore(language):
    """Rescore saved transcripts in place against the current references."""
    adapter = load_adapter(language)
    samples = {(s.case_id, s.recording_set): s
               for s in load_samples(language)}
    for path in sorted((Path(__file__).parent / "results" / language)
                       .glob("transcripts_*.json")):
        result = json.loads(path.read_text())
        strip_accents = result.get("strip_accents", False)
        for clip in result["clips"]:
            sample = samples[clip["case_id"], clip.get("recording_set", "")]
            if sample.reference is None:
                continue
            clip.update(score(adapter, sample.case_id, sample.reference,
                              clip["hyp"], strip_accents),
                        ref=sample.reference,
                        ref_norm=adapter.normalize(sample.reference,
                                                   strip_accents),
                        hyp_norm=adapter.normalize(clip["hyp"],
                                                   strip_accents))
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"{result['engine']:<28} {summary(result['clips'])}")


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
           "ts": "ts_asr_bench", "zu": "za_asr_bench", "st": "za_asr_bench",
           "syl": "bn_asr_bench"}


def main():
    parser = build_parser()
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument("--rescore", action="store_true",
                        help="rescore saved transcripts instead of running engines")
    args = parser.parse_args()
    language = args.language
    if args.rescore:
        return rescore(language)
    build_engines = importlib.import_module(RUNNERS[language]).build_engines
    args.language = importlib.import_module(LANGUAGES[language]).ASR_LANGUAGE
    return run(language, build_engines(args), args.engines or None,
               strip_accents=args.strip_accents, device=args.device,
               fw_device=args.fw_device, compute_type=args.compute_type)


if __name__ == "__main__":
    main()
