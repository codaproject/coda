"""Run one language-support ASR benchmark with shared scoring and reporting."""
import argparse
import importlib
import json
import time
from pathlib import Path

from dataset_io import load_samples
from metrics import cer, wer_details


LANGUAGES = {
    "bn": ("languages.bn", "bangla_asr_bench"),
    "pt-BR": ("languages.pt_br", "ptbr_asr_bench"),
}


def run(language, engines=None, *, strip_accents=False, **options):
    """Run selected engines and write results under results/<language>."""
    try:
        normalizer = importlib.import_module(LANGUAGES[language][0])
        engine_module = importlib.import_module(LANGUAGES[language][1])
    except KeyError as exc:
        raise ValueError(f"Unknown language {language!r}; choose {list(LANGUAGES)}") from exc

    samples = load_samples(language)
    args = argparse.Namespace(
        language="pt" if language == "pt-BR" else "bn",
        device=options.get("device", "cpu"),
        fw_device=options.get("fw_device", "cpu"),
        compute_type=options.get("compute_type", "float32"),
    )
    registry = engine_module.build_engines(args) if hasattr(engine_module, "build_engines") else engine_module.ENGINES
    selected = engines or list(registry)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"Unknown engines {unknown}; available: {list(registry)}")

    output_dir = Path(__file__).parent / "results" / language
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in selected:
        started = time.time()
        transcribe = registry[name]()
        load_sec = round(time.time() - started, 1)
        clips = []
        for sample in samples:
            started = time.time()
            hypothesis = transcribe(str(sample.audio_path))
            elapsed = time.time() - started
            normalize = (lambda text: normalizer.normalize(text, strip_accents)) \
                if language == "pt-BR" else normalizer.normalize
            w, s, d, i, n, accuracy = wer_details(sample.reference, hypothesis, normalize)
            c = cer(sample.reference, hypothesis, normalize)
            duration = engine_module.clip_duration(sample.audio_path)
            rtf = elapsed / duration if duration else None
            clips.append({"case_id": sample.case_id, "wer": round(w, 3), "cer": round(c, 3),
                          "S": s, "D": d, "I": i, "N": n, "accuracy": round(accuracy, 3),
                          "audio_sec": round(duration, 1) if duration else None,
                          "time_sec": round(elapsed, 2), "rtf": round(rtf, 3) if rtf else None,
                          "ref": sample.reference, "hyp": hypothesis,
                          "ref_norm": normalize(sample.reference),
                          "hyp_norm": normalize(hypothesis)})
        (output_dir / f"transcripts_{name}.json").write_text(
            json.dumps({"engine": name, "language": language, "load_sec": load_sec,
                        "clips": clips}, ensure_ascii=False, indent=2) + "\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument("engines", nargs="*")
    parser.add_argument("--strip-accents", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fw-device", default="cpu")
    parser.add_argument("--compute-type", default="float32")
    args = parser.parse_args()
    run(args.language, args.engines or None, strip_accents=args.strip_accents,
        device=args.device, fw_device=args.fw_device, compute_type=args.compute_type)


if __name__ == "__main__":
    main()
