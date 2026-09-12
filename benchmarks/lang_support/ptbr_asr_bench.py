"""Benchmark Whisper ASR on the Brazilian Portuguese COD clips (WER vs narrative).

Each clip's reference is the matching case's Portuguese narrative in
cases_ptbr_filtered.json. Two backends are supported, openai-whisper and
faster-whisper, both run with language="pt". Word-level WER is computed
in-process (no jiwer). Run with no args for all engines, or pass engine names.
"""
import argparse
import os
import subprocess
from pathlib import Path

from dataset_io import load_samples
from engines import make_faster_whisper, make_whisper
from languages.pt_br import normalize as normalize_portuguese
from metrics import wer_details as metric_wer_details

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BASE = Path(__file__).resolve().parent
SIZES = ["tiny", "base", "small", "medium", "large-v3"]


def hardware():
    def sc(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True,
                text=True).stdout.strip()
        except Exception:
            return ""
    mem = sc("hw.memsize")
    return {"chip": sc("machdep.cpu.brand_string"),
            "ram_gb": round(int(mem) / 1024 ** 3) if mem.isdigit() else None}


def clip_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return None


# Shared edit-distance accounting keeps the two language runners comparable.
def wer(ref, hyp, strip_accents=False):
    return metric_wer_details(
        ref, hyp, lambda text: normalize_portuguese(text, strip_accents)
    )[0]


def samples():
    return [(s.case_id, str(s.audio_path), s.reference, clip_duration(s.audio_path))
            for s in load_samples("pt-BR")]


def build_engines(args):
    engines = {}
    for size in SIZES:
        engines[f"whisper-{size}"] = \
            lambda size=size: make_whisper(size, args.language, args.device)
        engines[f"faster-whisper-{size}"] = \
            lambda size=size: make_faster_whisper(size, args.language,
                                                  args.fw_device, args.compute_type)
    return engines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*", default=None,
                    help="engine names to run (default: all)")
    ap.add_argument("--language", default="pt", help="Whisper language code")
    ap.add_argument("--device", default="cpu",
                    help="torch device for openai-whisper")
    ap.add_argument("--fw-device", default="cpu",
                    help="ctranslate2 device for faster-whisper")
    # float32 keeps faster-whisper comparable to openai-whisper, "auto" would
    # silently pick int8 on CPU
    ap.add_argument("--compute-type", default="float32",
                    help="faster-whisper compute type, e.g. int8 or float32")
    ap.add_argument("--strip-accents", action="store_true",
                    help="ignore diacritics when scoring")
    args = ap.parse_args()
    from run_benchmark import run
    return run(
        "pt-BR", args.engines or None,
        strip_accents=args.strip_accents,
        device=args.device, fw_device=args.fw_device,
        compute_type=args.compute_type,
    )


if __name__ == "__main__":
    main()
