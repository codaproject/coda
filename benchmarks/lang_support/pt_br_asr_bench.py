"""Benchmark Whisper ASR engines on the Brazilian Portuguese COD clips.

Two backends are supported, openai-whisper and faster-whisper. Scoring,
reporting and dataset loading are shared, see run_benchmark and languages.pt_br.
Run with no args for all engines, or pass engine names.
"""
import os

from engines import make_faster_whisper, make_whisper
from run_benchmark import main_for

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

SIZES = ["tiny", "base", "small", "medium", "large-v3"]


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
    return main_for("pt_br", build_engines)


if __name__ == "__main__":
    main()
