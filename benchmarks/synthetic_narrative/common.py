"""Shared paths for the hallucination benchmark.

Importing this module puts benchmarks/lang_support on sys.path, so its
reference loaders can be reused when running a script here from the repo root.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANG_SUPPORT = HERE.parent / "lang_support"

SAMPLE_RATE = 16000

if str(LANG_SUPPORT) not in sys.path:
    sys.path.insert(0, str(LANG_SUPPORT))
