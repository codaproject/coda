"""
Summarize ASR benchmark results across models and languages.

Reads all asr_benchmark_*.csv files from the pystow results directory,
extracts the __AGGREGATE__ rows, and produces:
  1. A per-language-per-model summary table
  2. A per-model runtime total table
  3. A per-language runtime comparison table
  4. Saves a combined summary CSV

Usage:
    python summarize_results.py

    # Show only models that have finished
    python summarize_results.py --models whisper-tiny whisper-base whisper-small

    # Save summary to a specific path
    python summarize_results.py --out_csv my_summary.csv
"""

import argparse
import csv
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import pystow

RESULTS_DIR = pystow.module("coda", "benchmarks", "kaggle_ambient_scribe").join("results")

LANGUAGES = ["en", "nl", "fr", "de", "es"]
LANGUAGE_NAMES = {"en": "English", "nl": "Dutch", "fr": "French", "de": "German", "es": "Spanish"}

# Metrics to show in summary
METRICS = ["encounters", "duration_s", "latency_s", "rtf", "WER", "CER"]


def find_csvs(results_dir: Path, model_filter: Optional[List[str]] = None) -> List[Path]:
    csvs = sorted(results_dir.glob("asr_benchmark_*.csv"))
    if model_filter:
        csvs = [f for f in csvs if any(m in f.name for m in model_filter)]
    return csvs


def parse_aggregate_row(csv_path: Path) -> Optional[Dict]:
    """Extract the __AGGREGATE__ row from a results CSV."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        if row.get("audio_file") == "__AGGREGATE__":
            # Parse filename to get language and model
            # Format: asr_benchmark_{lang}.{model}.csv
            stem = csv_path.stem  # e.g. asr_benchmark_nl.whisper-tiny
            parts = stem.replace("asr_benchmark_", "").split(".", 1)
            lang = parts[0]
            model = parts[1] if len(parts) > 1 else "unknown"

            return {
                "model": model,
                "language": lang,
                "language_name": LANGUAGE_NAMES.get(lang, lang),
                "encounters": sum(1 for r in rows if r.get("audio_file") != "__AGGREGATE__"),
                "duration_s": float(row.get("duration_s") or 0),
                "latency_s": float(row.get("latency_s") or 0),
                "rtf": float(row.get("rtf") or 0),
                "WER": float(row.get("WER") or 0),
                "CER": float(row.get("CER") or 0),
            }
    return None


def fmt(val, decimals=3):
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:04.1f}s"
    elif m > 0:
        return f"{m}m {s:04.1f}s"
    else:
        return f"{s:.1f}s"


def print_table(headers: List[str], rows: List[List], title: str = "") -> None:
    if title:
        print(f"\n{'─' * 72}")
        print(f"  {title}")
        print(f"{'─' * 72}")

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    fmt_row = lambda cells: "  " + "  ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells))

    print(fmt_row(headers))
    print("  " + "  ".join("─" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def main():
    ap = argparse.ArgumentParser(description="Summarize ASR benchmark results.")
    ap.add_argument(
        "--models", nargs="+", default=None,
        help="Filter to specific model name substrings, e.g. --models whisper-tiny whisper-small"
    )
    ap.add_argument(
        "--out_csv", default=None,
        help="Save combined summary to this CSV path."
    )
    args = ap.parse_args()

    csvs = find_csvs(RESULTS_DIR, args.models)
    if not csvs:
        print(f"No result CSVs found in {RESULTS_DIR}")
        return

    # Parse all aggregate rows
    records = []
    for csv_path in csvs:
        row = parse_aggregate_row(csv_path)
        if row:
            records.append(row)
        else:
            print(f"[WARN] No __AGGREGATE__ row found in {csv_path.name}")

    if not records:
        print("No aggregate rows found.")
        return

    # Sort by model then language
    model_order = ["whisper-tiny", "whisper-base", "whisper-small",
                   "whisper-medium", "whisper-large-v3-turbo", "whisper-large-v3"]
    records.sort(key=lambda r: (
        model_order.index(r["model"]) if r["model"] in model_order else 99,
        LANGUAGES.index(r["language"]) if r["language"] in LANGUAGES else 99,
    ))

    # ── Table 1: Full summary ─────────────────────────────────────────────
    headers = ["model", "lang", "encounters", "audio_dur", "latency", "rtf", "WER", "CER"]
    rows = []
    for r in records:
        rows.append([
            r["model"],
            r["language_name"],
            r["encounters"],
            seconds_to_hms(r["duration_s"]),
            seconds_to_hms(r["latency_s"]),
            fmt(r["rtf"], 3),
            fmt(r["WER"], 4),
            fmt(r["CER"], 4),
        ])
    print_table(headers, rows, title="Full Results — All Models × Languages")

    # ── Table 2: Per-model total runtime ──────────────────────────────────
    model_totals = defaultdict(lambda: {"latency_s": 0, "duration_s": 0, "languages": 0})
    for r in records:
        model_totals[r["model"]]["latency_s"] += r["latency_s"]
        model_totals[r["model"]]["duration_s"] += r["duration_s"]
        model_totals[r["model"]]["languages"] += 1

    headers2 = ["model", "languages_run", "total_audio", "total_runtime", "avg_rtf"]
    rows2 = []
    for model in model_order:
        if model not in model_totals:
            continue
        t = model_totals[model]
        avg_rtf = t["latency_s"] / t["duration_s"] if t["duration_s"] else 0
        rows2.append([
            model,
            t["languages"],
            seconds_to_hms(t["duration_s"]),
            seconds_to_hms(t["latency_s"]),
            fmt(avg_rtf, 3),
        ])
    print_table(headers2, rows2, title="Per-Model Total Runtime")

    # ── Table 3: Per-language runtime across models ───────────────────────
    lang_records = defaultdict(list)
    for r in records:
        lang_records[r["language"]].append(r)

    for lang in LANGUAGES:
        if lang not in lang_records:
            continue
        lang_rows = lang_records[lang]
        headers3 = ["model", "encounters", "audio_dur", "latency", "rtf", "WER", "CER"]
        rows3 = []
        for r in sorted(lang_rows, key=lambda x: model_order.index(x["model"]) if x["model"] in model_order else 99):
            rows3.append([
                r["model"],
                r["encounters"],
                seconds_to_hms(r["duration_s"]),
                seconds_to_hms(r["latency_s"]),
                fmt(r["rtf"], 3),
                fmt(r["WER"], 4),
                fmt(r["CER"], 4),
            ])
        print_table(headers3, rows3, title=f"{LANGUAGE_NAMES[lang]} — Runtime × Model")

    # ── Table 4: WER comparison across models (compact) ──────────────────
    all_models = [m for m in model_order if m in model_totals]
    headers4 = ["model"] + [LANGUAGE_NAMES[l] for l in LANGUAGES if l in lang_records] + ["avg_WER"]
    rows4 = []
    for model in all_models:
        row_wers = []
        model_recs = {r["language"]: r for r in records if r["model"] == model}
        wer_vals = []
        for lang in LANGUAGES:
            if lang in model_recs:
                w = model_recs[lang]["WER"]
                row_wers.append(fmt(w, 4))
                wer_vals.append(w)
            else:
                row_wers.append("—")
        avg = sum(wer_vals) / len(wer_vals) if wer_vals else 0
        rows4.append([model] + row_wers + [fmt(avg, 4)])
    print_table(headers4, rows4, title="WER Comparison Across Models")

    # ── Save summary CSV ──────────────────────────────────────────────────
    if args.out_csv:
        out_path = Path(args.out_csv)
    else:
        out_path = RESULTS_DIR.parent / "summary.csv"

    fieldnames = ["model", "language", "language_name", "encounters",
                  "duration_s", "latency_s", "rtf", "WER", "CER"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in fieldnames})

    print(f"\n  ✓ Summary CSV saved → {out_path}\n")


if __name__ == "__main__":
    main()