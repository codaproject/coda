"""Plot and summarize multilingual ASR benchmark results.

Reads the result CSVs written by benchmark_asr.py
(asr_benchmark_results.{backend}.{language}.{size}.csv) from the pystow results
directory and produces:
  - A printed summary and a combined summary.csv (from the aggregate rows)
  - Per-language plots of WER and RTF comparing whisper vs faster-whisper
    across model sizes
  - Overall (mean-over-languages) whisper vs faster-whisper plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pystow

RESULTS_DIR = pystow.module("coda", "benchmarks", "kaggle_ambient_scribe").join("results")

BACKENDS = ["whisper", "faster-whisper"]
SIZES = ["tiny", "base", "small", "medium", "large", "large-v2"]
LANGUAGES = ["en", "nl", "fr", "de", "es"]
LANGUAGE_NAMES = {"en": "English", "nl": "Dutch", "fr": "French",
                  "de": "German", "es": "Spanish"}

# One colour per backend for the whisper-vs-faster-whisper comparison
BACKEND_COLORS = {"whisper": "#4C72B0", "faster-whisper": "#C44E52"}


def load_results(results_dir: Path):
    """Load per-encounter rows and aggregate rows from all result CSVs."""
    enc_frames = []
    agg_records = []
    for backend in BACKENDS:
        for lang in LANGUAGES:
            for size in SIZES:
                path = results_dir / \
                    f"asr_benchmark_results.{backend}.{lang}.{size}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                agg = df[df["audio_file"] == "__AGGREGATE__"]
                enc = df[df["audio_file"] != "__AGGREGATE__"].copy()
                enc["backend"] = backend
                enc["size"] = size
                enc["language"] = lang
                enc_frames.append(
                    enc[["backend", "size", "language", "WER", "CER", "rtf"]])
                if not agg.empty:
                    a = agg.iloc[0]
                    agg_records.append({
                        "backend": backend,
                        "size": size,
                        "language": lang,
                        "language_name": LANGUAGE_NAMES[lang],
                        "encounters": len(enc),
                        "duration_s": float(a["duration_s"]),
                        "latency_s": float(a["latency_s"]),
                        "rtf": float(a["rtf"]),
                        "WER": float(a["WER"]),
                        "CER": float(a["CER"]),
                    })

    if not enc_frames:
        raise FileNotFoundError(
            f"No result CSVs found in {results_dir}.\n"
            f"Expected files like: asr_benchmark_results.whisper.en.tiny.csv"
        )

    return pd.concat(enc_frames, ignore_index=True), pd.DataFrame(agg_records)


def _sizes_present(df: pd.DataFrame) -> list:
    """Return SIZES filtered to those actually in the dataframe, in order."""
    return [s for s in SIZES if s in df["size"].unique()]


def seconds_to_hms(seconds: float) -> str:
    """Format a duration in seconds as a compact h/m/s string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:04.1f}s"
    if m > 0:
        return f"{m}m {s:04.1f}s"
    return f"{s:.1f}s"


# -------------------------------
# Summary tables (from aggregate rows)
# -------------------------------

def print_summary(agg_df: pd.DataFrame) -> None:
    """Print a per-backend-per-model-per-language summary and a WER comparison."""
    if agg_df.empty:
        return

    disp = agg_df.copy()
    disp["audio"] = disp["duration_s"].map(seconds_to_hms)
    disp["runtime"] = disp["latency_s"].map(seconds_to_hms)
    disp["backend"] = pd.Categorical(disp["backend"], categories=BACKENDS, ordered=True)
    disp["size"] = pd.Categorical(disp["size"], categories=SIZES, ordered=True)
    disp["language"] = pd.Categorical(disp["language"], categories=LANGUAGES, ordered=True)
    disp = disp.sort_values(["backend", "size", "language"])

    print("\n=== Results by backend, model, language ===")
    cols = ["backend", "size", "language_name", "encounters",
            "audio", "runtime", "rtf", "WER", "CER"]
    print(disp[cols].to_string(index=False))

    sizes = _sizes_present(agg_df)
    for metric in ("WER", "rtf"):
        print(f"\n=== Mean {metric} over languages (whisper vs faster-whisper) ===")
        piv = agg_df.pivot_table(index="size", columns="backend",
                                 values=metric, aggfunc="mean")
        piv = piv.reindex(index=[s for s in sizes if s in piv.index],
                          columns=[b for b in BACKENDS if b in piv.columns])
        print(piv.round(4).to_string())


def save_summary_csv(agg_df: pd.DataFrame, out_path: Path) -> None:
    """Save the aggregate summary to a single CSV."""
    cols = ["backend", "size", "language", "language_name", "encounters",
            "duration_s", "latency_s", "rtf", "WER", "CER"]
    agg_df[cols].to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")


# -------------------------------
# Plots (whisper vs faster-whisper)
# -------------------------------

def _plot_backend_lines(ax, agg_df, metric, sizes):
    """Plot one line per backend (metric vs size) on the given axes."""
    for backend in BACKENDS:
        sub = agg_df[agg_df["backend"] == backend]
        if sub.empty:
            continue
        xs, ys = [], []
        for i, size in enumerate(sizes):
            row = sub[sub["size"] == size]
            if not row.empty:
                xs.append(i)
                ys.append(float(row.iloc[0][metric]))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=backend,
                    color=BACKEND_COLORS[backend])
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("model size")
    ax.grid(alpha=0.3)


def plot_by_language(agg_df, metric, ylabel, title, output_path):
    """Per-language plots of a metric, whisper vs faster-whisper across sizes."""
    sizes = _sizes_present(agg_df)
    langs = [l for l in LANGUAGES if l in agg_df["language"].unique()]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes_flat = axes.flatten()
    for idx, lang in enumerate(langs):
        ax = axes_flat[idx]
        _plot_backend_lines(ax, agg_df[agg_df["language"] == lang], metric, sizes)
        ax.set_title(LANGUAGE_NAMES[lang])
        ax.set_ylabel(ylabel)
        ax.legend()
    for idx in range(len(langs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.show()


def plot_overall(agg_df, metric, ylabel, title, output_path):
    """Overall metric (mean over languages), whisper vs faster-whisper."""
    sizes = _sizes_present(agg_df)
    means = (agg_df.groupby(["backend", "size"])[metric].mean()
             .reset_index())
    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_backend_lines(ax, means, metric, sizes)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Plot and summarize multilingual ASR benchmark results "
                    "(whisper vs faster-whisper).")
    ap.add_argument("--out_dir", default=None,
                    help="Directory for plots and summary CSV (default: results dir).")
    ap.add_argument("--no_show", action="store_true",
                    help="Save figures without displaying them.")
    args = ap.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    _enc_df, agg_df = load_results(RESULTS_DIR)

    print_summary(agg_df)
    save_summary_csv(agg_df, out_dir / "summary.csv")

    plot_by_language(agg_df, "WER", "Word error rate (WER)",
                     "WER by model size: whisper vs faster-whisper",
                     out_dir / "wer_by_language.png")
    plot_by_language(agg_df, "rtf", "Real-time factor (RTF)",
                     "RTF by model size: whisper vs faster-whisper",
                     out_dir / "rtf_by_language.png")
    plot_overall(agg_df, "WER", "Mean WER over languages",
                 "Overall WER: whisper vs faster-whisper",
                 out_dir / "wer_overall.png")
    plot_overall(agg_df, "rtf", "Mean RTF over languages",
                 "Overall RTF: whisper vs faster-whisper",
                 out_dir / "rtf_overall.png")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
