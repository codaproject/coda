"""
Box plots of WER, CER, and RTF across Whisper model sizes for all
multilingual benchmark languages (Dutch, French, German, Spanish).

Mirrors the structure of the English plot_asr.py but covers four languages
and three metrics.  Produces five output figures:

  wer_by_language.png      — 2×2 WER box plots, one subplot per language
  cer_by_language.png      — 2×2 CER box plots, one subplot per language
  rtf_by_language.png      — 2×2 RTF box plots, one subplot per language
  wer_cross_language.png   — median WER line plot, all languages overlaid
  rtf_cross_language.png   — median RTF line plot, all languages overlaid

Usage:
    pip install matplotlib pandas
    python plot_asr_multilingual.py

    # Save plots to a specific directory
    python plot_asr_multilingual.py --out_dir ./plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import pystow

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = pystow.module("coda", "benchmarks", "kaggle_ambient_scribe").join("results")

# Full model tag as it appears in the CSV filename (after the language code)
MODEL_SIZES = [
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large-v3-turbo",
    "whisper-large-v3",
]

# Short labels for plot axes
MODEL_LABELS = {
    "whisper-tiny": "tiny",
    "whisper-base": "base",
    "whisper-small": "small",
    "whisper-medium": "medium",
    "whisper-large-v3-turbo": "large-v3-turbo",
    "whisper-large-v3": "large-v3",
}

LANGUAGES = ["en", "nl", "fr", "de", "es"]
LANGUAGE_NAMES = {
    "en": "English",
    "nl": "Dutch",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}

# One colour per language for cross-language plots
LANGUAGE_COLORS = {
    "en": "#8172B2",
    "nl": "#4C72B0",
    "fr": "#DD8452",
    "de": "#55A868",
    "es": "#C44E52",
}

# Box fill colours for per-language plots (matching English script palette)
WER_COLOR = "#7BAFD4"
CER_COLOR = "#C9A0DC"
RTF_COLOR = "#A8D5A2"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(results_dir: Path) -> pd.DataFrame:
    """
    Load all per-encounter rows (excludes __AGGREGATE__) from every
    asr_benchmark_{lang}.{model}.csv file in results_dir.
    """
    frames = []
    for lang in LANGUAGES:
        for model in MODEL_SIZES:
            path = results_dir / f"asr_benchmark_{lang}.{model}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            # Drop the aggregate summary row
            df = df[df["audio_file"] != "__AGGREGATE__"].copy()
            df["model"] = model
            df["language"] = lang
            df["language_name"] = LANGUAGE_NAMES[lang]
            frames.append(df[["model", "language", "language_name", "WER", "CER", "rtf"]])

    if not frames:
        raise FileNotFoundError(
            f"No result CSVs found in {results_dir}.\n"
            f"Expected files like: asr_benchmark_nl.whisper-tiny.csv"
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _models_present(df: pd.DataFrame) -> list:
    """Return MODEL_SIZES filtered to those actually in the dataframe."""
    return [m for m in MODEL_SIZES if m in df["model"].unique()]


def _boxplot(ax, data_list, labels, color, metric):
    """Draw a clean box plot on ax."""
    bp = ax.boxplot(
        data_list,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xlabel("Whisper model", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)


def _print_stats(df: pd.DataFrame, metric: str, models: list) -> None:
    print(f"\n=== {metric} ===")
    for lang in LANGUAGES:
        subset = df[df["language"] == lang]
        if subset.empty:
            continue
        print(f"\n  {LANGUAGE_NAMES[lang]}")
        stats = (
            subset.groupby("model")[metric]
            .describe()
            .reindex([m for m in models if m in subset["model"].unique()])
        )
        print(stats[["count", "mean", "std", "min", "50%", "max"]].to_string())


def save_stats(df: pd.DataFrame, metrics: list, models: list, out_path: Path) -> None:
    """Save descriptive stats for all metrics and languages to a single CSV."""
    rows = []
    for metric in metrics:
        for lang in LANGUAGES:
            subset = df[df["language"] == lang]
            if subset.empty:
                continue
            for model in models:
                model_subset = subset[subset["model"] == model][metric].dropna()
                if model_subset.empty:
                    continue
                rows.append({
                    "metric": metric,
                    "language": lang,
                    "language_name": LANGUAGE_NAMES[lang],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "count": model_subset.count(),
                    "mean": round(model_subset.mean(), 4),
                    "std": round(model_subset.std(), 4),
                    "min": round(model_subset.min(), 4),
                    "median": round(model_subset.median(), 4),
                    "max": round(model_subset.max(), 4),
                })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 1 & 2 & 3: per-language 2×2 box plots (WER / CER / RTF)
# ---------------------------------------------------------------------------


def _plot_metric_by_language(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    color: str,
    output_path: Path,
    ylim=None,
) -> None:
    models = _models_present(df)
    short_labels = [MODEL_LABELS[m] for m in models]
    langs_present = [l for l in LANGUAGES if l in df["language"].unique()]

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 9), sharey=False)
    axes_flat = axes.flatten()

    for idx, lang in enumerate(langs_present):
        ax = axes_flat[idx]
        lang_df = df[df["language"] == lang]
        data_list = [
            lang_df[lang_df["model"] == m][metric].dropna().values
            for m in models
        ]
        _boxplot(ax, data_list, short_labels, color, metric)
        ax.set_title(LANGUAGE_NAMES[lang], fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)

    # Hide unused subplots if fewer than 4 languages
    for idx in range(len(langs_present), nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 4 & 5: cross-language line plots (median WER / RTF vs model size)
# ---------------------------------------------------------------------------


def _plot_metric_cross_language(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    models = _models_present(df)
    short_labels = [MODEL_LABELS[m] for m in models]
    x = range(len(models))

    fig, ax = plt.subplots(figsize=(10, 5))

    for lang in LANGUAGES:
        lang_df = df[df["language"] == lang]
        if lang_df.empty:
            continue
        medians = [
            lang_df[lang_df["model"] == m][metric].median()
            for m in models
        ]
        ax.plot(
            x, medians,
            marker="o", linewidth=2, markersize=6,
            label=LANGUAGE_NAMES[lang],
            color=LANGUAGE_COLORS[lang],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Whisper model", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(title="Language", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Plot WER, CER, and RTF results for the multilingual ASR benchmark."
    )
    ap.add_argument(
        "--out_dir", default=None,
        help="Directory to save plots.  Default: same folder as this script.",
    )
    ap.add_argument(
        "--no_show", action="store_true",
        help="Save plots without displaying them (useful for headless runs).",
    )
    args = ap.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {RESULTS_DIR}")
    df = load_data(RESULTS_DIR)
    models = _models_present(df)
    print(f"Models found   : {[MODEL_LABELS[m] for m in models]}")
    print(f"Languages found: {[LANGUAGE_NAMES[l] for l in LANGUAGES if l in df['language'].unique()]}")
    print(f"Total encounter rows: {len(df)}")

    # Print descriptive stats to terminal and save to CSV
    _print_stats(df, "WER", models)
    _print_stats(df, "CER", models)
    _print_stats(df, "rtf", models)
    save_stats(df, metrics=["WER", "CER", "rtf"], models=models,
               out_path=out_dir / "stats_summary.csv")

    # ── WER by language ───────────────────────────────────────────────────
    _plot_metric_by_language(
        df, metric="WER",
        ylabel="Word error rate (WER)",
        title="Word Error Rate by Whisper model size",
        color=WER_COLOR,
        output_path=out_dir / "wer_by_language.png",
    )

    # ── CER by language ───────────────────────────────────────────────────
    _plot_metric_by_language(
        df, metric="CER",
        ylabel="Character error rate (CER)",
        title="Character Error Rate by Whisper model size",
        color=CER_COLOR,
        output_path=out_dir / "cer_by_language.png",
    )

    # ── RTF by language ───────────────────────────────────────────────────
    _plot_metric_by_language(
        df, metric="rtf",
        ylabel="Real-time factor (latency / audio duration)",
        title="Transcription Speed (RTF) by Whisper model size",
        color=RTF_COLOR,
        output_path=out_dir / "rtf_by_language.png",
    )

    # ── WER cross-language ────────────────────────────────────────────────
    _plot_metric_cross_language(
        df, metric="WER",
        ylabel="Median WER",
        title="Median WER across languages — model size comparison",
        output_path=out_dir / "wer_cross_language.png",
    )

    # ── RTF cross-language ────────────────────────────────────────────────
    _plot_metric_cross_language(
        df, metric="rtf",
        ylabel="Median RTF",
        title="Median RTF across languages — model size comparison",
        output_path=out_dir / "rtf_cross_language.png",
    )

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()