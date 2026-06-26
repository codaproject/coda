"""Plot and summarize multilingual ASR benchmark results.

Reads the per-language result CSVs written by benchmark_asr.py
(asr_benchmark_results.{lang}.{model}.csv) from the pystow results directory
and produces:
  - A printed summary and a combined summary.csv (from the aggregate rows)
  - Per-language box plots of WER, CER, and RTF across Whisper model sizes
  - Cross-language line plots of median WER and RTF across model sizes
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import pystow

RESULTS_DIR = pystow.module("coda", "benchmarks", "kaggle_ambient_scribe").join("results")

# Model tags as they appear in the CSV filename (after the language code)
MODEL_SIZES = [
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large",
    "whisper-large-v2",
]

MODEL_LABELS = {m: m.replace("whisper-", "") for m in MODEL_SIZES}

LANGUAGES = ["en", "nl", "fr", "de", "es"]
LANGUAGE_NAMES = {"en": "English", "nl": "Dutch", "fr": "French", "de": "German", "es": "Spanish"}

# One colour per language for cross-language line plots
LANGUAGE_COLORS = {
    "en": "#8172B2",
    "nl": "#4C72B0",
    "fr": "#DD8452",
    "de": "#55A868",
    "es": "#C44E52",
}

# Box fill colours per metric
WER_COLOR = "#7BAFD4"
CER_COLOR = "#C9A0DC"
RTF_COLOR = "#A8D5A2"


def load_results(results_dir: Path):
    """Load per-encounter rows and aggregate rows from all result CSVs."""
    enc_frames = []
    agg_records = []
    for lang in LANGUAGES:
        for model in MODEL_SIZES:
            path = results_dir / f"asr_benchmark_results.{lang}.{model}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            agg = df[df["audio_file"] == "__AGGREGATE__"]
            enc = df[df["audio_file"] != "__AGGREGATE__"].copy()
            enc["model"] = model
            enc["language"] = lang
            enc_frames.append(enc[["model", "language", "WER", "CER", "rtf"]])
            if not agg.empty:
                a = agg.iloc[0]
                agg_records.append({
                    "model": model,
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
            f"Expected files like: asr_benchmark_results.nl.whisper-tiny.csv"
        )

    return pd.concat(enc_frames, ignore_index=True), pd.DataFrame(agg_records)


def _models_present(df: pd.DataFrame) -> list:
    """Return MODEL_SIZES filtered to those actually in the dataframe."""
    return [m for m in MODEL_SIZES if m in df["model"].unique()]


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

def print_summary(agg_df: pd.DataFrame, models: list) -> None:
    """Print a per-model-per-language summary and a WER comparison matrix."""
    if agg_df.empty:
        return

    disp = agg_df.copy()
    disp["audio"] = disp["duration_s"].map(seconds_to_hms)
    disp["runtime"] = disp["latency_s"].map(seconds_to_hms)
    disp["model"] = pd.Categorical(disp["model"], categories=models, ordered=True)
    disp["language"] = pd.Categorical(disp["language"], categories=LANGUAGES, ordered=True)
    disp = disp.sort_values(["model", "language"])

    print("\n=== Results by model and language ===")
    cols = ["model", "language_name", "encounters", "audio", "runtime", "rtf", "WER", "CER"]
    print(disp[cols].to_string(index=False))

    print("\n=== WER by model and language ===")
    matrix = agg_df.pivot_table(index="model", columns="language", values="WER")
    matrix = matrix.reindex(
        index=[m for m in models if m in matrix.index],
        columns=[l for l in LANGUAGES if l in matrix.columns],
    )
    matrix["avg"] = matrix.mean(axis=1)
    print(matrix.round(4).to_string())


def save_summary_csv(agg_df: pd.DataFrame, out_path: Path) -> None:
    """Save the aggregate summary to a single CSV."""
    cols = ["model", "language", "language_name", "encounters",
            "duration_s", "latency_s", "rtf", "WER", "CER"]
    agg_df[cols].to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")


# -------------------------------
# Plots
# -------------------------------

def _boxplot(ax, data_list, labels, color):
    """Draw a box plot on the given axes."""
    bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
    ax.set_xlabel("Whisper model")
    ax.grid(axis="y", alpha=0.3)


def plot_metric_by_language(df, metric, ylabel, title, color, output_path, ylim=None):
    """Per-language box plots of a metric across model sizes (one subplot per language)."""
    models = _models_present(df)
    labels = [MODEL_LABELS[m] for m in models]
    langs = [l for l in LANGUAGES if l in df["language"].unique()]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes_flat = axes.flatten()

    for idx, lang in enumerate(langs):
        ax = axes_flat[idx]
        lang_df = df[df["language"] == lang]
        data_list = [lang_df[lang_df["model"] == m][metric].dropna().values for m in models]
        _boxplot(ax, data_list, labels, color)
        ax.set_title(LANGUAGE_NAMES[lang])
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)

    for idx in range(len(langs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.show()


def plot_metric_cross_language(df, metric, ylabel, title, output_path):
    """Line plot of median metric per language across model sizes."""
    models = _models_present(df)
    labels = [MODEL_LABELS[m] for m in models]
    x = range(len(models))

    fig, ax = plt.subplots(figsize=(10, 5))
    for lang in LANGUAGES:
        lang_df = df[df["language"] == lang]
        if lang_df.empty:
            continue
        medians = [lang_df[lang_df["model"] == m][metric].median() for m in models]
        ax.plot(x, medians, marker="o", linewidth=2, label=LANGUAGE_NAMES[lang],
                color=LANGUAGE_COLORS[lang])

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Whisper model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Language")
    ax.grid(alpha=0.3)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Plot and summarize multilingual ASR benchmark results.")
    ap.add_argument("--out_dir", default=None, help="Directory for plots and summary CSV (default: results dir).")
    ap.add_argument("--no_show", action="store_true", help="Save figures without displaying them.")
    args = ap.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_df, agg_df = load_results(RESULTS_DIR)
    models = _models_present(enc_df)

    print_summary(agg_df, models)
    save_summary_csv(agg_df, out_dir / "summary.csv")

    plot_metric_by_language(enc_df, "WER", "Word error rate (WER)",
                            "Word error rate (WER) by Whisper model size", WER_COLOR,
                            out_dir / "wer_by_language.png", ylim=(0, 0.4))
    plot_metric_by_language(enc_df, "CER", "Character error rate (CER)",
                            "Character error rate (CER) by Whisper model size", CER_COLOR,
                            out_dir / "cer_by_language.png")
    plot_metric_by_language(enc_df, "rtf", "Real-time factor (processing time / audio duration)",
                            "Transcription speed by Whisper model size", RTF_COLOR,
                            out_dir / "rtf_by_language.png")
    plot_metric_cross_language(enc_df, "WER", "Median WER",
                               "Median WER across languages by model size",
                               out_dir / "wer_cross_language.png")
    plot_metric_cross_language(enc_df, "rtf", "Median RTF",
                               "Median RTF across languages by model size",
                               out_dir / "rtf_cross_language.png")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
