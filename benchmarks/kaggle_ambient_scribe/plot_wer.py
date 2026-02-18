"""Box plots of Word Error Rate and Real-Time Factor across Whisper model sizes."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent

MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2"]


def load_data() -> pd.DataFrame:
    """Load WER and RTF values from all model CSV files into a single DataFrame."""
    frames = []
    for model in MODEL_SIZES:
        path = RESULTS_DIR / f"asr_benchmark_results.{model}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["model"] = model
        frames.append(df[["model", "WER", "rtf"]])
    return pd.concat(frames, ignore_index=True)


def _boxplot(ax, df, column, models, color):
    """Draw a box plot on the given axes."""
    data = [df[df["model"] == m][column].values for m in models]
    bp = ax.boxplot(data, labels=models, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
    ax.set_xlabel("Whisper Model")
    ax.grid(axis="y", alpha=0.3)


def plot_wer_boxplot(df: pd.DataFrame, output_path: Path = None):
    """Create a box plot of WER by model size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models_present = [m for m in MODEL_SIZES if m in df["model"].values]

    _boxplot(ax, df, "WER", models_present, "#7BAFD4")
    ax.set_ylim(0, 0.4)
    ax.set_ylabel("Word error rate (WER)")
    ax.set_title("Word error rate (WER) by Whisper model size")

    fig.tight_layout()
    if output_path is None:
        output_path = RESULTS_DIR / "wer_boxplot.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")
    plt.show()


def plot_rtf_boxplot(df: pd.DataFrame, output_path: Path = None):
    """Create a box plot of Real-Time Factor by model size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models_present = [m for m in MODEL_SIZES if m in df["model"].values]

    _boxplot(ax, df, "rtf", models_present, "#A8D5A2")
    ax.set_ylabel("Real-time factor (processing time / audio duration)")
    ax.set_title("Transcription speed by Whisper model size")

    fig.tight_layout()
    if output_path is None:
        output_path = RESULTS_DIR / "rtf_boxplot.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    df = load_data()
    print("=== WER ===")
    print(df.groupby("model")["WER"].describe().loc[MODEL_SIZES])
    print("\n=== RTF ===")
    print(df.groupby("model")["rtf"].describe().loc[MODEL_SIZES])
    plot_wer_boxplot(df)
    plot_rtf_boxplot(df)
