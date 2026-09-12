"""Shared statistics and plotting for language-support benchmarks."""
import json
from pathlib import Path


def load_results(results_dir):
    results = {}
    for path in sorted(Path(results_dir).glob("transcripts_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        clips = [clip for clip in data.get("clips", []) if "wer" in clip]
        if not clips:
            continue
        engine = data.get("engine", path.stem.removeprefix("transcripts_"))
        if engine in results:
            continue
        rtfs = [clip["rtf"] for clip in clips if clip.get("rtf") is not None]
        cers = [clip["cer"] for clip in clips if clip.get("cer") is not None]
        results[engine] = {
            "mean_wer": sum(clip["wer"] for clip in clips) / len(clips),
            "mean_cer": sum(cers) / len(cers) if cers else None,
            "mean_rtf": sum(rtfs) / len(rtfs) if rtfs else None,
            "n": len(clips),
        }
    return results


def print_summary(results):
    print(f"\n{'engine':30s} {'mean WER':>10s} {'mean CER':>10s} {'mean RTF':>10s} {'n':>4s}")
    for engine in sorted(results, key=lambda name: results[name]["mean_wer"]):
        row = results[engine]
        cer = f"{row['mean_cer']:.3f}" if row["mean_cer"] is not None else "n/a"
        rtf = f"{row['mean_rtf']:.2f}" if row["mean_rtf"] is not None else "n/a"
        print(f"{engine:30s} {row['mean_wer']:10.3f} {cer:>10s} {rtf:>10s} {row['n']:4d}")


def plot_comparison(results, out_path, title, *, show=False):
    import matplotlib.pyplot as plt

    engines = sorted(results, key=lambda name: results[name]["mean_wer"])
    values = ([results[name]["mean_wer"] for name in engines],
              [results[name]["mean_cer"] or 0 for name in engines],
              [results[name]["mean_rtf"] or 0 for name in engines])
    labels = ("Mean WER", "Mean CER", "Mean RTF (latency / audio duration)")
    titles = ("Mean WER by engine (lower is better)",
              "Mean CER by engine (lower is better)",
              "Speed by engine (lower is faster)")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for index, axis in enumerate(axes):
        axis.barh(engines, values[index], color="#55A868" if index == 0 else "#4C72B0")
        axis.set_xlabel(labels[index])
        axis.set_title(titles[index])
        axis.grid(axis="x", alpha=0.3, linestyle="--")
        if index in (0, 2):
            axis.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        for row, value in enumerate(values[index]):
            axis.text(value + 0.02, row, f"{value:.3f}" if index < 2 else f"{value:.2f}", va="center")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def report(results_dir, out_dir, title, *, show=False):
    results = load_results(results_dir)
    if not results:
        raise FileNotFoundError(f"No transcripts_*.json found in {results_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print_summary(results)
    plot_comparison(results, Path(out_dir) / "engine_comparison.png", title, show=show)
    return results
