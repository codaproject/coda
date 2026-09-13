"""Shared statistics and plotting for language-support benchmarks."""
import json
import subprocess
from pathlib import Path


def hardware():
    """Describe the host so recorded RTF values can be compared across machines."""
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True,
                                  text=True).stdout.strip()
        except Exception:
            return ""
    memory = sysctl("hw.memsize")
    return {"chip": sysctl("machdep.cpu.brand_string"),
            "ram_gb": round(int(memory) / 1024 ** 3) if memory.isdigit() else None}


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


def wer_colors(values):
    """Color WER bars by quality tier: usable, poor, unusable."""
    return ["#55A868" if v < 0.15 else "#DD8452" if v < 0.6 else "#C44E52"
            for v in values]


def plot_comparison(results, out_path, title, *, show=False):
    import matplotlib.pyplot as plt

    engines = sorted(results, key=lambda name: results[name]["mean_wer"])
    wers = [results[name]["mean_wer"] for name in engines]
    cers = [results[name]["mean_cer"] or 0 for name in engines]
    rtfs = [results[name]["mean_rtf"] or 0 for name in engines]
    # A missing CER is drawn gray so it reads as absent rather than as zero
    cer_colors = ["#4C72B0" if results[name]["mean_cer"] is not None else "#CCCCCC"
                  for name in engines]
    panels = (
        (wers, "Mean WER", "Mean WER by engine (lower is better)",
         wer_colors(wers), "{:.3f}"),
        (cers, "Mean CER", "Mean CER by engine (lower is better)",
         cer_colors, "{:.3f}"),
        (rtfs, "Mean RTF (latency / audio duration)",
         "Speed by engine (lower is faster)", "#4C72B0", "{:.2f}"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for axis, (values, xlabel, subtitle, color, fmt) in zip(axes, panels):
        axis.barh(engines, values, color=color)
        axis.set_xlabel(xlabel)
        axis.set_title(subtitle)
        axis.grid(axis="x", alpha=0.3, linestyle="--")
        # Only mark the break-even line when the data reaches it, otherwise it
        # stretches the axis and squeezes every bar into a corner
        if max(values) > 0.75:
            axis.axvline(1.0, color="black", linestyle="--", linewidth=1,
                         alpha=0.5)
        axis.set_xlim(0, max(max(values), 1.0 if max(values) > 0.75 else 0) * 1.15)
        # Offset labels by a fraction of the axis, not a fixed amount, so they
        # stay beside their bars whatever the scale
        offset = axis.get_xlim()[1] * 0.01
        for row, value in enumerate(values):
            label = fmt.format(value) if color is not cer_colors \
                or results[engines[row]]["mean_cer"] is not None else "n/a"
            axis.text(value + offset, row, label, va="center", fontsize=9)
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
