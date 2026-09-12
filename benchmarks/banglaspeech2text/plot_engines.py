"""
Quick comparison plot across all engines tested by bangla_asr_bench.py.

Reads results/transcripts_{engine}.json (the script's own output format)
and produces a three-panel bar chart: mean WER, mean CER, and mean RTF per engine,
sorted by WER (best first).

Usage:
    python3 plot_engines.py
    python3 plot_engines.py --results_dir results --out_dir plots
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(results_dir: Path):
    """Load every transcripts_*.json in results_dir into {engine: data}."""
    results = {}
    for path in sorted(results_dir.glob("transcripts_*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        engine = data.get("engine", path.stem.replace("transcripts_", ""))
        clips = [c for c in data.get("clips", []) if "wer" in c]
        if not clips:
            print(f"[skip] {path.name}: no clips with WER found")
            continue
        if engine in results:
            print(f"[warn] {path.name}: duplicate engine {engine!r} "
                  f"(already loaded from another file) -- keeping the first one")
            continue
        mean_wer = sum(c["wer"] for c in clips) / len(clips)
        rtfs = [c["rtf"] for c in clips if c.get("rtf") is not None]
        mean_rtf = sum(rtfs) / len(rtfs) if rtfs else None
        cers = [c["cer"] for c in clips if c.get("cer") is not None]
        mean_cer = sum(cers) / len(cers) if cers else None
        results[engine] = {"mean_wer": mean_wer, "mean_cer": mean_cer, "mean_rtf": mean_rtf, "n": len(clips)}
    return results


def plot_comparison(results: dict, out_path: Path, *, show: bool = True):
    engines = sorted(results, key=lambda e: results[e]["mean_wer"])
    wers = [results[e]["mean_wer"] for e in engines]
    cers = [results[e]["mean_cer"] if results[e]["mean_cer"] is not None else 0 for e in engines]
    has_cer = [results[e]["mean_cer"] is not None for e in engines]
    rtfs = [results[e]["mean_rtf"] if results[e]["mean_rtf"] is not None else 0 for e in engines]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    colors = ["#55A868" if w < 0.15 else "#DD8452" if w < 0.6 else "#C44E52" for w in wers]
    ax.barh(engines, wers, color=colors)
    ax.set_xlabel("Mean WER")
    ax.set_title("Mean WER by engine (lower is better)")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    for i, w in enumerate(wers):
        ax.text(w + 0.02, i, f"{w:.3f}", va="center", fontsize=9)

    ax = axes[1]
    cer_colors = ["#4C72B0" if has_cer[i] else "#CCCCCC" for i in range(len(engines))]
    ax.barh(engines, cers, color=cer_colors)
    ax.set_xlabel("Mean CER")
    ax.set_title("Mean CER by engine (lower is better)")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    for i, (c, present) in enumerate(zip(cers, has_cer)):
        label = f"{c:.3f}" if present else "n/a"
        ax.text((c if present else 0) + 0.01, i, label, va="center", fontsize=9)

    ax = axes[2]
    ax.barh(engines, rtfs, color="#4C72B0")
    ax.set_xlabel("Mean RTF (latency / audio duration)")
    ax.set_title("Speed by engine (lower is faster)")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    for i, r in enumerate(rtfs):
        ax.text(r + 0.02, i, f"{r:.2f}", va="center", fontsize=9)

    fig.suptitle("Bengali ASR engine comparison (bangla_asr_bench.py)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    if not results:
        raise FileNotFoundError(f"No transcripts_*.json found in {results_dir}")

    print(f"\n{'engine':30s} {'mean WER':>10s} {'mean CER':>10s} {'mean RTF':>10s} {'n':>4s}")
    for engine in sorted(results, key=lambda e: results[e]["mean_wer"]):
        r = results[engine]
        cer_str = f"{r['mean_cer']:.3f}" if r["mean_cer"] is not None else "n/a"
        rtf_str = f"{r['mean_rtf']:.2f}" if r["mean_rtf"] is not None else "n/a"
        print(f"{engine:30s} {r['mean_wer']:10.3f} {cer_str:>10s} {rtf_str:>10s} {r['n']:4d}")

    plot_comparison(results, out_dir / "engine_comparison.png", show=not args.no_show)


if __name__ == "__main__":
    main()
