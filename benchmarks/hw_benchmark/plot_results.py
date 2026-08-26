"""Plot every results/*.json so configs compare without conflating model size.

One subplot per model family (small multiples), each plotting mean inference
latency vs memory bandwidth across the machines that ran it, so hardware is the
only axis that varies within a panel. Points are colored by vendor (Apple vs AMD)
and labeled with the chip. A final panel shows speech-to-text real-time keep-up
per machine (<=1.0 means it transcribes in real time).

Qwen2.5-7B is the one model every machine runs, so its panel is the direct
apples-to-apples hardware comparison. Latency and keep-up are backend-agnostic, so
Ollama/MLX (Mac) and Lemonade (Ryzen) land on the same axes. Writes comparison.png
and summary.md next to this script.
"""
import glob
import json
from collections import defaultdict
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e7e6e2"
SURFACE = "#fcfcfb"

# Fixed family order (roughly ascending model size), for stable panel layout.
FAMILY_ORDER = ["Qwen2.5-7B", "Qwen3-8B", "Qwen3-14B", "gpt-oss-20B",
                "Gemma-4-12B", "Gemma-4-26B", "Qwen3-30B-A3B"]
# Colorblind-safe categorical slots (validated): identity here is vendor.
VENDOR_COLOR = {"Apple": "#2a78d6", "AMD": "#eb6834"}
SHARED_FAMILY = "Qwen2.5-7B"


def family_of(name, model_id):
    s = f"{name} {model_id or ''}".lower()
    rules = [
        ("Qwen2.5-7B", ("qwen2.5-7b", "qwen2.5:7b")),
        ("Qwen3-30B-A3B", ("qwen3-30b",)),
        ("Qwen3-14B", ("qwen3-14b",)),
        ("Qwen3-8B", ("qwen3-8b",)),
        ("gpt-oss-20B", ("gpt-oss-20b", "gpt-oss:20b")),
        ("Gemma-4-12B", ("gemma-4-12b", "gemma4-12b")),
        ("Gemma-4-26B", ("gemma-4-26b", "gemma4-26b", "gemma4:26b")),
    ]
    for fam, needles in rules:
        if any(n in s for n in needles):
            return fam
    return name


def short_chip(chip):
    c = (chip or "").replace("Apple ", "")
    if "RYZEN AI MAX" in c.upper():
        head = c.split(" w/")[0].split(" with")[0]
        return head.title().replace("Amd", "AMD").replace("Ryzen Ai", "Ryzen AI")
    return c


def vendor(chip):
    return "AMD" if "ryzen" in (chip or "").lower() else "Apple"


def load_machines():
    """One record per chip, keeping each (chip, family)'s best measurement."""
    stt = {}
    bw = {}
    vend = {}
    per_family = defaultdict(dict)
    for f in sorted(glob.glob(str(HERE / "results" / "*.json"))):
        d = json.load(open(f))
        hw = d.get("hardware", {})
        chip = short_chip(hw.get("chip"))
        if not chip:
            continue
        bw[chip] = hw.get("bandwidth_gbps")
        vend[chip] = vendor(hw.get("chip"))
        s = d.get("stt") or {}
        if isinstance(s, dict) and s.get("keep_up_mean") is not None:
            if chip not in stt or d.get("timestamp", "") > stt[chip][1]:
                stt[chip] = (s["keep_up_mean"], d.get("timestamp", ""))
        for m in d.get("models", []):
            if m.get("mean_latency_sec") is None:
                continue
            fam = family_of(m.get("name", ""), m.get("id"))
            row = {"mean": m["mean_latency_sec"], "sd": m.get("stdev_latency_sec") or 0,
                   "valid": m.get("valid_rate", 0), "ts": d.get("timestamp", "")}
            prev = per_family[chip].get(fam)
            if prev is None or (row["valid"], row["ts"]) > (prev["valid"], prev["ts"]):
                per_family[chip][fam] = row
    return {"bw": bw, "vendor": vend, "stt": {k: v[0] for k, v in stt.items()},
            "families": per_family}


def _panel_style(ax):
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _model_panel(ax, data, fam, xlim):
    ys = []
    for chip, fams in data["families"].items():
        row = fams.get(fam)
        x = data["bw"].get(chip)
        if not row or x is None:
            continue
        v = data["vendor"][chip]
        ax.errorbar(x, row["mean"], yerr=row["sd"], fmt="o", ms=8,
            color=VENDOR_COLOR[v], ecolor=VENDOR_COLOR[v], elinewidth=1,
            capsize=3, mec=SURFACE, mew=1.2, zorder=3)
        ax.annotate(chip, (x, row["mean"]), textcoords="offset points",
            xytext=(6, 3), fontsize=7, color=INK)
        ys.append(row["mean"] + (row["sd"] or 0))
    title = fam + ("  (shared)" if fam == SHARED_FAMILY else "")
    ax.set_title(title, fontsize=10, fontweight="bold", color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, (max(ys) * 1.3) if ys else 1)
    _panel_style(ax)


def _stt_panel(ax, data):
    chips = sorted(data["stt"], key=lambda c: data["bw"].get(c) or 0)
    vals = [data["stt"][c] for c in chips]
    ypos = list(range(len(chips)))
    colors = [VENDOR_COLOR[data["vendor"][c]] for c in chips]
    ax.barh(ypos, vals, color=colors, height=0.6, zorder=3)
    ax.axvline(1.0, color=MUTED, ls="--", lw=1, zorder=2)
    for y, v in zip(ypos, vals):
        ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{c} ({data['bw'].get(c)})" for c in chips], fontsize=7)
    ax.set_xlim(0, max(vals) * 1.3 if vals else 1.3)
    ax.set_title("STT keep-up  (<=1.0 = real time)", fontsize=10,
        fontweight="bold", color=INK)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot(data):
    plt.rcParams.update({"font.size": 10, "text.color": INK,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "svg.fonttype": "none"})

    present = [f for f in FAMILY_ORDER if any(f in v for v in data["families"].values())]
    bws = [b for b in data["bw"].values() if b]
    xlim = (min(bws) - 40, max(bws) + 70) if bws else (0, 700)

    n = len(present) + 1
    ncols = 4
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 3.3),
        squeeze=False)
    flat = axes.flatten()

    fig.suptitle("CODA hardware benchmark - inference latency per narrative "
        "(lower is faster), one panel per model", fontsize=13, fontweight="bold",
        color=INK, x=0.5, y=0.99)

    for idx, fam in enumerate(present):
        _model_panel(flat[idx], data, fam, xlim)
        if idx % ncols == 0:
            flat[idx].set_ylabel("mean latency (s)")
    _stt_panel(flat[len(present)], data)
    for j in range(n, len(flat)):
        flat[j].axis("off")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=VENDOR_COLOR[v],
        mec=SURFACE, mew=1, label=v) for v in ("Apple", "AMD")]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="lower center",
        fontsize=9, bbox_to_anchor=(0.5, 0.005))
    fig.supxlabel("Memory bandwidth (GB/s)  (STT panel: keep-up)", fontsize=10,
        color=INK, y=0.055)

    fig.tight_layout(rect=(0, 0.08, 1, 0.965))
    out = HERE / "comparison.png"
    fig.savefig(out, dpi=150)
    return out


def write_summary(data):
    lines = ["# CODA hardware benchmark summary", "",
        "Mean inference latency (s) per model family, best measurement per machine. "
        f"`{SHARED_FAMILY}` runs on every machine.", ""]
    fams = [f for f in FAMILY_ORDER if any(f in v for v in data["families"].values())]
    header = ["Machine", "BW GB/s", "STT keep-up"] + fams
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for chip in sorted(data["families"], key=lambda c: data["bw"].get(c) or 0):
        row = [chip, str(data["bw"].get(chip) or "?"),
               f"{data['stt'].get(chip, float('nan')):.2f}" if chip in data["stt"] else "-"]
        for f in fams:
            r = data["families"][chip].get(f)
            row.append(f"{r['mean']:.1f}" if r else "-")
        lines.append("| " + " | ".join(row) + " |")
    out = HERE / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main():
    data = load_machines()
    png = plot(data)
    md = write_summary(data)
    print(f"Wrote {png}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
