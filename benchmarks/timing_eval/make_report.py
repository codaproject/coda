"""Build the timing-eval report into real_cases_report/:

  accuracy_over_time.png  aggregate top-1/top-3 accuracy vs spoken time (all models
                          found under real_cases_results/)
  final_predictions.csv   whole-file top-1/top-3 vs reference causes, plus timing
  index.html              the plot + per-case reference causes, final predictions,
                          reasoning, and follow-up questions (VA vs VA+clinical)
"""
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).parent
RESULTS_ROOT = HERE / "real_cases_results"
REPORT = HERE / "real_cases_report"
TRUTH = json.loads((HERE / "real_cases" / "true_labels_group.json").read_text())
CASES = sorted(TRUTH, key=int)

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
COLORS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#56b4e9", "#e69f00", "#000000"]
PHASES = [("phase1_va", "VA narrative only"),
          ("phase2_va_clinical", "VA + clinical")]


def discover_models(root):
    """Every model under the results root as [(name, dir, color)], sorted by name.
    A model is any subfolder that holds case* directories."""
    if not root.exists():
        return []
    dirs = sorted((d for d in root.iterdir() if d.is_dir() and any(d.glob("case*"))),
                  key=lambda d: d.name)
    return [(d.name, d, COLORS[i % len(COLORS)]) for i, d in enumerate(dirs)]


MODELS = discover_models(RESULTS_ROOT)


def ref_dir():
    """A model dir to read model-independent fields from (audio length,
    transcription time). These match across models because the transcript cache
    is shared, so the first model is as good as any."""
    return MODELS[0][1] if MODELS else None


def ref_cause(k):
    return TRUTH[k].get("Underlying Cause", ["?"])[0]


def ref_types(k):
    """[(type, [causes])] in COD order for case k."""
    order = ["Underlying Cause", "Immediate Cause of Death", "Morbid Conditions"]
    return [(name, TRUTH[k][name]) for name in order if name in TRUTH[k]]


# Accuracy vs spoken time

def chunk_points(root, k, phase):
    """[(elapsed_s, top1_correct, top3_correct)] over a case's chunks."""
    ref = ref_cause(k)
    pts = []
    for line in (root / f"case{k}" / phase / "chunked" / "chunks.jsonl").open():
        r = json.loads(line)
        ranked = [v["name"] for v in sorted(r.get("causes", {}).values(),
                                            key=lambda x: -x["score"])]
        pts.append((r["audio_elapsed_s"], ranked[:1] == [ref], ref in ranked[:3]))
    return pts


def va_duration(k):
    p = ref_dir() / f"case{k}" / "phase1_va" / "whole" / "inference.json"
    return json.loads(p.read_text()).get("audio_duration_s")


def step_onto(grid, pts, idx):
    """Forward-fill correctness (component idx of each point) onto grid; 0 before
    the first chunk, frozen at the last value after the final chunk."""
    out = np.zeros(len(grid))
    ci, cur = 0, 0
    for i, t in enumerate(grid):
        while ci < len(pts) and pts[ci][0] <= t:
            cur = int(pts[ci][idx])
            ci += 1
        out[i] = cur
    return out


def phase_curves(root, phase, grid):
    per_case = {k: chunk_points(root, k, phase) for k in CASES}
    top1 = np.mean([step_onto(grid, per_case[k], 1) for k in CASES], axis=0)
    top3 = np.mean([step_onto(grid, per_case[k], 2) for k in CASES], axis=0)
    return top1, top3


def phase_grid(phase):
    tmax = max(chunk_points(ref_dir(), k, phase)[-1][0] for k in CASES)
    return np.arange(0, tmax + 1, 1.0)


def write_accuracy_png():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (phase, label) in zip(axes, PHASES):
        ax.set_facecolor(SURFACE)
        grid = phase_grid(phase)
        if phase == "phase2_va_clinical":
            vas = [va_duration(k) for k in CASES]
            ax.axvspan(min(vas), max(vas), color=MUTED, alpha=0.08, zorder=0)
            ax.text(np.median(vas), 1.02, "clinical onset (range across cases)",
                    color=MUTED, fontsize=8, ha="center")
        for name, root, color in MODELS:
            if not root.exists():
                continue
            top1, top3 = phase_curves(root, phase, grid)
            ax.plot(grid, top1, color=color, lw=2.2, ls="-", label=f"{name} top-1")
            ax.plot(grid, top3, color=color, lw=2.2, ls=":", label=f"{name} top-3")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, grid[-1])
        ax.set_title(label, fontsize=12, color=INK)
        ax.set_xlabel("spoken audio time (s)", fontsize=10, color=MUTED)
        ax.tick_params(labelsize=9, colors=MUTED, labelleft=True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(True, color=GRID, lw=0.6)
    axes[0].set_ylabel("accuracy (fraction of 20 cases)", fontsize=10, color=MUTED)
    handles = [Line2D([0], [0], color=c, lw=2.2) for _, _, c in MODELS]
    labels = [n for n, _, _ in MODELS]
    handles += [Line2D([0], [0], color=INK, lw=2.2, ls="-"),
                Line2D([0], [0], color=INK, lw=2.2, ls=":")]
    labels += ["top-1", "top-3"]
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 6),
               fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    out = REPORT / "accuracy_over_time.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# Per-case table + HTML

def whole_top(path, n=3):
    if not path.exists():
        return []
    c = json.loads(path.read_text()).get("causes", {})
    return [v["name"] for v in sorted(c.values(), key=lambda x: -x["score"])][:n]


def whole_timing(path):
    """(audio_duration_s, transcription_s, total_processing_s) from a whole-file run."""
    if not path.exists():
        return None, None, None
    t = json.loads(path.read_text())
    tm = t.get("timing", {})
    return t.get("audio_duration_s"), tm.get("transcription_s"), tm.get("total_s")


def chunk_final(chunked_dir):
    """Final prediction plus timing: top-3, reasoning, questions from the last
    chunk with causes; audio length and the final chunk's inference time (that
    last inference runs over the whole accumulated transcript)."""
    empty = {"top3": [], "reasoning": "", "questions": [], "audio_s": None, "infer_s": None}
    p = chunked_dir / "chunks.jsonl"
    if not p.exists():
        return empty
    rows = [json.loads(l) for l in p.open()]
    with_causes = next((r for r in reversed(rows) if r.get("causes")), None)
    if not rows or with_causes is None:
        return empty
    top3 = [(v["name"], v["score"]) for v in
            sorted(with_causes["causes"].values(), key=lambda x: -x["score"])[:3]]
    return {"top3": top3, "reasoning": with_causes.get("reasoning") or "",
            "questions": with_causes.get("questions") or [],
            "audio_s": rows[-1].get("audio_elapsed_s"),
            "infer_s": rows[-1].get("timing", {}).get("inference_s")}


def write_table():
    out = REPORT / "final_predictions.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "case", "underlying", "immediate", "morbid_conditions",
                    "va_top1", "va_top3", "combined_top1", "combined_top3",
                    "va_audio_s", "va_processing_s",
                    "combined_audio_s", "combined_processing_s"])
        for name, mdir, _ in MODELS:
            for k in CASES:
                t = TRUTH[k]
                va_p = mdir / f"case{k}" / "phase1_va" / "whole" / "inference.json"
                cm_p = mdir / f"case{k}" / "phase2_va_clinical" / "whole" / "inference.json"
                va, cm = whole_top(va_p), whole_top(cm_p)
                va_audio, _, va_proc = whole_timing(va_p)
                cm_audio, _, cm_proc = whole_timing(cm_p)
                w.writerow([name, k,
                            "; ".join(t.get("Underlying Cause", [])),
                            "; ".join(t.get("Immediate Cause of Death", [])),
                            "; ".join(t.get("Morbid Conditions", [])),
                            va[0] if va else "-", " | ".join(va),
                            cm[0] if cm else "-", " | ".join(cm),
                            va_audio, va_proc, cm_audio, cm_proc])
    print(f"wrote {out}")


def model_detail(name, mdir, k, phase):
    d = chunk_final(mdir / f"case{k}" / phase / "chunked")
    top3 = ", ".join(f"{html.escape(n)} ({s:.2f})" for n, s in d["top3"]) or "-"
    qs = "".join(f"<li>{html.escape(q)}</li>" for q in d["questions"])
    infer = f"{d['infer_s']:.1f}s" if d["infer_s"] is not None else "-"
    return (f"<div class='model'><p class='mname'>{html.escape(name)} "
            f"<span class='timing'>({infer} inference)</span></p>"
            f"<p class='top'><b>Final top-3:</b> {top3}</p>"
            f"<p class='reason'>{html.escape(d['reasoning'])}</p>"
            f"<b style='font-size:13px'>Follow-up questions</b><ul>{qs}</ul></div>")


def phase_block(title, k, phase):
    audio_s, trans_s, _ = whole_timing(
        ref_dir() / f"case{k}" / phase / "whole" / "inference.json")
    audio = f"{audio_s:.0f}s" if audio_s is not None else "-"
    trans = f"{trans_s:.0f}s" if trans_s is not None else "-"
    details = "".join(model_detail(name, mdir, k, phase) for name, mdir, _ in MODELS)
    return (f"<div class='phase'><h3>{title}</h3>"
            f"<p class='timing'>Audio duration: {audio}, transcription: {trans}</p>"
            f"{details}</div>")


def write_index():
    model_names = ", ".join(name for name, _, _ in MODELS)
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>CODA timing evaluation on CHAMPS cases</title>",
             "<style>",
             "body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;",
             "margin:0;background:#fcfcfb;color:#0b0b0b}",
             "header{position:sticky;top:0;background:#fcfcfb;border-bottom:1px solid #e6e5e2;",
             "padding:12px 24px}",
             "nav a{margin-right:10px;color:#2a78d6;text-decoration:none;font-size:13px}",
             "main{padding:0 24px 60px}",
             ".case{border-bottom:1px solid #e6e5e2;padding:22px 0}",
             ".case h2{margin:0 0 4px}",
             ".refs{color:#52514e;font-size:14px;margin:0 0 4px}",
             ".refs b{color:#0b0b0b}",
             ".detail{display:flex;gap:24px;margin-top:12px}",
             ".phase{flex:1;background:#f7f6f3;border:1px solid #e6e5e2;",
             "border-radius:6px;padding:12px 16px}",
             ".phase h3{margin:0 0 6px;font-size:14px}",
             ".phase .timing{font-size:12px;color:#52514e;margin:0 0 8px}",
             ".phase .top{font-size:13px;color:#0b0b0b;margin:0 0 8px}",
             ".phase .reason{font-size:13px;color:#52514e;margin:0 0 8px}",
             ".phase ul{margin:0;padding-left:18px;font-size:13px;color:#52514e}",
             ".phase .model{border-top:1px solid #e6e5e2;margin-top:8px;padding-top:8px}",
             ".phase .mname{font-size:13px;font-weight:600;margin:0 0 4px}",
             ".overview{padding:20px 0;border-bottom:1px solid #e6e5e2}",
             ".overview img{max-width:100%;height:auto}",
             ".overview p{max-width:1100px;color:#52514e;font-size:14px;margin:0 0 10px}",
             "</style></head><body>",
             "<header><b>CODA timing evaluation on CHAMPS cases</b>, reference causes vs final "
             f"prediction ({model_names} inference, faster-whisper medium transcription, gilda).",
             "<nav>" + " ".join(f"<a href='#c{k}'>#{k}</a>" for k in CASES) + "</nav></header>",
             "<main>",
             "<div class='overview'><h2>Accuracy vs spoken time (n=20)</h2>"
             "<p>Plots showing whether the CHAMPS reference group cause matches the top CODA-inferred "
             "cause (blue) or is found in the top 3 CODA-inferred causes (orange) based on "
             "voice input containing only a VA narrative (left) or the same VA narrative "
             "followed by a clinical narrative (right).</p>"
             "<p>Since each case's voice recording length differs, the range of time at which "
             "there is a VA narrative to clinical narrative switch over is shown as a gray "
             "shaded area. Once a case's recording ends, the final inferred causes are used "
             "when calculating the average accuracy across cases.</p>"
             "<img src='accuracy_over_time.png' alt='accuracy over time'></div>"]
    for k in CASES:
        refs = "  ".join(f"<b>{name}:</b> {', '.join(c)}" for name, c in ref_types(k))
        parts.append(f"<div class='case' id='c{k}'><h2>Case {k}</h2>"
                     f"<p class='refs'>{refs}</p>"
                     f"<div class='detail'>{phase_block('VA only', k, 'phase1_va')}"
                     f"{phase_block('VA + clinical', k, 'phase2_va_clinical')}</div></div>")
    parts.append("</main></body></html>")
    out = REPORT / "index.html"
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


def main():
    if not MODELS:
        raise SystemExit(f"No model results under {RESULTS_ROOT}. Run run_eval.py first.")
    REPORT.mkdir(exist_ok=True)
    write_accuracy_png()
    write_table()
    write_index()


if __name__ == "__main__":
    main()
