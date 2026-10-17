"""The configurations the ticket asks to compare, and how to run them.

A configuration is a name plus the dialogue.whisper_livekit overrides it
changes from the shipped settings.yaml. Each configuration is run on every
variant and language selected, seven recordings each, one after another, and
the summary table is printed at the end.

    python compare_configurations.py --languages en
    python compare_configurations.py --configs baseline vac_off --variants controlled
    python compare_configurations.py --table      reprint the summary without running

Results land in results/<feed>/<config>/<variant>/<language>/<case>.json
(run.py) with <case>.score.json beside them (score.py).
"""
import argparse
import json
from pathlib import Path

from common import HERE
import run as runner
import score as scorer

BENCHMARK_DIR = HERE / "data" / "benchmark"
VARIANTS = ("controlled", "ambient")
LANGUAGES = ("en", "pt_br", "bn", "ts")  # ts runs on its fine-tune, see run.py

# The ticket's comparison rows. baseline is the shipped settings.yaml:
# WhisperLiveKit's voice gate on, faster-whisper's own thresholds. Repeats of
# the baseline measure run-to-run variation so differences smaller than it
# are not read as findings.
CONFIGS = {
    "baseline": {},
    "baseline_rep2": {},
    "baseline_rep3": {},
    "vac_off": {"VAC": "false"},
    "ns_0.3": {"DECODE_OPTIONS__NO_SPEECH_THRESHOLD": "0.3"},
    "ns_0.45": {"DECODE_OPTIONS__NO_SPEECH_THRESHOLD": "0.45"},
    "ns_0.75": {"DECODE_OPTIONS__NO_SPEECH_THRESHOLD": "0.75"},
    "ns_0.9": {"DECODE_OPTIONS__NO_SPEECH_THRESHOLD": "0.9"},
    "cr_1.8": {"DECODE_OPTIONS__COMPRESSION_RATIO_THRESHOLD": "1.8"},
    "cr_off": {"DECODE_OPTIONS__COMPRESSION_RATIO_THRESHOLD": "@none"},
}


def rows(configs=None, variants=None, languages=None):
    """Every (config, variant, language) row, in a fixed order."""
    out = []
    for config in configs or CONFIGS:
        for variant in variants or VARIANTS:
            for language in languages or LANGUAGES:
                out.append({"config": config, "variant": variant,
                            "language": language})
    return out


def composites(variant, language):
    return sorted((BENCHMARK_DIR / variant / language).glob("*.wav"))


def run_row(row, model="small", realtime=False):
    """Run and score the seven recordings of one row; returns their scores."""
    overrides = CONFIGS[row["config"]]
    scored = []
    for composite in composites(row["variant"], row["language"]):
        out = (runner.RESULTS_DIR / ("realtime" if realtime else "lockstep")
               / row["config"] / row["variant"] / row["language"]
               / composite.with_suffix(".json").name)
        result = json.loads(out.read_text()) if out.exists() else None
        if not (result and result.get("complete")):
            result = runner.run(composite, model, overrides, realtime, out)
        _, s = scorer.score_file(out)
        print(scorer.summary_line(result, s), flush=True)
        scored.append(s)
    return scored


def table():
    """Mean of each metric per (feed, config, variant, language) over its cases."""
    groups = {}
    for path in sorted(runner.RESULTS_DIR.rglob("*.score.json")):
        parts = path.relative_to(runner.RESULTS_DIR).parts
        if len(parts) < 5:
            continue  # not in the results/<feed>/<config>/<variant>/<lang>/ layout
        feed, config, variant, language = parts[:4]
        groups.setdefault((feed, config, variant, language), []).append(
            json.loads(path.read_text()))
    noises = sorted({n for ss in groups.values() for s in ss
                     for n in s["non_speech"] if n != "total"})
    header = ["feed", "config", "variant", "lang", "n", "gap w/min",
              *[f"{n} w/min" for n in noises], "WER", "slowdown", "incomplete"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else float("nan")

    for (feed, config, variant, language), ss in sorted(groups.items()):
        cells = [feed, config, variant, language, str(len(ss)),
                 f"{mean([s['non_speech']['total']['words_per_min'] for s in ss]):.1f}",
                 *[f"{mean([s['non_speech'].get(n, {}).get('words_per_min') for s in ss]):.1f}"
                   for n in noises],
                 f"{mean([s['speech']['wer'] for s in ss]):.3f}",
                 f"{mean([s['slowdown'] for s in ss]):.2f}",
                 str(sum(not s["complete"] for s in ss))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=None)
    parser.add_argument("--model", default="small")
    parser.add_argument("--realtime", action="store_true",
                        help="feed at real time (see run.py); default is lock-step")
    parser.add_argument("--table", action="store_true",
                        help="reprint the summary table without running anything")
    args = parser.parse_args()

    if not args.table:
        for r in rows(args.configs, args.variants, args.languages):
            print(f"== {r['config']} / {r['variant']} / {r['language']} ==", flush=True)
            run_row(r, args.model, args.realtime)
    print(table())


if __name__ == "__main__":
    main()
