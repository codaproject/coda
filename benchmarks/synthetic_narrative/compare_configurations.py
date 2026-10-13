"""The configurations the ticket asks to compare, and how to run them.

A configuration is a name plus the dialogue.whisper_livekit overrides it
changes from the shipped settings.yaml. A row is one configuration on one
variant and one language, i.e. seven recordings run one after another in one
process (WhisperLiveKit's engine is a process-wide singleton, so a process
holds one configuration). Rows are numbered so a cluster job array can pick
one by index.

    python compare_configurations.py --list                      show every row with its index
    python compare_configurations.py --row 12                    run one row here
    python compare_configurations.py --config-index 3            run every row of one configuration
    python compare_configurations.py --languages en --configs baseline vac_off   run a subset here
    python compare_configurations.py --table                     summarize every scored result

Results land in results/<config>/<variant>/<language>/<case>.json (run.py)
with <case>.score.json beside them (score.py).
"""
import argparse
import json
from pathlib import Path

from common import HERE
import run as runner
import score as scorer

BENCHMARK_DIR = HERE / "data" / "benchmark"
VARIANTS = ("controlled", "ambient")
# Tsonga is not one of Whisper's languages: the whisper-livekit path refuses
# to start with it ('tso' is not a valid language code), so its recordings
# exist in data/benchmark but cannot be run through this path.
LANGUAGES = ("en", "pt_br", "bn")

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
        if out.exists() and json.loads(out.read_text()).get("complete"):
            result = json.loads(out.read_text())
        else:
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
                 f"{mean([s['keep_up'] for s in ss]):.2f}",
                 str(sum(not s["complete"] for s in ss))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the rows")
    parser.add_argument("--row", type=int, default=None, help="run one row by index")
    parser.add_argument("--config-index", type=int, default=None,
                        help="run every row of the Nth configuration (one "
                             "cluster job per configuration)")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=None)
    parser.add_argument("--model", default="small")
    parser.add_argument("--realtime", action="store_true",
                        help="feed at real time (see run.py); default is lock-step")
    parser.add_argument("--table", action="store_true",
                        help="print the summary table of all scored results")
    args = parser.parse_args()

    if args.table:
        print(table())
        return
    all_rows = rows()
    if args.list:
        for i, r in enumerate(all_rows):
            print(f"{i:3d}  {r['config']:14s} {r['variant']:10s} {r['language']}")
        print(f"{len(all_rows)} rows")
        return
    if args.row is not None:
        selected = [all_rows[args.row]]
    elif args.config_index is not None:
        selected = rows([list(CONFIGS)[args.config_index]])
    else:
        selected = rows(args.configs, args.variants, args.languages)
    for r in selected:
        print(f"== {r['config']} / {r['variant']} / {r['language']} ==", flush=True)
        run_row(r, args.model, args.realtime)


if __name__ == "__main__":
    main()
