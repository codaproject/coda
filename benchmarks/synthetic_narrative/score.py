"""Scorer: count one run's words against its recording's region map.

For each result JSON written by run.py, produce the two ticket metrics and
write them next to it as <case>.score.json:

  non_speech  words heard during noise gaps, per noise type and in total, as
              words per minute of that noise. A word counts only if it lies at
              least MARGIN_S away from every speech region, because Whisper
              stamps the first word after a pause a little early; words the
              margin moves to speech are reported as `edge_words`.
  speech      word error rate of the words heard during speech regions, in
              time order, against the reference text, using the language's
              lang_support normalizer.

A "word" is one timed piece from the transcriber (faster-whisper's word
timestamps), which is also what CODA emits. slowdown is clock time divided by
audio time: above 1.0 the transcriber took longer than the recording.

    python benchmarks/synthetic_narrative/score.py results/lockstep/baseline
    python benchmarks/synthetic_narrative/score.py results/lockstep/baseline/controlled/en/lri_1.json
"""
import argparse
import importlib
import json
import re
import unicodedata
from pathlib import Path

from common import HERE
from metrics import wer_details

MARGIN_S = 0.5


def normalizer(language):
    if language != "en":
        adapter = importlib.import_module(f"languages.{language}")
        return lambda text: adapter.normalize(text, False)

    def normalize_en(text):
        text = "".join(" " if unicodedata.category(c).startswith("P") else c
                       for c in text.lower())
        return re.sub(r"\s+", " ", text).strip()
    return normalize_en


def attribute(words, regions, margin=MARGIN_S):
    """Label each word 'speech', 'gap' or 'edge' (a gap word within margin of speech)."""
    speech = [(r["start"], r["end"]) for r in regions if r["kind"] == "speech"]
    end = regions[-1]["end"] if regions else 0.0
    out = []
    for w in words:
        mid = min(max((w["start"] + w["end"]) / 2, 0.0), end - 1e-6)
        region = next((r for r in regions if r["start"] <= mid < r["end"]), None)
        if region is None or region["kind"] == "speech":
            label = "speech"
        elif any(s - margin < w["end"] and w["start"] < e + margin
                 for s, e in speech):
            label = "edge"
        else:
            label = "gap"
        out.append((label, region["name"] if region else "speech", w))
    return out


def score(result, splice_map):
    regions = splice_map["regions"]
    labelled = attribute(result["words"], regions)
    per_noise = {name: {"seconds": secs, "words": 0, "text": []}
                 for name, secs in splice_map["non_speech_seconds"].items()}
    speech_words, edge_words = [], []
    for label, name, w in labelled:
        if label == "gap":
            per_noise[name]["words"] += 1
            per_noise[name]["text"].append(w["text"])
        elif label == "edge":
            edge_words.append(w["text"].strip())
            speech_words.append(w["text"])
        else:
            speech_words.append(w["text"])

    total_s = sum(v["seconds"] for v in per_noise.values())
    for v in per_noise.values():
        v["text"] = "".join(v["text"]).strip()
        v["words_per_min"] = round(60 * v["words"] / v["seconds"], 2) if v["seconds"] else None
    total_words = sum(v["words"] for v in per_noise.values())

    hyp = "".join(speech_words).strip()
    normalize = normalizer(result["language"])
    wer, s, d, i, n, _ = wer_details(splice_map["reference"], hyp, normalize)
    return {
        "non_speech": {**per_noise,
                       "total": {"seconds": round(total_s, 3), "words": total_words,
                                 "words_per_min": round(60 * total_words / total_s, 2)
                                 if total_s else None}},
        "edge_words": len(edge_words), "edge_text": edge_words,
        "speech": {"seconds": splice_map["speech_seconds"], "wer": round(wer, 4),
                   "S": s, "D": d, "I": i, "N": n, "hyp": hyp},
        "slowdown": round(result["wall_seconds"] / result["audio_seconds"], 3),
        "complete": result.get("complete", True),
    }


def score_file(path):
    result = json.loads(Path(path).read_text())
    splice_map = json.loads((HERE / result["map"]).read_text())
    scored = score(result, splice_map)
    out = Path(path).with_suffix(".score.json")
    out.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n")
    return result, scored


def summary_line(result, scored):
    ns = scored["non_speech"]
    return (f"{result['variant']}/{result['language']}/{Path(result['composite']).stem:10s}"
            f"  gap words {ns['total']['words']:3d} ({ns['total']['words_per_min']:5.1f}/min)"
            f"  edge {scored['edge_words']:2d}  WER {scored['speech']['wer']:.3f}"
            f"  slowdown {scored['slowdown']:.2f}"
            + ("" if scored["complete"] else "  INCOMPLETE"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+",
                        help="result JSON files or directories to scan")
    args = parser.parse_args()
    files = []
    for p in args.paths:
        p = Path(p)
        files += sorted(f for f in p.rglob("*.json")
                        if not f.name.endswith(".score.json")) if p.is_dir() else [p]
    for f in files:
        result, scored = score_file(f)
        print(summary_line(result, scored))


if __name__ == "__main__":
    main()
