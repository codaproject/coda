"""Splicer: interleave non-speech clips into each speech-bank reading.

Writes data/benchmark/<variant>/<language>/<case_id>.wav and .json for two
variants of every case:
  controlled  speech and gaps only; between noises the room is digitally silent.
  ambient     the same composite with a continuous living-room recording under
              everything at BED_DB below the voice: a real pause is the room
              carrying on without the speaker.
Every choice is drawn from a seeded generator, so the same seed and banks give
the same bytes, and every choice is recorded in the JSON region map.

For one reading:
  1. Find the pauses (runs of at least half a second well below the reading's
     speech level, which is sentence ends and the longer mid-sentence pauses)
     and use them, plus the start and the end, as slots for gaps.
  2. Shuffle all non-speech clips and spread them over the slots, so every
     clip appears once per reading; a short reading may hold two in a slot.
  3. Give each gap a length from GAP_SECONDS, balanced so each length is used
     equally, and take it from a random offset within the one-minute clip.
  4. Scale each gap to LEVEL_DB below the reading's speech level, fade every
     edge over CROSSFADE_MS so no junction clicks, and concatenate.
  5. Write that as the controlled variant, then add the bed and write the
     ambient variant.

    python benchmarks/hallucination/splice.py --languages en bn --seed 0
"""
import argparse
import json

import numpy as np

from common import HERE, SAMPLE_RATE
import audio

SPEECH_DIR = HERE / "data" / "transcript_to_speech"
NON_SPEECH_DIR = HERE / "data" / "non_speech"
OUT_DIR = HERE / "data" / "benchmark"
LANGUAGES = ("en", "pt_br", "bn", "ts")

VARIANTS = ("controlled", "ambient")
GAP_SECONDS = (3.0, 7.0, 10.0)   # short, medium and long interruptions
LEVEL_DB = -20.0                 # gap level relative to the speech level
BED = "living_room"              # ambient variant: this clip under everything
BED_DB = -28.0                   # bed level relative to the speech level
CROSSFADE_MS = 30.0
# Pause detection: a run of frames at least this far below the reading's own
# speech level, for at least this long, is a pause a gap can go into. Half a
# second keeps to sentence ends and the longer mid-sentence pauses.
FRAME_MS = 20.0
PAUSE_BELOW_SPEECH_DB = 25.0
MIN_PAUSE_MS = 500.0
SPEECH_DBFS = -35.0              # frames above this count as speech for level


def _frame_dbfs(pcm):
    frame = int(FRAME_MS / 1000 * SAMPLE_RATE)
    x = np.asarray(pcm, dtype=np.float64)[: len(pcm) // frame * frame]
    rms = np.sqrt(np.mean(x.reshape(-1, frame) ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-9) / 32767)


def sentence_cuts(pcm, level):
    """Sample indices at the middle of each pause (see PAUSE_BELOW_SPEECH_DB)."""
    db = _frame_dbfs(pcm)
    frame = int(FRAME_MS / 1000 * SAMPLE_RATE)
    min_frames = int(MIN_PAUSE_MS / FRAME_MS)
    quiet = db < level - PAUSE_BELOW_SPEECH_DB
    cuts, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if i - start >= min_frames and start > 0 and i < len(db):
                cuts.append((start + i) // 2 * frame)
            start = None
    return cuts


def speech_level(pcm):
    db = _frame_dbfs(pcm)
    voiced = db[db > SPEECH_DBFS]
    return float(np.mean(voiced)) if voiced.size else float(np.mean(db))


def fade_edges(pcm):
    n = min(int(CROSSFADE_MS / 1000 * SAMPLE_RATE), len(pcm) // 2)
    x = np.asarray(pcm, dtype=np.float64).copy()
    ramp = np.linspace(0, 1, n)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def scale_to(pcm, target_dbfs):
    x = np.asarray(pcm, dtype=np.float64)
    rms = np.sqrt(np.mean(x * x))
    if rms == 0:
        return x
    return x * (10 ** (target_dbfs / 20) * 32767 / rms)


def plan_gaps(rng, clips, n_slots):
    """Deal every clip onto the slots with a balanced set of gap lengths."""
    names = list(clips)
    rng.shuffle(names)
    lengths = [GAP_SECONDS[i % len(GAP_SECONDS)] for i in range(len(names))]
    rng.shuffle(lengths)
    # Spread the clips over the reading: a random subset of slots when there
    # are more slots than clips, round-robin (two per slot) when fewer.
    if n_slots >= len(names):
        slots = sorted(rng.choice(n_slots, size=len(names), replace=False))
    else:
        slots = [i % n_slots for i in range(len(names))]
    gaps = [[] for _ in range(n_slots)]
    for slot, name, seconds in zip(slots, names, lengths):
        clip = clips[name]
        max_offset = max(len(clip) - int(seconds * SAMPLE_RATE), 0)
        offset = int(rng.integers(0, max_offset + 1))
        gaps[slot].append({"name": name, "seconds": seconds,
                           "offset": offset / SAMPLE_RATE})
    return gaps


def build_case(language, case_id, text, clips, seed):
    """Write both variants of one case; returns {variant: region map}."""
    rng = np.random.default_rng([seed, LANGUAGES.index(language),
                                 sorted_case_index(language, case_id)])
    speech = audio.read_wav(SPEECH_DIR / language / f"{case_id}.wav")
    level = speech_level(speech)
    cuts = sentence_cuts(speech, level)
    bounds = [0, *cuts, len(speech)]
    parts = [speech[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    gaps = plan_gaps(rng, clips, n_slots=len(parts) + 1)  # before, between, after

    pieces, regions, t = [], [], 0.0

    def add(piece, kind, name, **extra):
        nonlocal t
        pieces.append(fade_edges(piece))
        regions.append({"kind": kind, "name": name, "start": round(t, 3),
                        "end": round(t + len(piece) / SAMPLE_RATE, 3), **extra})
        t += len(piece) / SAMPLE_RATE

    def add_gaps(slot):
        for g in gaps[slot]:
            start = int(g["offset"] * SAMPLE_RATE)
            raw = clips[g["name"]][start:start + int(g["seconds"] * SAMPLE_RATE)]
            add(scale_to(raw, level + LEVEL_DB), "non_speech", g["name"],
                clip_offset=g["offset"])

    add_gaps(0)
    for i, part in enumerate(parts):
        add(part, "speech", "speech", part=i)
        add_gaps(i + 1)

    composite = np.concatenate(pieces)
    bed_clip = clips[BED].astype(np.float64)
    bed_offset = int(rng.integers(0, len(bed_clip)))
    looped = np.tile(np.roll(bed_clip, -bed_offset),
                     int(np.ceil(len(composite) / len(bed_clip))))[:len(composite)]
    variants = {
        "controlled": (composite, None),
        "ambient": (composite + scale_to(looped, level + BED_DB),
                    {"name": BED, "level_db": BED_DB,
                     "clip_offset": round(bed_offset / SAMPLE_RATE, 3)}),
    }

    non_speech = {}
    for r in regions:
        if r["kind"] == "non_speech":
            non_speech[r["name"]] = round(
                non_speech.get(r["name"], 0.0) + r["end"] - r["start"], 3)
    infos = {}
    for variant, (signal, bed) in variants.items():
        out_dir = OUT_DIR / variant / language
        audio.write_wav(out_dir / f"{case_id}.wav",
                        np.clip(signal, -32768, 32767).astype(np.int16))
        info = {
            "case_id": case_id, "language": language, "variant": variant,
            "seed": seed, "reference": text,
            "speech_file": str((SPEECH_DIR / language / f"{case_id}.wav")
                               .relative_to(HERE)),
            "speech_level_dbfs": round(level, 1), "gap_level_db": LEVEL_DB,
            "gap_seconds": GAP_SECONDS, "crossfade_ms": CROSSFADE_MS,
            "bed": bed,
            "total_seconds": round(t, 3),
            "speech_seconds": round(len(speech) / SAMPLE_RATE, 3),
            "non_speech_seconds": non_speech,
            "sentence_cuts": len(cuts),
            "regions": regions,
        }
        (out_dir / f"{case_id}.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n")
        infos[variant] = info
    return infos


_case_order = {}


def sorted_case_index(language, case_id):
    if language not in _case_order:
        manifest = json.loads((SPEECH_DIR / language / "manifest.json").read_text())
        _case_order[language] = sorted(manifest["cases"])
    return _case_order[language].index(case_id)


def load_clips():
    manifest = json.loads((NON_SPEECH_DIR / "manifest.json").read_text())
    return {name: audio.read_wav(NON_SPEECH_DIR / info["file"])
            for name, info in manifest["clips"].items()}


def build(language, seed=0):
    clips = load_clips()
    manifest = json.loads((SPEECH_DIR / language / "manifest.json").read_text())
    cases = {}
    for case_id in sorted(manifest["cases"]):
        infos = build_case(language, case_id, manifest["cases"][case_id]["text"],
                           clips, seed)
        info = infos["controlled"]
        cases[case_id] = {"total_seconds": info["total_seconds"],
                          "speech_seconds": info["speech_seconds"],
                          "sentence_cuts": info["sentence_cuts"]}
        ns = sum(info["non_speech_seconds"].values())
        print(f"  {language}/{case_id:10s} {info['total_seconds']:6.1f}s total  "
              f"{info['speech_seconds']:5.1f}s speech in {info['sentence_cuts'] + 1} "
              f"parts  {ns:5.1f}s non-speech in {len(clips)} gaps  "
              f"x{len(VARIANTS)} variants", flush=True)
    for variant in VARIANTS:
        (OUT_DIR / variant / language / "manifest.json").write_text(json.dumps(
            {"language": language, "variant": variant, "seed": seed,
             "gap_seconds": GAP_SECONDS, "gap_level_db": LEVEL_DB,
             "bed": {"name": BED, "level_db": BED_DB} if variant == "ambient"
             else None, "cases": cases}, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES,
                        default=list(LANGUAGES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for language in args.languages:
        build(language, args.seed)


if __name__ == "__main__":
    main()
