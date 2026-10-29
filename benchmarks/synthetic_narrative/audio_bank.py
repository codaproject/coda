"""Speech bank: a synthetic reading of every lang_support narrative.

Writes data/transcript_to_speech/<language>/<case_id>.wav (16 kHz mono int16)
and a manifest.json per language recording the model and the exact text read,
so WER against it is exact. All four languages cover the same seven cases: en
is the `narrative` field of the same reference files.

One engine for every language, so differences between languages are not
differences between voices: Meta's MMS-TTS (facebook/mms-tts-<iso3>) through
transformers, offline after a ~150 MB download per language, CPU is enough.

    python benchmarks/synthetic_narrative/audio_bank.py --languages en pt_br bn ts
"""
import argparse
import json
import re

import numpy as np

from common import HERE, LANG_SUPPORT, SAMPLE_RATE
import audio

OUT_DIR = HERE / "data" / "transcript_to_speech"
REFERENCES_DIR = LANG_SUPPORT / "data"

# language -> (reference file, candidate text fields), as the lang_support
# adapters read them.
REFERENCES = {
    "en": ("bn/references/cases_bn_filtered.json", ("narrative",)),
    "bn": ("bn/references/cases_bn_filtered.json", ("bn_narrative",)),
    "pt_br": ("pt_br/references/cases_ptbr_filtered.json",
              ("ptbr_narrative", "bn_narrative")),
    "ts": ("ts/references/cases_ts.json", ("ts_narrative",)),
}

MODELS = {
    "en": "facebook/mms-tts-eng",
    "pt_br": "facebook/mms-tts-por",
    "bn": "facebook/mms-tts-ben",
    "ts": "facebook/mms-tts-tso",
}

PAUSE_S = 0.35  # silence between sentences


def references(language):
    """case_id -> reference text, read the way lang_support reads it."""
    from dataset_io import read_references
    path, keys = REFERENCES[language]
    return read_references(REFERENCES_DIR / path, keys)


_mms = {}


def _mms_model(model_id):
    if model_id not in _mms:
        import torch
        from transformers import AutoTokenizer, VitsModel
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id).eval()
        torch.manual_seed(0)  # VITS samples noise; keep renders reproducible
        _mms[model_id] = (tokenizer, model)
    return _mms[model_id]


def _sentences(text):
    # Split on sentence enders incl. the Bengali danda; keep the ender.
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p for p in parts if p]


def render(text, model_id):
    """Synthesize sentence by sentence and join with short pauses."""
    import torch
    tokenizer, model = _mms_model(model_id)
    rate = model.config.sampling_rate
    pause = np.zeros(int(PAUSE_S * rate), dtype=np.float32)
    pieces = []
    for sentence in _sentences(text):
        inputs = tokenizer(sentence, return_tensors="pt")
        if inputs["input_ids"].shape[-1] == 0:
            continue
        with torch.no_grad():
            wave = model(**inputs).waveform[0].numpy()
        pieces += [wave, pause]
    wave = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        wave = resample_poly(wave, SAMPLE_RATE, rate)
    peak = float(np.max(np.abs(wave))) or 1.0
    return (wave / peak * 0.8 * 32767).astype(np.int16)


def build(language, force=False):
    """Render every case of one language; returns the manifest."""
    out_dir = OUT_DIR / language
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = MODELS[language]
    manifest = {"language": language, "engine": "mms-tts", "model": model_id,
                "cases": {}}
    for case_id, text in references(language).items():
        path = out_dir / f"{case_id}.wav"
        if force or not path.exists():
            audio.write_wav(path, render(text, model_id))
        pcm = audio.read_wav(path)
        manifest["cases"][case_id] = {
            "file": path.name, "seconds": round(audio.seconds(pcm), 2),
            "words": len(text.split()), "text": text}
        print(f"  {language}/{case_id:10s} {audio.seconds(pcm):6.1f}s  "
              f"{len(text.split()):4d} words  [{model_id}]", flush=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--languages", nargs="+", choices=list(MODELS),
                        default=list(MODELS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for language in args.languages:
        build(language, force=args.force)


if __name__ == "__main__":
    main()
