"""Benchmark Bangla ASR models on the COD audio clips (WER vs bn_narrative).

Each clip's reference is the matching case's bn_narrative in data/bn/references/cases_bn_filtered.json.
Whisper-based models run through the transformers ASR pipeline with long-form
chunking; indic-seamless and indic-conformer use their own code paths. Word-level
WER is computed in-process (no jiwer). Run with no args for all engines, or pass
engine names.
"""
import argparse
import os
import re
import subprocess
import unicodedata
from pathlib import Path

from dataset_io import load_samples
from languages.bn import normalize as normalize_bengali
from metrics import cer as metric_cer, wer_details as metric_wer_details

import numpy as np

from coda.config import settings

# Initialize shared configuration before model libraries read environment variables.
settings.validators.validate()

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def hardware():
    def sc(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True,
                text=True).stdout.strip()
        except Exception:
            return ""
    mem = sc("hw.memsize")
    return {"chip": sc("machdep.cpu.brand_string"),
            "ram_gb": round(int(mem) / 1024 ** 3) if mem.isdigit() else None}


def clip_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return None

BASE = Path(__file__).resolve().parent


BENGALI_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
# Ambiguous words such as নয় (nine / is not) are left unchanged.
BENGALI_CARDINAL_WORDS = {
    "শূন্য": "0", "এক": "1", "দুই": "2", "দু": "2", "দো": "2",
    "তিন": "3", "চার": "4", "চারি": "4", "পাঁচ": "5",
    "ছয়": "6", "সাত": "7", "আট": "8",
    "দশ": "10", "এগারো": "11", "বারো": "12", "তেরো": "13",
    "চৌদ্দ": "14", "চোদ্দ": "14", "পনেরো": "15", "পনর": "15",
    "ষোলো": "16", "সতেরো": "17", "সতর": "17", "আঠারো": "18", "আঠেরো": "18",
    "ঊনিশ": "19", "ঊন্নিশ": "19", "বিশ": "20", "কুড়ি": "20", "একুশ": "21",
    "ত্রিশ": "30", "তিরিশ": "30", "চল্লিশ": "40", "পঞ্চাশ": "50",
    "ষাট": "60", "ষাটি": "60", "ষাইট": "60", "সত্তর": "70", "আশি": "80",
    "নব্বই": "90", "নব্বুই": "90", "শত": "100", "একশ": "100",
}


def normalize_bengali_numerals(text):
    """Canonicalize Bengali digits and cardinal words to Western digit strings.

    Both "৫" and "পাঁচ" become "5", so numeral-form differences don't register
    as WER/CER errors.
    """
    text = text.translate(BENGALI_DIGIT_MAP)
    words = text.split()
    return " ".join(BENGALI_CARDINAL_WORDS.get(w, w) for w in words)


def _is_punctuation(ch):
    """True if ch is punctuation, meaning a Unicode category starting with P.

    A [^\\w\\s] regex is not equivalent, it also strips Bengali vowel signs and
    diacritics (categories Mc/Mn, e.g. া ি ু ে ঁ ্) because \\w does not match
    combining marks.
    """
    return unicodedata.category(ch).startswith("P")


def strip_punctuation(text):
    return "".join(" " if _is_punctuation(ch) else ch for ch in text)


def norm(t):
    t = strip_punctuation(t)
    t = re.sub(r"\s+", " ", t).strip()
    t = normalize_bengali_numerals(t)
    return t


def levenshtein_ops(a, b):
    """Return (S, D, I, N): substitutions, deletions, insertions, len(a).

    Generic over any equality-comparable sequence, used for both word-level
    and character-level edit distance.
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "E"
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                dele = dp[i - 1][j] + 1
                best = min(sub, ins, dele)
                dp[i][j] = best
                op[i][j] = "S" if best == sub else ("I" if best == ins else "D")
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        cur = op[i][j]
        if cur == "E":
            i -= 1
            j -= 1
        elif cur == "S":
            S += 1
            i -= 1
            j -= 1
        elif cur == "I":
            I += 1
            j -= 1
        elif cur == "D":
            D += 1
            i -= 1
        else:
            break
    return S, D, I, n


def wer_details(ref, hyp):
    """Full breakdown: (WER, S, D, I, N, Accuracy)."""
    r, h = norm(ref).split(), norm(hyp).split()
    S, D, I, N = levenshtein_ops(r, h)
    w = (S + D + I) / N if N else float("nan")
    acc = (N - S - D - I) / N if N else float("nan")
    return w, S, D, I, N, acc


def cer(ref, hyp):
    """Character-level error rate, edits divided by reference character count.

    Uses the same normalization as wer_details(), on characters instead of
    words, and reports a total edit rate with no S/D/I breakdown.
    """
    r, h = list(norm(ref).replace(" ", "")), list(norm(hyp).replace(" ", ""))
    S, D, I, N = levenshtein_ops(r, h)
    return (S + D + I) / N if N else float("nan")


# Shared scoring is the source of truth; the legacy definitions above remain
# temporarily for compatibility with notebooks that imported them directly.
def wer_details(ref, hyp):
    return metric_wer_details(ref, hyp, normalize_bengali)


def cer(ref, hyp):
    return metric_cer(ref, hyp, normalize_bengali)


def samples():
    return [(s.case_id, str(s.audio_path), s.reference, clip_duration(s.audio_path))
            for s in load_samples("bn")]


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_audio(path, sr=16000):
    """Decode any format to mono float32 at sr via ffmpeg (avoids soundfile/m4a
    and torchcodec issues in the transformers pipeline)."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(path), "-f", "f32le", "-ac", "1",
         "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def make_whisper(repo, base_config_repo=None):
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoConfig
    dev = _device()
    proc = AutoProcessor.from_pretrained(repo)
    if base_config_repo:
        config = AutoConfig.from_pretrained(base_config_repo)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            repo, config=config, dtype=torch.float32
        ).to(dev).eval()
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(repo, dtype=torch.float32).to(dev).eval()

    try:
        forced = proc.get_decoder_prompt_ids(language="bn", task="transcribe")
    except Exception:
        forced = None
    model.generation_config.forced_decoder_ids = None

    def run(path):
        audio = load_audio(path)
        win = 30 * 16000
        texts = []
        for i in range(0, len(audio), win):
            chunk = audio[i:i + win]
            if len(chunk) < 1600:
                continue
            feats = proc(chunk, sampling_rate=16000,
                         return_tensors="pt").input_features.to(dev).to(model.dtype)
            kw = {"forced_decoder_ids": forced} if forced else {}
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=440, **kw)
            texts.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
        return " ".join(texts)
    return run


def make_seamless(repo, tgt_lang="ben"):
    import os
    from transformers import AutoProcessor, SeamlessM4Tv2Model
    token = os.environ.get("HF_TOKEN")
    proc = AutoProcessor.from_pretrained(repo, token=token)
    model = SeamlessM4Tv2Model.from_pretrained(repo, token=token).to(_device())

    def run(path):
        inputs = proc(audio=load_audio(path), sampling_rate=16000,
                      return_tensors="pt").to(_device())
        out = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        return proc.decode(out[0].tolist()[0], skip_special_tokens=True)
    return run


def make_conformer(decoding="ctc"):
    """Build a run(path) -> str closure using the IndicConformerTranscriber
    class from src/coda/dialogue/indic_conformer.py
    """
    import asyncio
    from coda.dialogue.indic_conformer import IndicConformerTranscriber

    transcriber = IndicConformerTranscriber.create(model=decoding)

    def run(path):
        audio_float = load_audio(path)  # float32 mono @ 16kHz via ffmpeg

        # REQUIRED: transcribe_audio expects int16 PCM, not float32 [-1,1] casting directly truncates instead of scaling.
        pcm = (np.clip(audio_float, -1.0, 1.0) * 32767.0).astype(np.int16)

        return asyncio.run(
            transcriber.transcribe_audio(pcm, sample_rate=16000, language="bn", task="transcribe")
        )
    return run


def make_bst(size):
    from banglaspeech2text import Speech2Text
    stt = Speech2Text(size)
    return lambda path: stt.recognize(path)


def make_mlx(repo):
    import mlx_whisper
    return lambda path: mlx_whisper.transcribe(
        path, path_or_hf_repo=repo, language="bn")["text"]

def make_speechmatics(model="enhanced"):
    """Build a run(path) -> str closure using the SpeechmaticsTranscriber
    class from src/coda/dialogue/speechmatics.py.

    SpeechmaticsTranscriber.transcribe_audio() is async; this script's
    ENGINES closures are synchronous, so asyncio.run() bridges the two.
    """
    import asyncio
    from coda.dialogue.speechmatics import SpeechmaticsTranscriber

    transcriber = SpeechmaticsTranscriber.create(model=model)

    def run(path):
        audio_float = load_audio(path)  # float32 mono @ 16kHz, via ffmpeg (already resamples)

        # REQUIRED: transcribe_audio expects int16 PCM, not float32 [-1,1].
        # Casting float32 directly to int16 truncates instead of scaling
        # (e.g. 0.5 -> 0, not ~16383)
        pcm = (np.clip(audio_float, -1.0, 1.0) * 32767.0).astype(np.int16)

        return asyncio.run(
            transcriber.transcribe_audio(
                pcm, sample_rate=16000, language="bn", task="transcribe"
            )
        )
    return run


ENGINES = {
    "mlx-whisper-small": lambda: make_mlx("mlx-community/whisper-small-mlx"),
    "banglaspeech2text-base": lambda: make_bst("base"),
    "banglaspeech2text-large": lambda: make_bst("large"),
    "indic-whisper": lambda: make_whisper(
        "parthiv11/indic_whisper_nodcil", base_config_repo="openai/whisper-large-v2",), # This checkpoint's own config.json doesn't match its actual weights
    "tugstugi-regional-medium": lambda: make_whisper(
        "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"),
    "tugstugi-medium": lambda: make_whisper(
        "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"),
    "bangla-whisper-large-v3": lambda: make_whisper(
        "utshobs/bangla_whisper_large_v3_finetuned"),
    "indic-seamless": lambda: make_seamless("ai4bharat/indic-seamless"),
    "indic-conformer": lambda: make_conformer(decoding="ctc"),
    "speechmatics-enhanced": lambda: make_speechmatics("enhanced"),
    "speechmatics-standard": lambda: make_speechmatics("standard"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*", default=None,
                    help="engine names to run (default: all)")
    args = ap.parse_args()
    from run_benchmark import run
    return run("bn", args.engines or None)


if __name__ == "__main__":
    main()
