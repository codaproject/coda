"""
Batch benchmark Whisper ASR against plain-text transcripts from the
Kaggle Multilingual Ambient Scribe dataset.

Extends the English-only benchmark to Dutch, French, German, and Spanish
(and keeps English available for a clean comparison run).

Key changes vs. the original benchmark_asr.py
──────────────────────────────────────────────
1.  LanguageConfig dataclass holds per-language folder names, filename
    templates, regex patterns, and the Whisper language name.
2.  Whisper decoding now passes forced_decoder_ids so the model is
    explicitly told which language to transcribe (rather than auto-detecting).
    Falls back to passing language= kwarg directly for newer transformers.
3.  `--language` accepts en | nl | fr | de | es | all.
    `all` processes nl, fr, de, es in one run loading the model once.
4.  Output CSV gains a `language` column; per-language files are saved
    separately so they can be compared side-by-side.
5.  Override flags (--lang_folder, --audio_pattern, --audio_template,
    --transcript_template) let you fix up folder/filename mismatches
    without editing the source.
6.  Apple Silicon MPS support: device order is cuda → mps → cpu.

    Run with --dry_run to check discovered paths without doing any ASR.

Setup
─────
    pip install kagglehub pystow soundfile transformers torch numpy

    # One-time Kaggle auth:
    # 1. kaggle.com → Account → API → Create New Token → downloads kaggle.json
    # 2. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
    # 3. chmod 600 ~/.kaggle/kaggle.json
    # 4. Accept dataset license at the Kaggle URL in your browser

Usage
─────
    # Dry run first – confirms folder/filename paths without doing any ASR
    python benchmark_asr_multilingual.py --language all --dry_run

    # Single language
    python benchmark_asr_multilingual.py --language fr --model_id openai/whisper-small

    # All non-English languages (model loaded once)
    python benchmark_asr_multilingual.py --language all --model_id openai/whisper-large-v3

    # Override folder name if it differs in the dataset
    python benchmark_asr_multilingual.py --language nl --lang_folder "Netherlands Dutch"

Recommended model progression
──────────────────────────────
    openai/whisper-tiny           # smoke test – verify pipeline end-to-end
    openai/whisper-base
    openai/whisper-small
    openai/whisper-medium
    openai/whisper-large-v3-turbo
    openai/whisper-large-v3       # best quality

    Avoid *.en variants (whisper-tiny.en etc.) – English-only, useless here.
"""

import csv
import math
import re
import time
import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kagglehub
import numpy as np
import pystow
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ---------------------------------------------------------------------------
# Dataset root (kagglehub caches on first run, then reuses the local copy)
# ---------------------------------------------------------------------------

DATASET_ROOT = (
    Path(kagglehub.dataset_download("imeritinc/multilingual-ambient-scribe-dataset"))
    / "iMerit_Multilingual_Ambient_Scribe_Dataset"
)

RESULTS_BASE = pystow.module("coda", "benchmarks", "kaggle_ambient_scribe")

# ---------------------------------------------------------------------------
# Per-language configuration
#
# folder         – sub-folder of DATASET_ROOT containing audio/ and transcripts/
# audio_re       – regex to extract the encounter ID from an audio filename
# audio_tpl      – filename template for audio files  (use {id})
# transcript_tpl – filename template for transcripts  (use {id})
# whisper_lang   – language name passed to WhisperProcessor / model.generate()
# ---------------------------------------------------------------------------


@dataclass
class LanguageConfig:
    code: str            # ISO 639-1 code used as dict key
    whisper_lang: str    # Whisper language name (e.g. "french")
    folder: str          # Sub-folder under DATASET_ROOT
    audio_re: re.Pattern
    audio_tpl: str
    transcript_tpl: str


LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {
    "en": LanguageConfig(
        code="en",
        whisper_lang="english",
        folder="UK English",
        audio_re=re.compile(r"recording_uk_encounter_(\d+)\.mp3"),
        audio_tpl="recording_uk_encounter_{id}.mp3",
        transcript_tpl="Encounter {id}_UK.txt",
    ),
    "nl": LanguageConfig(
        code="nl",
        whisper_lang="dutch",
        folder="Dutch",                            
        audio_re=re.compile(r"recording_dutch_encounter_(\d+)\.mp3"),
        audio_tpl="recording_dutch_encounter_{id}.mp3",
        transcript_tpl="Encounter {id}_Dutch.txt",
    ),
    "fr": LanguageConfig(
        code="fr",
        whisper_lang="french",
        folder="French",                           
        audio_re=re.compile(r"recording_french_encounter_(\d+)\.mp3"),
        audio_tpl="recording_french_encounter_{id}.mp3",
        transcript_tpl="Encounter {id}_French.txt",
    ),
    "de": LanguageConfig(
        code="de",
        whisper_lang="german",
        folder="German",                           
        audio_re=re.compile(r"recording_german_encounter_(\d+)\.mp3"),
        audio_tpl="recording_german_encounter_{id}.mp3",
        transcript_tpl="Encounter {id}_German.txt",
    ),
    "es": LanguageConfig(
        code="es",
        whisper_lang="spanish",
        folder="Spanish",                          
        audio_re=re.compile(r"recording_spanish_encounter_(\d+)\.mp3"),
        audio_tpl="recording_spanish_encounter_{id}.mp3",
        transcript_tpl="Encounter {id}_Spanish.txt",
    ),
}

# 'all' runs these four (English is already benchmarked)
ALL_LANGUAGES = ["nl", "fr", "de", "es"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 16_000  # Whisper expects 16 kHz mono

# ---------------------------------------------------------------------------
# Device selection  (cuda → mps → cpu)
# Covers: NVIDIA GPU, Apple Silicon (M1/M2/M3/M4), Intel/AMD CPU
# ---------------------------------------------------------------------------


def best_device(override: Optional[str] = None) -> str:
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Audio utilities  (unchanged from original)
# ---------------------------------------------------------------------------


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1).astype(np.float32, copy=False)


def resample_audio_linear(
    audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR
) -> np.ndarray:
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    duration = len(audio) / float(orig_sr)
    tgt_len = int(round(duration * target_sr))
    if tgt_len <= 1:
        return audio[:1].astype(np.float32, copy=False)
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, duration, num=tgt_len, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Text normalisation  (unchanged from original)
# ---------------------------------------------------------------------------

PUNCT_STRIP = re.compile(r"[^\w\s]", flags=re.UNICODE)


def remove_punctuation(text: str) -> str:
    return PUNCT_STRIP.sub(" ", text)


def normalize_for_scoring(
    text: str, lowercase: bool = True, strip_punct: bool = True
) -> str:
    t = text.lower() if lowercase else text
    if strip_punct:
        t = remove_punctuation(t)
    return re.sub(r"\s+", " ", t).strip()


def words(text: str) -> List[str]:
    return text.split()


# ---------------------------------------------------------------------------
# Reference transcript loading  (unchanged from original)
# ---------------------------------------------------------------------------


def load_reference_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    while lines and lines[0].strip() == "":
        lines.pop(0)
    if lines and lines[0].strip().lower() == "transcripted line":
        lines = lines[1:]
    return normalize_for_scoring("\n".join(lines), lowercase=True, strip_punct=True)


# ---------------------------------------------------------------------------
# Edit-distance metrics  (unchanged from original)
# ---------------------------------------------------------------------------


def levenshtein_ops(
    ref: List[str], hyp: List[str]
) -> Tuple[int, int, int, int]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op: List[List[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            hj = hyp[j - 1]
            if ri == hj:
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
            i -= 1; j -= 1
        elif cur == "S":
            S += 1; i -= 1; j -= 1
        elif cur == "I":
            I += 1; j -= 1
        elif cur == "D":
            D += 1; i -= 1
        else:
            break
    return S, D, I, n


def cer_ops(ref_text: str, hyp_text: str) -> Tuple[int, int]:
    r, h = list(ref_text), list(hyp_text)
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m], n


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_id: str, device: Optional[str] = None
) -> Tuple[WhisperProcessor, WhisperForConditionalGeneration, str]:
    device = best_device(device)
    print(f"  Device   : {device}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.to(device).eval()
    return processor, model, device


# ---------------------------------------------------------------------------
# Whisper decoding – language-aware
#
# Passes forced_decoder_ids so the model is told which language to transcribe
# rather than auto-detecting.  Falls back to language= kwarg for newer
# transformers versions that deprecated forced_decoder_ids (≥4.47).
# If you see deprecation warnings, they are harmless; the fallback fires
# automatically.
# ---------------------------------------------------------------------------


def _build_forced_decoder_ids(
    processor: WhisperProcessor,
    language: str,
    task: str,
) -> Optional[list]:
    try:
        return processor.get_decoder_prompt_ids(language=language, task=task)
    except Exception:
        return None


def _decode_whisper_once(
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
    audio_chunk: np.ndarray,
    sr: int,
    task: str = "transcribe",
    language: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt")
    gen_kwargs: Dict = {"temperature": temperature}

    if language:
        forced_ids = _build_forced_decoder_ids(processor, language, task)
        if forced_ids is not None:
            gen_kwargs["forced_decoder_ids"] = forced_ids
        else:
            # Newer transformers: pass directly
            gen_kwargs["language"] = language
            gen_kwargs["task"] = task
    else:
        gen_kwargs["task"] = task

    with torch.no_grad():
        pred_ids = model.generate(inputs.input_features.to(device), **gen_kwargs)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


def _seconds_to_samples(seconds: float, sr: int) -> int:
    return int(round(seconds * sr))


def _smart_join(prev: str, new: str, max_overlap_words: int = 8) -> str:
    if not prev:
        return new
    pw = prev.split()
    nw = new.split()
    max_k = min(max_overlap_words, len(pw), len(nw))
    for k in range(max_k, 0, -1):
        if pw[-k:] == nw[:k]:
            return prev + " " + " ".join(nw[k:])
    return prev + " " + new


def transcribe_chunked(
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
    audio_rs: np.ndarray,
    sr: int = TARGET_SR,
    task: str = "transcribe",
    language: Optional[str] = None,
    chunk_length_s: float = 29.5,
    overlap_s: float = 1.0,
    temperature: float = 0.0,
    temperature_fallbacks: Tuple[float, ...] = (0.2, 0.4),
) -> str:
    L = len(audio_rs)
    if L == 0:
        return ""
    hop = _seconds_to_samples(max(0.01, chunk_length_s - overlap_s), sr)
    win = _seconds_to_samples(chunk_length_s, sr)
    if hop <= 0:
        hop = win
    start = 0
    full_text = ""
    while start < L:
        end = min(start + win, L)
        chunk = audio_rs[start:end]
        txt = _decode_whisper_once(
            processor, model, device, chunk, sr,
            task=task, language=language, temperature=temperature,
        )
        if not txt:
            for t in temperature_fallbacks:
                txt = _decode_whisper_once(
                    processor, model, device, chunk, sr,
                    task=task, language=language, temperature=t,
                )
                if txt:
                    break
        if txt:
            full_text = _smart_join(full_text, txt.strip())
        if end == L:
            break
        start += hop
    return " ".join(full_text.split())


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------


def get_lang_dirs(cfg: LanguageConfig) -> Tuple[Path, Path]:
    base = DATASET_ROOT / cfg.folder
    return base / "audio", base / "transcripts"


def discover_encounter_ids(audio_dir: Path, audio_re: re.Pattern) -> List[int]:
    ids = []
    for f in audio_dir.iterdir():
        m = audio_re.match(f.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Per-language evaluation
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "language", "encounter_id", "audio_file", "transcript_file",
    "duration_s", "latency_s", "rtf",
    "ref_words", "hyp_words", "S", "D", "I",
    "WER", "Accuracy", "CER",
    "ref_text", "hyp_text",
]


def evaluate_language(
    cfg: LanguageConfig,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
    *,
    out_csv: Optional[str],
    model_id: str,
    task: str,
    chunk_length_s: float,
    overlap_s: float,
    dry_run: bool = False,
) -> None:
    audio_dir, transcripts_dir = get_lang_dirs(cfg)

    print(f"\n{'=' * 64}")
    print(f"  Language : {cfg.code.upper()} ({cfg.whisper_lang})")
    print(f"  Folder   : {DATASET_ROOT / cfg.folder}")
    print(f"  Model    : {model_id}  |  Task: {task}")
    print(f"{'=' * 64}")

    if not audio_dir.exists():
        print(
            f"[ERROR] Audio directory not found: {audio_dir}\n"
            f"        Check LANGUAGE_CONFIGS['{cfg.code}'].folder "
            f"or pass --lang_folder to override."
        )
        return

    encounter_ids = discover_encounter_ids(audio_dir, cfg.audio_re)
    if not encounter_ids:
        print(
            f"[ERROR] No files matched '{cfg.audio_re.pattern}' in {audio_dir}\n"
            f"        Use --audio_pattern to override the regex."
        )
        return

    preview = encounter_ids[:5]
    suffix = "..." if len(encounter_ids) > 5 else ""
    print(f"  Found {len(encounter_ids)} encounters: {preview}{suffix}")

    if dry_run:
        # Spot-check first encounter paths
        first = encounter_ids[0]
        ap = audio_dir / cfg.audio_tpl.format(id=first)
        tp = transcripts_dir / cfg.transcript_tpl.format(id=first)
        print(f"  [DRY RUN] Sample audio      : {ap}  {'✓' if ap.exists() else '✗ NOT FOUND'}")
        print(f"  [DRY RUN] Sample transcript : {tp}  {'✓' if tp.exists() else '✗ NOT FOUND'}")
        print("  [DRY RUN] Skipping ASR.")
        return

    rows: List[Dict] = []
    agg_S = agg_D = agg_I = agg_N = 0
    agg_cedits = agg_cref = 0
    total_dur = total_asr_time = 0.0

    for enc_id in encounter_ids:
        audio_path = audio_dir / cfg.audio_tpl.format(id=enc_id)
        txt_path = transcripts_dir / cfg.transcript_tpl.format(id=enc_id)

        if not txt_path.exists():
            print(f"  [WARN] No transcript for encounter {enc_id} → {txt_path.name}; skipping.")
            continue

        # Reference
        ref_norm = load_reference_from_txt(str(txt_path))
        ref_words_list = words(ref_norm)

        # Audio
        audio_in, sr_in = sf.read(str(audio_path), dtype="float32")
        audio_in = to_mono(audio_in)
        dur_s = len(audio_in) / float(sr_in) if sr_in else 0.0
        total_dur += dur_s
        audio_rs = resample_audio_linear(audio_in, sr_in, TARGET_SR)

        # ASR
        t0 = time.time()
        hyp_raw = transcribe_chunked(
            processor, model, device, audio_rs,
            sr=TARGET_SR, task=task,
            language=cfg.whisper_lang,
            chunk_length_s=chunk_length_s,
            overlap_s=overlap_s,
        )
        asr_time = time.time() - t0
        total_asr_time += asr_time
        rtf = asr_time / dur_s if dur_s > 0 else math.nan

        # Normalise hypothesis identically to reference
        hyp_norm = normalize_for_scoring(hyp_raw, lowercase=True, strip_punct=True)
        hyp_words_list = words(hyp_norm)

        # Metrics
        S, D, I, N = levenshtein_ops(ref_words_list, hyp_words_list)
        wer = (S + D + I) / N if N > 0 else math.nan
        acc = (N - S - D - I) / N if N > 0 else math.nan
        cedits, cref = cer_ops(
            ref_norm.replace(" ", ""), hyp_norm.replace(" ", "")
        )
        cer = cedits / cref if cref > 0 else math.nan

        agg_S += S; agg_D += D; agg_I += I; agg_N += N
        agg_cedits += cedits; agg_cref += cref

        rows.append({
            "language": cfg.code,
            "encounter_id": enc_id,
            "audio_file": audio_path.name,
            "transcript_file": txt_path.name,
            "duration_s": round(dur_s, 3),
            "latency_s": round(asr_time, 3),
            "rtf": round(rtf, 3) if not math.isnan(rtf) else "",
            "ref_words": N,
            "hyp_words": len(hyp_words_list),
            "S": S, "D": D, "I": I,
            "WER": round(wer, 4) if not math.isnan(wer) else "",
            "Accuracy": round(acc, 4) if not math.isnan(acc) else "",
            "CER": round(cer, 4) if not math.isnan(cer) else "",
            "ref_text": ref_norm,
            "hyp_text": hyp_norm,
        })

    if not rows:
        print("  [WARN] No encounters processed — check audio/transcript paths.")
        return

    # Aggregate row
    wer_micro = (agg_S + agg_D + agg_I) / agg_N if agg_N else math.nan
    cer_micro = agg_cedits / agg_cref if agg_cref else math.nan
    rtf_corpus = total_asr_time / total_dur if total_dur else math.nan

    rows.append({
        "language": cfg.code,
        "encounter_id": "",
        "audio_file": "__AGGREGATE__",
        "transcript_file": "",
        "duration_s": round(total_dur, 3),
        "latency_s": round(total_asr_time, 3),
        "rtf": round(rtf_corpus, 3) if not math.isnan(rtf_corpus) else "",
        "ref_words": agg_N,
        "hyp_words": "",
        "S": agg_S, "D": agg_D, "I": agg_I,
        "WER": round(wer_micro, 4) if not math.isnan(wer_micro) else "",
        "Accuracy": round((agg_N - agg_S - agg_D - agg_I) / agg_N, 4) if agg_N else "",
        "CER": round(cer_micro, 4) if not math.isnan(cer_micro) else "",
        "ref_text": "",
        "hyp_text": "",
    })

    # Output path
    if out_csv:
        out_path = Path(out_csv)
    else:
        model_tag = model_id.split("/")[-1] if "/" in model_id else model_id
        out_path = RESULTS_BASE.join(
            "results", name=f"asr_benchmark_{cfg.code}.{model_tag}.csv"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    agg = rows[-1]
    print(
        f"\n  ✓ Wrote {len(rows) - 1} encounter rows + aggregate → {out_path}"
        f"\n  Micro WER = {agg['WER']}  |  Micro CER = {agg['CER']}  |  RTF = {agg['rtf']}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

VALID_LANG_CHOICES = list(LANGUAGE_CONFIGS.keys()) + ["all"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark Whisper ASR on the Kaggle Multilingual Ambient Scribe dataset. "
            "Supports nl, fr, de, es, en, or all four non-English languages in one run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--language",
        default="all",
        choices=VALID_LANG_CHOICES,
        help=(
            "Language(s) to benchmark.  'all' runs nl, fr, de, es in sequence "
            "(English already done).  Default: all."
        ),
    )

    # ── Dataset override flags (single-language mode only) ────────────────
    ov = ap.add_argument_group(
        "Dataset overrides",
        "Adjust folder/filename conventions if they differ from the defaults.",
    )
    ov.add_argument(
        "--lang_folder", default=None,
        help="Override dataset sub-folder name.  E.g. --lang_folder 'Netherlands Dutch'",
    )
    ov.add_argument(
        "--audio_pattern", default=None,
        help="Override audio filename regex (group 1 must be the encounter ID).  "
             "E.g. --audio_pattern 'recording_nl_encounter_(\\d+)\\.mp3'",
    )
    ov.add_argument(
        "--audio_template", default=None,
        help="Override audio filename template with {id} placeholder.  "
             "E.g. --audio_template 'recording_nl_encounter_{id}.mp3'",
    )
    ov.add_argument(
        "--transcript_template", default=None,
        help="Override transcript filename template with {id} placeholder.  "
             "E.g. --transcript_template 'Encounter {id}_NL.txt'",
    )

    # ── Model / decode settings ───────────────────────────────────────────
    ap.add_argument(
        "--model_id", default="openai/whisper-small",
        help="HuggingFace model ID.  Avoid *.en variants — English-only.  "
             "Default: openai/whisper-small",
    )
    ap.add_argument(
        "--task", default="transcribe", choices=["transcribe", "translate"],
        help="Whisper task.  Use 'transcribe' for native-language output.  Default: transcribe",
    )
    ap.add_argument(
        "--chunk_length_s", type=float, default=29.5,
        help="Chunk length in seconds for chunked decoding.  Default: 29.5",
    )
    ap.add_argument(
        "--overlap_s", type=float, default=1.0,
        help="Overlap between consecutive chunks in seconds.  Default: 1.0",
    )
    ap.add_argument(
        "--device", default=None,
        help="Torch device override: 'cpu', 'mps', 'cuda', 'cuda:1'.  "
             "Default: auto (cuda → mps → cpu)",
    )

    # ── Output ────────────────────────────────────────────────────────────
    ap.add_argument(
        "--out_csv", default=None,
        help="Explicit output CSV path.  Single-language mode only; "
             "ignored when --language all is used.  Default: pystow cache.",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Check discovered paths and spot-check first encounter files "
             "without running any ASR.  Run this first.",
    )

    args = ap.parse_args()

    # Resolve language list
    if args.language == "all":
        lang_codes = ALL_LANGUAGES
        if args.out_csv:
            print("[WARN] --out_csv is ignored in 'all' mode; per-language CSVs are saved automatically.")
    else:
        lang_codes = [args.language]

    # Apply single-language overrides via dataclasses.replace (no mutation of defaults)
    if len(lang_codes) == 1:
        code = lang_codes[0]
        cfg = LANGUAGE_CONFIGS[code]
        if args.lang_folder:
            cfg = replace(cfg, folder=args.lang_folder)
        if args.audio_pattern:
            cfg = replace(cfg, audio_re=re.compile(args.audio_pattern))
        if args.audio_template:
            cfg = replace(cfg, audio_tpl=args.audio_template)
        if args.transcript_template:
            cfg = replace(cfg, transcript_tpl=args.transcript_template)
        LANGUAGE_CONFIGS[code] = cfg

    # Load model once; reuse across all languages
    if not args.dry_run:
        print(f"\nLoading model : {args.model_id}")
        processor, model, device = load_model(args.model_id, device=args.device)
    else:
        processor = model = None   # type: ignore[assignment]
        device = best_device(args.device)
        print(f"\n[DRY RUN]  device would be: {device}")

    for code in lang_codes:
        evaluate_language(
            cfg=LANGUAGE_CONFIGS[code],
            processor=processor,
            model=model,
            device=device,
            out_csv=args.out_csv if len(lang_codes) == 1 else None,
            model_id=args.model_id,
            task=args.task,
            chunk_length_s=args.chunk_length_s,
            overlap_s=args.overlap_s,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()