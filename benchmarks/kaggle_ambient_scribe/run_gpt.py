#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch benchmark Whisper ASR against plain-text transcripts with direct basename matching,
16 kHz resampling, and chunked decoding to capture full-length transcriptions.

Direct match rule:
  Audio:       <number>.mp3 OR <number>.wav (e.g., 1.mp3, 6.wav)
  Transcript:  <number>.txt                 (e.g., 1.txt, 6.txt)

Transcript preprocessing (reference):
  - Drop first line if it equals 'Transcripted Line' (case-insensitive; trims spaces).
  - Remove punctuation.
  - Lowercase and collapse whitespace.

Hypothesis preprocessing:
  - Apply the SAME normalization as reference before scoring.

Metrics:
  - WER with S/D/I and N, Accuracy, CER
  - Duration (sec), ASR latency (sec), Real-Time Factor (RTF = latency/duration)
  - Per-file rows + one aggregate row

Usage:
  python benchmark_asr_from_txt.py \
      --audio_dir ./audio \
      --transcript_dir ./transcripts \
      --out_csv ./asr_benchmark_results.csv \
      --model_id openai/whisper-tiny \
      --task transcribe \
      --chunk_length_s 29.5 \
      --overlap_s 1.0
"""

import os
import re
import csv
import time
import math
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# -------------------------------
# Constants
# -------------------------------

TARGET_SR = 16000  # Whisper expects 16 kHz


# -------------------------------
# Audio utilities (mono + resample to 16k)
# -------------------------------

def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert (n,) or (n, ch) -> (n,) mono (float32).
    """
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1).astype(np.float32, copy=False)


def resample_audio_linear(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Dependency-free linear interpolation resampler to target_sr.
    Sufficient for ASR preprocessing; avoids extra packages.
    """
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    duration = len(audio) / float(orig_sr)
    tgt_len = int(round(duration * target_sr))
    if tgt_len <= 1:
        return audio[:1].astype(np.float32, copy=False)

    # time bases
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, duration, num=tgt_len, endpoint=False, dtype=np.float64)
    rs = np.interp(x_new, x_old, audio).astype(np.float32)
    return rs


# -------------------------------
# Normalization (text)
# -------------------------------

# Remove all punctuation: keep letters/digits/underscore/whitespace only
PUNCT_STRIP = re.compile(r"[^\w\s]", flags=re.UNICODE)

def remove_punctuation(text: str) -> str:
    return PUNCT_STRIP.sub(" ", text)

def normalize_for_scoring(text: str, lowercase: bool = True, strip_punct: bool = True) -> str:
    t = text
    if lowercase:
        t = t.lower()
    if strip_punct:
        t = remove_punctuation(t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def words(text: str) -> List[str]:
    return text.split()


# -------------------------------
# Reference transcript loading
# -------------------------------

def load_reference_from_txt(path: str) -> str:
    """
    Load a transcript, remove the first line if it equals 'Transcripted Line',
    remove punctuation, lowercase, and collapse whitespace.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    # Drop leading empty lines
    while lines and lines[0].strip() == "":
        lines.pop(0)

    # Remove the header line 'Transcripted Line' if present
    if lines and lines[0].strip().lower() == "transcripted line":
        lines = lines[1:]

    raw = "\n".join(lines)
    ref_norm = normalize_for_scoring(raw, lowercase=True, strip_punct=True)
    return ref_norm


# -------------------------------
# Edit-distance metrics
# -------------------------------

def levenshtein_ops(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """Return (S, D, I, N): substitutions, deletions, insertions, and reference length."""
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    op = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i; op[i][0] = 'D'
    for j in range(1, m+1):
        dp[0][j] = j; op[0][j] = 'I'
    for i in range(1, n+1):
        ri = ref[i-1]
        for j in range(1, m+1):
            hj = hyp[j-1]
            if ri == hj:
                dp[i][j] = dp[i-1][j-1]; op[i][j] = 'E'
            else:
                sub = dp[i-1][j-1] + 1
                ins = dp[i][j-1] + 1
                dele = dp[i-1][j] + 1
                best = min(sub, ins, dele)
                dp[i][j] = best
                op[i][j] = 'S' if best == sub else ('I' if best == ins else 'D')
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        cur = op[i][j]
        if cur == 'E':
            i -= 1; j -= 1
        elif cur == 'S':
            S += 1; i -= 1; j -= 1
        elif cur == 'I':
            I += 1; j -= 1
        elif cur == 'D':
            D += 1; i -= 1
        else:
            break
    return S, D, I, n

def cer_ops(ref_text: str, hyp_text: str) -> Tuple[int, int]:
    """
    Return (edits, ref_len) for CER after normalization (punctuation removed, lowercase).
    """
    r = list(ref_text)
    h = list(hyp_text)
    n, m = len(r), len(h)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        ri = r[i-1]
        for j in range(1, m+1):
            hj = h[j-1]
            cost = 0 if ri == hj else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m], n


# -------------------------------
# Whisper model (mirrors run_test.py style)
# -------------------------------

def load_model(model_id: str, device: Optional[str] = None):
    """
    Load Whisper processor and model (default: openai/whisper-tiny).
    """
    # Optional: set a local HF cache like in your original script:
    # os.environ['HF_HOME'] = './hf_models_cache'
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


# -------------------------------
# Chunked decoding (to avoid 30s truncation)
# -------------------------------

def _decode_whisper_once(
    processor,
    model,
    device,
    audio_chunk: np.ndarray,
    sr: int,
    task: str = "transcribe",
    temperature: float = 0.0,
) -> str:
    """
    One decode call for a single <=30s chunk.
    """
    inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt")
    gen_kwargs = {"task": task, "temperature": temperature}
    with torch.no_grad():
        pred_ids = model.generate(inputs.input_features.to(device), **gen_kwargs)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


def _seconds_to_samples(seconds: float, sr: int) -> int:
    return int(round(seconds * sr))


def _smart_join(prev: str, new: str, max_overlap_words: int = 8) -> str:
    """
    Join two text segments while trimming a repeated overlap (simple prefix/suffix check).
    """
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
    processor,
    model,
    device,
    audio_rs: np.ndarray,      # float32 mono @ 16k
    sr: int = 16000,
    task: str = "transcribe",
    chunk_length_s: float = 29.5,   # safe under ~30s
    overlap_s: float = 1.0,         # small overlap to avoid cutting words
    temperature: float = 0.0,       # greedy first
    temperature_fallbacks: Tuple[float, ...] = (0.2, 0.4),  # optional retries for blank chunks
) -> str:
    """
    Slide a 30s window across the audio with overlap. Concatenate decoded text with
    a simple overlap-aware join to minimize duplicates.
    """
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

        # Greedy decode
        txt = _decode_whisper_once(processor, model, device, chunk, sr, task=task, temperature=temperature)

        # Fallbacks if blank/low-info
        if not txt or len(txt.strip()) == 0:
            for t in temperature_fallbacks:
                txt = _decode_whisper_once(processor, model, device, chunk, sr, task=task, temperature=t)
                if txt and len(txt.strip()) > 0:
                    break

        if txt:
            full_text = _smart_join(full_text, txt.strip())

        if end == L:
            break
        start += hop

    # collapse whitespace to be consistent
    return " ".join(full_text.split())


# -------------------------------
# Main evaluation (direct numeric match)
# -------------------------------

def evaluate(
    audio_dir: str,
    transcript_dir: str,
    out_csv: str,
    model_id: str = "openai/whisper-tiny",
    task: str = "transcribe",
    chunk_length_s: float = 29.5,
    overlap_s: float = 1.0,
) -> None:
    processor, model, device = load_model(model_id)

    rows: List[Dict[str, object]] = []
    agg_S = agg_D = agg_I = agg_N = 0
    agg_cedits = agg_cref = 0
    total_dur = 0.0
    total_asr_time = 0.0

    # Collect audio files with numeric basenames (allow .mp3 and .wav)
    audio_files = []
    for fname in os.listdir(audio_dir):
        fpath = os.path.join(audio_dir, fname)
        if not os.path.isfile(fpath):
            continue
        lower = fname.lower()
        if lower.endswith(".mp3") or lower.endswith(".wav"):
            stem = os.path.splitext(fname)[0]
            if re.fullmatch(r"\d+", stem):  # numeric basename only
                audio_files.append(fname)

    # Sort numerically (1,2,3,...)
    audio_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    for fname in audio_files:
        base_num = os.path.splitext(fname)[0]  # e.g., "1"
        audio_path = os.path.join(audio_dir, fname)
        txt_path = os.path.join(transcript_dir, f"{base_num}.txt")

        if not os.path.exists(txt_path):
            print(f"[WARN] Transcript not found for {fname} -> expected {base_num}.txt; skipping.")
            continue

        # Load & normalize reference
        ref_norm = load_reference_from_txt(txt_path)
        ref_words_list = words(ref_norm)

        # ---- Load audio raw (keep original sampling rate for duration) ----
        audio_in, sr_in = sf.read(audio_path, dtype="float32")
        audio_in = to_mono(audio_in)

        dur_s = len(audio_in) / float(sr_in) if sr_in and sr_in > 0 else 0.0  # original duration for RTF
        total_dur += dur_s

        # ---- Force resample to TARGET_SR mono for Whisper ----
        audio_rs = resample_audio_linear(audio_in, sr_in, TARGET_SR)
        sr = TARGET_SR

        # ASR with chunking
        t0 = time.time()
        hyp_raw = transcribe_chunked(
            processor, model, device,
            audio_rs, sr=sr, task=task,
            chunk_length_s=chunk_length_s, overlap_s=overlap_s,
        )
        asr_time = time.time() - t0
        total_asr_time += asr_time
        rtf = asr_time / dur_s if dur_s > 0 else math.nan

        # Normalize hypothesis identically to reference
        hyp_norm = normalize_for_scoring(hyp_raw, lowercase=True, strip_punct=True)
        hyp_words_list = words(hyp_norm)

        # Metrics
        S, D, I, N = levenshtein_ops(ref_words_list, hyp_words_list)
        wer = (S + D + I) / N if N > 0 else math.nan
        acc = (N - S - D - I) / N if N > 0 else math.nan

        cedits, cref = cer_ops(ref_norm.replace(" ", ""), hyp_norm.replace(" ", ""))
        cer = (cedits / cref) if cref > 0 else math.nan

        agg_S += S; agg_D += D; agg_I += I; agg_N += N
        agg_cedits += cedits; agg_cref += cref

        rows.append({
            "audio_file": fname,
            "transcript_file": f"{base_num}.txt",
            "duration_s": round(dur_s, 3),
            "latency_s": round(asr_time, 3),
            "rtf": round(rtf, 3) if not math.isnan(rtf) else "",
            "ref_words": N,
            "hyp_words": len(hyp_words_list),
            "S": S, "D": D, "I": I,
            "WER": round(wer, 4) if wer == wer else "",
            "Accuracy": round(acc, 4) if acc == acc else "",
            "CER": round(cer, 4) if cer == cer else "",
            "ref_text": ref_norm,
            "hyp_text": hyp_norm,
        })

    # Aggregate row
    wer_micro = ((agg_S + agg_D + agg_I) / agg_N) if agg_N > 0 else math.nan
    cer_micro = (agg_cedits / agg_cref) if agg_cref > 0 else math.nan
    rtf_corpus = (total_asr_time / total_dur) if total_dur > 0 else math.nan

    rows.append({
        "audio_file": "__AGGREGATE__",
        "transcript_file": "",
        "duration_s": round(total_dur, 3),
        "latency_s": round(total_asr_time, 3),
        "rtf": round(rtf_corpus, 3) if rtf_corpus == rtf_corpus else "",
        "ref_words": agg_N,
        "hyp_words": "",
        "S": sum(r["S"] for r in rows[:-1]) if rows[:-1] else 0,
        "D": sum(r["D"] for r in rows[:-1]) if rows[:-1] else 0,
        "I": sum(r["I"] for r in rows[:-1]) if rows[:-1] else 0,
        "WER": round(wer_micro, 4) if wer_micro == wer_micro else "",
        "Accuracy": round((agg_N - agg_S - agg_D - agg_I)/agg_N, 4) if agg_N > 0 else "",
        "CER": round(cer_micro, 4) if cer_micro == cer_micro else "",
        "ref_text": "",
        "hyp_text": "",
    })

    # Write CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None
    fieldnames = [
        "audio_file","transcript_file","duration_s","latency_s","rtf",
        "ref_words","hyp_words","S","D","I","WER","Accuracy","CER",
        "ref_text","hyp_text"
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Done. Wrote {len(rows)-1} file rows + 1 aggregate to {out_csv}")
    if agg_N > 0:
        print(f"Micro WER={rows[-1]['WER']}  |  Micro CER={rows[-1]['CER']}  |  Corpus RTF={rows[-1]['rtf']}")
    else:
        print("No references processed; verify that <number>.txt exists for each <number>.mp3/.wav.")


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark Whisper ASR vs. transcripts with direct basename matching (N.* ↔ N.txt), "
                    "with 16k resampling and chunked decoding."
    )
    ap.add_argument("--audio_dir", required=True, help="Directory with audio files named <number>.mp3 or <number>.wav.")
    ap.add_argument("--transcript_dir", required=True, help="Directory with transcripts named <number>.txt.")
    ap.add_argument("--out_csv", required=True, help="Path to write metrics CSV.")
    ap.add_argument("--model_id", default="openai/whisper-tiny", help="HF model ID (default: openai/whisper-tiny).")
    ap.add_argument("--task", default="transcribe", choices=["transcribe","translate"], help="Whisper task.")
    ap.add_argument("--chunk_length_s", type=float, default=29.5, help="Chunk length (seconds) per decode window.")
    ap.add_argument("--overlap_s", type=float, default=1.0, help="Overlap (seconds) between consecutive chunks.")
    args = ap.parse_args()

    evaluate(
        audio_dir=args.audio_dir,
        transcript_dir=args.transcript_dir,
        out_csv=args.out_csv,
        model_id=args.model_id,
        task=args.task,
        chunk_length_s=args.chunk_length_s,
        overlap_s=args.overlap_s,
    )

if __name__ == "__main__":
    main()
