"""Benchmark Bangla ASR models on the COD audio clips (WER vs bn_narrative).

Each clip's reference is the matching case's bn_narrative in coda-audio/cases_bn.json.
Whisper-based models run through the transformers ASR pipeline with long-form
chunking; indic-seamless and indic-conformer use their own code paths. Word-level
WER is computed in-process (no jiwer). Run with no args for all engines, or pass
engine names.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

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
CASES = {c["case_id"]: c["bn_narrative"]
         for c in json.loads((BASE / "coda-audio" / "cases_bn.json").read_text())}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(t):
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", t)).strip()


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    return dp[n][m] / n if n else float("nan")


def samples():
    out = []
    for f in sorted(glob.glob(str(BASE / "coda-audio" / "*.m4a"))):
        cid = re.sub(r".*cod-case_id_|\.m4a", "", f)
        if cid in CASES:
            out.append((cid, f, CASES[cid], clip_duration(f)))
    return out


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


def make_whisper(repo):
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    dev = _device()
    proc = AutoProcessor.from_pretrained(repo)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(repo).to(dev).eval()
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
    from transformers import AutoProcessor, SeamlessM4Tv2Model
    proc = AutoProcessor.from_pretrained(repo)
    model = SeamlessM4Tv2Model.from_pretrained(repo).to(_device())

    def run(path):
        inputs = proc(audio=load_audio(path), sampling_rate=16000,
                      return_tensors="pt").to(_device())
        out = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        return proc.decode(out[0].tolist()[0], skip_special_tokens=True)
    return run


def make_conformer(repo):
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(repo, trust_remote_code=True)

    def run(path):
        wav = torch.from_numpy(load_audio(path)).unsqueeze(0)
        return model(wav, "bn", "ctc")
    return run


def make_bst(size):
    from banglaspeech2text import Speech2Text
    stt = Speech2Text(size)
    return lambda path: stt.recognize(path)


def make_mlx(repo):
    import mlx_whisper
    return lambda path: mlx_whisper.transcribe(
        path, path_or_hf_repo=repo, language="bn")["text"]


ENGINES = {
    "mlx-whisper-small": lambda: make_mlx("mlx-community/whisper-small-mlx"),
    "banglaspeech2text-base": lambda: make_bst("base"),
    "banglaspeech2text-large": lambda: make_bst("large"),
    "indic-whisper": lambda: make_whisper("parthiv11/indic_whisper_nodcil"),
    "tugstugi-regional-medium": lambda: make_whisper(
        "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"),
    "tugstugi-medium": lambda: make_whisper(
        "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"),
    "bangla-whisper-large-v3": lambda: make_whisper(
        "utshobs/bangla_whisper_large_v3_finetuned"),
    "indic-seamless": lambda: make_seamless("ai4bharat/indic-seamless"),
    "indic-conformer": lambda: make_conformer(
        "ai4bharat/indic-conformer-600m-multilingual"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*", default=None,
                    help="engine names to run (default: all)")
    args = ap.parse_args()
    which = args.engines or list(ENGINES)
    data = samples()
    hw = hardware()
    print(f"Hardware: {hw['chip']} {hw['ram_gb']}GB  device={_device()}", flush=True)
    for name in which:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            fn = ENGINES[name]()
        except Exception as e:
            print(f"  engine load failed: {str(e)[:150]}")
            continue
        load_s = round(time.time() - t0, 1)
        wers, rtfs, clips = [], [], []
        for cid, path, ref, dur in data:
            try:
                t1 = time.time()
                hyp = fn(path)
                dt = time.time() - t1
                w = wer(ref, hyp)
                rtf = dt / dur if dur else None
                wers.append(w)
                if rtf is not None:
                    rtfs.append(rtf)
                clips.append({"case_id": cid, "wer": round(w, 3),
                              "audio_sec": round(dur, 1) if dur else None,
                              "time_sec": round(dt, 2),
                              "rtf": round(rtf, 3) if rtf else None,
                              "ref": ref, "hyp": hyp})
                print(f"  {cid:<14} WER={w:.3f}  {dt:5.1f}s  "
                      f"RTF={rtf:.2f}" if rtf else f"  {cid:<14} WER={w:.3f}",
                      flush=True)
            except Exception as e:
                print(f"  {cid:<14} ERROR {str(e)[:90]}", flush=True)
        if wers:
            mean_rtf = sum(rtfs) / len(rtfs) if rtfs else None
            print(f"  MEAN WER={sum(wers)/len(wers):.3f}  load={load_s}s  "
                  f"mean_RTF={mean_rtf:.2f}  (n={len(wers)})" if mean_rtf else
                  f"  MEAN WER={sum(wers)/len(wers):.3f}  load={load_s}s  "
                  f"(n={len(wers)})", flush=True)
        if clips:
            out = BASE / "results" / f"transcripts_{name}.json"
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(
                {"engine": name, "hardware": hw, "load_sec": load_s, "clips": clips},
                ensure_ascii=False, indent=2))
            print(f"  transcripts -> {out}", flush=True)


if __name__ == "__main__":
    main()
