"""Benchmark Whisper ASR on the Brazilian Portuguese COD clips (WER vs narrative).

Each clip's reference is the matching case's Portuguese narrative in
cases_ptbr_filtered.json. Two backends are supported, openai-whisper and
faster-whisper, both run with language="pt". Word-level WER is computed
in-process (no jiwer). Run with no args for all engines, or pass engine names.
"""
import argparse
import json
import os
import re
import subprocess
import time
import unicodedata
from pathlib import Path

from dataset_io import load_samples
from languages.pt_br import normalize as normalize_portuguese
from metrics import wer_details as metric_wer_details

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BASE = Path(__file__).resolve().parent
SIZES = ["tiny", "base", "small", "medium", "large-v3"]


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


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(t, strip_accents=False):
    t = t.lower()
    if strip_accents:
        t = "".join(c for c in unicodedata.normalize("NFD", t)
                    if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", t)).strip()


def wer(ref, hyp, strip_accents=False):
    r, h = norm(ref, strip_accents).split(), norm(hyp, strip_accents).split()
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


# Shared edit-distance accounting keeps the two language runners comparable.
def wer(ref, hyp, strip_accents=False):
    return metric_wer_details(
        ref, hyp, lambda text: normalize_portuguese(text, strip_accents)
    )[0]


def samples():
    return [(s.case_id, str(s.audio_path), s.reference, clip_duration(s.audio_path))
            for s in load_samples("pt-BR")]


def make_whisper(size, language, device):
    import whisper
    model = whisper.load_model(size, device=device)

    def run(path):
        return model.transcribe(str(path), language=language,
                                fp16=(device != "cpu"))["text"]
    return run


def make_faster_whisper(size, language, device, compute_type):
    from faster_whisper import WhisperModel
    model = WhisperModel(size, device=device, compute_type=compute_type)

    def run(path):
        segments, _ = model.transcribe(str(path), language=language)
        return " ".join(s.text for s in segments)
    return run


def build_engines(args):
    engines = {}
    for size in SIZES:
        engines[f"whisper-{size}"] = \
            lambda size=size: make_whisper(size, args.language, args.device)
        engines[f"faster-whisper-{size}"] = \
            lambda size=size: make_faster_whisper(size, args.language,
                                                  args.fw_device, args.compute_type)
    return engines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*", default=None,
                    help="engine names to run (default: all)")
    ap.add_argument("--language", default="pt", help="Whisper language code")
    ap.add_argument("--device", default="cpu",
                    help="torch device for openai-whisper")
    ap.add_argument("--fw-device", default="cpu",
                    help="ctranslate2 device for faster-whisper")
    # float32 keeps faster-whisper comparable to openai-whisper, "auto" would
    # silently pick int8 on CPU
    ap.add_argument("--compute-type", default="float32",
                    help="faster-whisper compute type, e.g. int8 or float32")
    ap.add_argument("--strip-accents", action="store_true",
                    help="ignore diacritics when scoring")
    args = ap.parse_args()

    all_engines = build_engines(args)
    which = args.engines or list(all_engines)
    unknown = [n for n in which if n not in all_engines]
    if unknown:
        ap.error(f"unknown engines: {unknown}. Available: {list(all_engines)}")

    data = samples()
    hw = hardware()
    print(f"Hardware: {hw['chip']} {hw['ram_gb']}GB  language={args.language}  "
          f"clips={len(data)}", flush=True)
    for name in which:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            fn = all_engines[name]()
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
                w = wer(ref, hyp, args.strip_accents)
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
            out = BASE / "results" / "pt-BR" / f"transcripts_{name}.json"
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(
                {"engine": name, "language": args.language, "hardware": hw,
                 "compute_type": args.compute_type, "load_sec": load_s,
                 "clips": clips},
                ensure_ascii=False, indent=2))
            print(f"  transcripts -> {out}", flush=True)


if __name__ == "__main__":
    main()
