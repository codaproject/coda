"""Benchmark CODA's speech-to-text and cause-of-death inference on a Ryzen AI PC.

Ryzen/Windows counterpart of the Mac benchmark. Runs the same two CODA stages with
the same prompt, schema, narratives, and clip, so results plot against the Macs:
  1. Speech-to-text: WhisperLiveKit (faster-whisper, CPU) fed the clip as real-time PCM.
  2. Inference: the CHAMPS system prompt, COD_OUTPUT_SCHEMA, and schema-constrained
     decoding, across the models, served by Lemonade over its OpenAI-compatible API
     (NPU + iGPU hybrid for ONNX models, llama.cpp/Vulkan for GGUF).

The vendored CHAMPS prompt, schema, and request shapes (coda_snapshot.py + champs/)
are copied byte-for-byte from the Mac benchmark, so the only difference is hardware
and serving backend. Reports STT real-time keep-up plus per-model warm latency,
validity, and the predicted top cause, tagged with the machine's hardware.
"""
import argparse
import asyncio
import json
import statistics as stats
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import coda_snapshot
from hwinfo import hardware_info
from stt_bench import run_stream as stt_run_stream

HERE = Path(__file__).resolve().parent
LEMONADE_BASE = "http://localhost:13305/api/v1"

# Verify the exact ids against your Lemonade install (Model Manager in the app,
# or `lemonade-server list`). Match family/size/quant to the Mac rows so the
# comparison is hardware, not model. Edit ids here to whatever you pulled.
MODELS = [
    {"name": "Qwen2.5-7B-Instruct", "backend": "lemonade",
     "id": "Qwen2.5-7B-Instruct-Hybrid"},
    {"name": "gpt-oss-20b", "backend": "lemonade", "id": "gpt-oss-20b-GGUF"},
    {"name": "gemma-3-27b", "backend": "lemonade", "id": "gemma-3-27b-it-GGUF"},
]


def wait_ready(url, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def infer_once(model, system, user):
    if model["backend"] == "ollama":
        return coda_snapshot.ollama_infer(model["id"], system, user,
            think=model.get("think"))
    return coda_snapshot.openai_infer(model["id"], system, user,
        base_url=LEMONADE_BASE)


def top_cause(resp):
    causes = resp.get("top_causes") or []
    return causes[0]["cause_name"] if causes else None


def bench_model(model, narratives, system):
    result = {"name": model["name"], "backend": model["backend"],
              "id": model["id"], "cases": []}
    try:
        infer_once(model, system, coda_snapshot.user_prompt(narratives[0]))
    except Exception:
        pass
    for i, narrative in enumerate(narratives):
        user = coda_snapshot.user_prompt(narrative)
        t0 = time.time()
        try:
            resp = infer_once(model, system, user)
            valid, top = True, top_cause(resp)
        except Exception:
            valid, top = False, None
        result["cases"].append({"i": i, "latency_sec": round(time.time() - t0, 2),
                                "valid": valid, "top_cause": top})

    lat = [c["latency_sec"] for c in result["cases"] if c["valid"]]
    n = len(result["cases"])
    result["valid_rate"] = sum(c["valid"] for c in result["cases"]) / n if n else 0
    result["mean_latency_sec"] = round(stats.mean(lat), 2) if lat else None
    result["median_latency_sec"] = round(stats.median(lat), 2) if lat else None
    result["stdev_latency_sec"] = round(stats.stdev(lat), 2) if len(lat) > 1 else None
    result["min_latency_sec"] = round(min(lat), 2) if lat else None
    result["max_latency_sec"] = round(max(lat), 2) if lat else None
    return result


def run_stt(audio, reps):
    runs, err = [], None
    for _ in range(reps):
        try:
            runs.append(asyncio.run(stt_run_stream(audio, "small", "faster-whisper",
                "localagreement", "en")))
        except Exception as e:
            err = str(e)[:200]
            break
    if not runs:
        return {"error": err or "no runs"}
    keeps = [r["keep_up"] for r in runs if r.get("keep_up") is not None]
    return {"reps": len(runs), "clip_sec": runs[0].get("clip_sec"),
            "keep_up_mean": round(stats.mean(keeps), 3) if keeps else None,
            "keep_up_median": round(stats.median(keeps), 3) if keeps else None,
            "peak_gb": round(max(r.get("peak_gb") or 0 for r in runs), 2),
            "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--narratives", default=str(HERE / "narratives.json"))
    ap.add_argument("--audio", default=str(HERE / "assets" / "clip.wav"))
    ap.add_argument("--models", nargs="*", default=None,
        help="subset of model names to run")
    ap.add_argument("--skip-stt", action="store_true",
        help="skip the speech-to-text stage")
    ap.add_argument("--stt-reps", type=int, default=3,
        help="how many real-time STT passes to average")
    args = ap.parse_args()

    if not wait_ready(f"{LEMONADE_BASE}/models", 5):
        print(f"WARNING: Lemonade not reachable at {LEMONADE_BASE}. "
              "Start it with `lemonade-server serve` and confirm the port.\n")

    narratives = json.loads(Path(args.narratives).read_text())
    system = coda_snapshot.build_system_prompt()
    hw = hardware_info()
    print(f"Hardware: {hw['chip']} {hw['ram_gb']}GB {hw['bandwidth_gbps']}GB/s")
    print(f"Narratives: {len(narratives)}\n")

    report = {"hardware": hw, "timestamp": datetime.now().isoformat(timespec="seconds"),
              "n_narratives": len(narratives), "stt": None, "models": []}

    if not args.skip_stt:
        print(f"[speech-to-text] WhisperLiveKit faster-whisper small x{args.stt_reps}...")
        stt = run_stt(args.audio, args.stt_reps)
        if "error" in stt:
            print(f"  ERROR: {stt['error']}\n")
        else:
            print(f"  keep_up mean={stt['keep_up_mean']} median={stt['keep_up_median']} "
                  f"(<=1.0 real time)  peak={stt['peak_gb']}GB\n")
        report["stt"] = stt

    models = [m for m in MODELS if not args.models or m["name"] in args.models]
    for m in models:
        print(f"[{m['name']}] ({m['backend']})...")
        try:
            r = bench_model(m, narratives, system)
        except Exception as e:
            r = {"name": m["name"], "backend": m["backend"], "error": str(e)[:200]}
        report["models"].append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}\n")
        else:
            print(f"  valid={r['valid_rate']*100:.0f}%  mean={r['mean_latency_sec']}s  "
                  f"median={r['median_latency_sec']}s\n")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    tag = (hw["chip"] or "ryzen").replace(" ", "_")
    out = out_dir / f"{tag}_{report['timestamp'].replace(':', '')}.json"
    out.write_text(json.dumps(report, indent=2))

    print("=" * 68)
    stt = report["stt"]
    if stt and "error" not in stt:
        print(f"speech-to-text (WhisperLiveKit faster-whisper small, {stt['reps']} reps): "
              f"keep_up={stt.get('keep_up_mean')}  peak={stt.get('peak_gb')}GB")
    elif stt:
        print(f"speech-to-text: ERROR {stt['error']}")
    print(f"{'model':<40}{'valid':>7}{'mean_s':>9}{'med_s':>7}{'sd':>6}")
    for r in report["models"]:
        if "error" in r:
            print(f"{r['name']:<40}{'ERR':>7}")
        elif r.get("mean_latency_sec") is None:
            print(f"{r['name']:<40}{r['valid_rate']*100:>6.0f}%{'-':>9}")
        else:
            sd = r.get("stdev_latency_sec")
            print(f"{r['name']:<40}{r['valid_rate']*100:>6.0f}%"
                  f"{r['mean_latency_sec']:>9}{r['median_latency_sec']:>7}"
                  f"{(sd if sd is not None else 0):>6}")
    print(f"\nReport written to {out}")
    print("Send that JSON file back to the group.")


if __name__ == "__main__":
    main()
