"""Benchmark CODA's speech-to-text and cause-of-death inference on a laptop.

Runs the two stages of CODA's pipeline the way CODA runs them, identically on every
machine so results are comparable:
  1. Speech-to-text: WhisperLiveKit fed the shared clip as real-time PCM.
  2. Inference: the CHAMPS system prompt, COD_OUTPUT_SCHEMA, and schema-constrained
     decoding, across the config's models.

The CHAMPS prompt, schema, request shapes (coda_snapshot.py + champs/), narratives,
and clip are shared. Only the per-platform model list and serving backend differ,
and those live in configs/. The runner auto-selects mac vs ryzen from the OS
(override with --config). Writes one report per run to the shared results/ folder,
tagged with the machine's hardware, for plot_results.py to chart together.
"""
import argparse
import asyncio
import json
import re
import statistics as stats
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import coda_snapshot
import stt_bench
from configs import load as load_config
from hwinfo import hardware_info
from stt_bench import run_stream as stt_run_stream

HERE = Path(__file__).resolve().parent
MLX_PORT = 8080


def wait_ready(url, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def start_mlx_server(model_id):
    subprocess.run(["pkill", "-f", "mlx-openai-server"], capture_output=True)
    time.sleep(2)
    proc = subprocess.Popen(
        ["mlx-openai-server", "launch", "--model-path", model_id,
         "--model-type", "lm", "--port", str(MLX_PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_ready(f"http://localhost:{MLX_PORT}/v1/models", 180):
        proc.terminate()
        raise RuntimeError("mlx-openai-server did not become ready")
    return proc


def infer_once(model, system, user, lemonade_base):
    backend = model["backend"]
    if backend == "ollama":
        return coda_snapshot.ollama_infer(model["id"], system, user,
            think=model.get("think"))
    if backend == "mlx":
        return coda_snapshot.openai_infer(model["id"], system, user,
            base_url=f"http://localhost:{MLX_PORT}/v1")
    return coda_snapshot.openai_infer(model["id"], system, user,
        base_url=lemonade_base)


def top_cause(resp):
    causes = resp.get("top_causes") or []
    return causes[0]["cause_name"] if causes else None


def bench_model(model, narratives, system, lemonade_base):
    result = {"name": model["name"], "backend": model["backend"],
              "id": model["id"], "cases": []}
    proc = None
    if model["backend"] == "mlx":
        proc = start_mlx_server(model["id"])
    try:
        try:
            infer_once(model, system, coda_snapshot.user_prompt(narratives[0]),
                lemonade_base)
        except Exception:
            pass
        for i, narrative in enumerate(narratives):
            user = coda_snapshot.user_prompt(narrative)
            t0 = time.time()
            try:
                resp = infer_once(model, system, user, lemonade_base)
                valid, top = True, top_cause(resp)
            except Exception:
                valid, top = False, None
            result["cases"].append({"i": i, "latency_sec": round(time.time() - t0, 2),
                                    "valid": valid, "top_cause": top})
    finally:
        if proc is not None:
            subprocess.run(["pkill", "-f", "mlx-openai-server"], capture_output=True)

    lat = [c["latency_sec"] for c in result["cases"] if c["valid"]]
    n = len(result["cases"])
    result["valid_rate"] = sum(c["valid"] for c in result["cases"]) / n if n else 0
    result["mean_latency_sec"] = round(stats.mean(lat), 2) if lat else None
    result["median_latency_sec"] = round(stats.median(lat), 2) if lat else None
    result["stdev_latency_sec"] = round(stats.stdev(lat), 2) if len(lat) > 1 else None
    result["min_latency_sec"] = round(min(lat), 2) if lat else None
    result["max_latency_sec"] = round(max(lat), 2) if lat else None
    return result


def run_stt(audio, reps, backend):
    runs, err = [], None
    for _ in range(reps):
        try:
            runs.append(asyncio.run(stt_run_stream(audio, "small", backend,
                "localagreement", "en")))
        except Exception as e:
            err = str(e)[:200]
            break
    if not runs:
        return {"error": err or "no runs"}
    keeps = [r["keep_up"] for r in runs if r.get("keep_up") is not None]
    return {"backend": backend, "reps": len(runs), "clip_sec": runs[0].get("clip_sec"),
            "keep_up_mean": round(stats.mean(keeps), 3) if keeps else None,
            "keep_up_median": round(stats.median(keeps), 3) if keeps else None,
            "peak_gb": round(max(r.get("peak_gb") or 0 for r in runs), 2),
            "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, choices=("mac", "ryzen"),
        help="platform config (default: auto-detect from OS)")
    ap.add_argument("--narratives", default=str(HERE / "narratives.json"))
    ap.add_argument("--audio", default=str(HERE / "assets" / "clip.wav"))
    ap.add_argument("--models", nargs="*", default=None,
        help="subset of model names to run")
    ap.add_argument("--skip-stt", action="store_true",
        help="skip the speech-to-text stage")
    ap.add_argument("--stt-reps", type=int, default=3,
        help="how many real-time STT passes to average")
    args = ap.parse_args()

    cfg = load_config(args.config)
    lemonade_base = cfg["lemonade_base"]
    if lemonade_base:
        stt_bench.LEMONADE_BASE = lemonade_base
        if not wait_ready(f"{lemonade_base}/models", 5):
            print(f"WARNING: Lemonade not reachable at {lemonade_base}. "
                  "Start it (`lemonade serve`) and confirm the port.\n")

    narratives = json.loads(Path(args.narratives).read_text())
    system = coda_snapshot.build_system_prompt()
    hw = hardware_info()
    print(f"Config: {cfg['name']}   Hardware: {hw['chip']} {hw['ram_gb']}GB "
          f"{hw['bandwidth_gbps']}GB/s")
    print(f"Narratives: {len(narratives)}\n")

    report = {"config": cfg["name"], "hardware": hw,
              "timestamp": datetime.now().isoformat(timespec="seconds"),
              "n_narratives": len(narratives), "stt": None, "models": []}

    if not args.skip_stt:
        print(f"[speech-to-text] WhisperLiveKit {cfg['stt_backend']} small "
              f"x{args.stt_reps}...")
        stt = run_stt(args.audio, args.stt_reps, cfg["stt_backend"])
        if "error" in stt:
            print(f"  ERROR: {stt['error']}\n")
        else:
            print(f"  keep_up mean={stt['keep_up_mean']} median={stt['keep_up_median']} "
                  f"(<=1.0 real time)  peak={stt['peak_gb']}GB\n")
        report["stt"] = stt

    models = [m for m in cfg["models"] if not args.models or m["name"] in args.models]
    for m in models:
        print(f"[{m['name']}] ({m['backend']})...")
        try:
            r = bench_model(m, narratives, system, lemonade_base)
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
    tag = re.sub(r"[^A-Za-z0-9._+-]+", "_", hw["chip"] or cfg["name"])
    out = out_dir / f"{tag}_{report['timestamp'].replace(':', '')}.json"
    out.write_text(json.dumps(report, indent=2))

    print("=" * 68)
    stt = report["stt"]
    if stt and "error" not in stt:
        print(f"speech-to-text (WhisperLiveKit {stt.get('backend')} small, "
              f"{stt['reps']} reps): keep_up={stt.get('keep_up_mean')}  "
              f"peak={stt.get('peak_gb')}GB")
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
