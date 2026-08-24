# CODA Ryzen AI benchmark

Ryzen/Windows counterpart of `mac_inference_bench`, built so results are directly
comparable to the Mac runs. It benchmarks the same two CODA stages, with the same
CHAMPS prompt, `COD_OUTPUT_SCHEMA`, schema-constrained decoding, the same 10
narratives, and the same audio clip:

1. **Speech-to-text** — WhisperLiveKit (`faster-whisper`, `small`, CPU) fed the
   clip as real-time PCM, the same streaming path CODA uses. Reports `keep_up`
   (`<= 1.0` means real time).
2. **Inference** — cause-of-death across the models, served by **Lemonade** over
   its OpenAI-compatible API (`response_format` json_schema). Lemonade runs ONNX
   models on the NPU+iGPU hybrid and GGUF models on llama.cpp (Vulkan/ROCm).

The comparability anchors (`coda_snapshot.py`, `champs/`, `narratives.json`,
`assets/clip.wav`) are copied byte-for-byte from `mac_inference_bench`. Only the
hardware probe, STT backend, and model serving differ.

## What differs from the Macs (report these caveats)
- **Quantization**: Macs run MLX 4-bit; Lemonade runs ONNX-hybrid or GGUF. The
  exact `id` per model is recorded in the report so a reader knows the difference.
  `valid_rate` is the guard for whether constrained decoding held.
- **STT weights**: `faster-whisper` small vs Apple's `mlx-whisper` small are
  different weights, so `keep_up` compares throughput, not word error rate.
- **Bandwidth axis**: `hwinfo.py` hardcodes ~256 GB/s for the Ryzen AI Max+
  (256-bit LPDDR5X-8000), matching how the Mac table looks up per-chip bandwidth,
  so the point lands on the shared `bandwidth_vs_latency` plot.

## Setup

Lemonade must already be installed and the models pulled. Then:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

This makes a venv at `%USERPROFILE%\.virtualenvs\coda-inbench` and installs the
Python deps. The whisper-small weights download on the first STT run.

## Confirm model ids, then run

Lemonade catalog names vary by build, so check yours and edit `MODELS` at the top
of `run_bench.py` to match:

```powershell
lemonade-server list
lemonade-server serve   # if not already running (port 13305)
```

Then:

```powershell
%USERPROFILE%\.virtualenvs\coda-inbench\Scripts\python.exe run_bench.py
```

It runs the STT stage (several real-time reps, averaged), then each model over the
10 narratives (warm call first, then timed), and reports:
- **STT**: mean/median `keep_up` and peak memory
- per model: **valid** (fraction returning schema-valid JSON) and **mean / median
  / stdev / min / max** warm latency

The report is written to `results/<chip>_<time>.json`. **Send that JSON back.**

## Options
- `--models Qwen2.5-7B-Instruct gpt-oss-20b` — run a subset.
- `--narratives path.json` — your own narratives (a JSON list of strings).
- `--audio path.wav` — your own 16 kHz mono clip for the STT stage.
- `--stt-reps N` — real-time STT passes to average (default 3).
- `--skip-stt` — inference only.

## Files
- `run_bench.py` — the benchmark (STT + Lemonade inference across the models)
- `coda_snapshot.py` — vendored CHAMPS prompt, schema, and request shapes (shared)
- `champs/` — CHAMPS system prompt, schema guidance, allowed causes (shared)
- `stt_bench.py` — speech-to-text stage (faster-whisper; also runnable standalone)
- `hwinfo.py` — Windows/AMD hardware probe, tags every report
- `narratives.json` — verbal-autopsy inputs (shared)
- `assets/clip.wav` — standard audio clip for the STT stage (shared)
- `setup.ps1` — venv + deps installer
