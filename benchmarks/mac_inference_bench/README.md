# CODA Mac benchmark

Benchmarks both stages of CODA's pipeline on a Mac laptop, the way CODA runs
them:

1. **Speech-to-text** — WhisperLiveKit (mlx-whisper, small) fed the clip as
   real-time PCM, the same path CODA uses. Reports `keep_up` (`<= 1.0` means the
   machine transcribes in real time).
2. **Inference** — cause-of-death across three models using CODA's inference
   request: the CHAMPS system prompt, `COD_OUTPUT_SCHEMA`, and schema-constrained
   decoding. Each model runs on the backend that works for it:

| Model | Backend |
|---|---|
| qwen2.5-7b-instruct | Ollama (`format=schema`) |
| gpt-oss-20b | Ollama (`format=schema`) |
| Qwen3-30B-A3B-Instruct-2507-4bit | mlx-openai-server (`response_format` json_schema) |

This folder is **self-contained** (the CHAMPS prompt, schema, and request shapes
are vendored in `coda_snapshot.py` + `champs/`), so it runs without installing
CODA or using git, just unzip and go.

## Setup (one command)

```bash
bash setup.sh
```

Requires an Apple Silicon Mac and Ollama installed (`brew install ollama`, or
from ollama.com). `setup.sh` starts Ollama if needed, pulls the two Ollama
models, and installs the Python deps into `~/.virtualenvs/coda-inbench`. First
benchmark run downloads the MLX Qwen3-30B (~17 GB) and the mlx-whisper model, so
allow ~20 GB free.

## Run

```bash
~/.virtualenvs/coda-inbench/bin/python run_bench.py
```

Runs the STT stage (several real-time reps, averaged), then each model over the
10 verbal-autopsy narratives in `narratives.json` (warm call first, then timed),
and reports:
- **STT**: mean/median `keep_up` and peak memory
- per model: **valid** (fraction returning schema-valid JSON) and **mean /
  median / stdev / min / max** warm latency across the 10 narratives

The report is written to `results/<chip>_<time>.json`. **Send that JSON file
back.** The full run takes ~10-15 min.

## Options

- `--models qwen2.5-7b-instruct gpt-oss-20b` — run a subset of inference models.
- `--narratives path.json` — your own narratives (a JSON list of strings).
- `--audio path.wav` — your own 16 kHz mono clip for the STT stage.
- `--stt-reps N` — real-time STT passes to average (default 3).
- `--skip-stt` — inference only.

## Distributing (make the zip)

From the repo, zip the folder without caches or prior results:

```bash
cd benchmarks
zip -r coda_mac_benchmark.zip mac_inference_bench \
    -x '*/results/*' '*/__pycache__/*' '*.pyc'
```

Recipients unzip it anywhere, `cd` in, and run the two commands above. No git or
CODA install needed.

## Files
- `run_bench.py` — the benchmark (STT + inference across the three models)
- `coda_snapshot.py` — vendored CHAMPS prompt, schema, and request shapes
- `champs/` — CHAMPS system prompt, schema guidance, allowed causes
- `stt_bench.py` — speech-to-text stage (also runnable standalone)
- `hwinfo.py` — hardware probe, tags every report
- `narratives.json` — verbal-autopsy inputs
- `assets/clip.wav` — standard audio clip for the STT stage
