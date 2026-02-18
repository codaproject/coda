#!/usr/bin/env bash
# Run ASR and grounding benchmarks for a given Whisper model size.
# Usage: ./run_benchmark.sh [model_size]
# Example: ./run_benchmark.sh base

MODEL=${1:-tiny}

echo "=== ASR Benchmark (whisper-${MODEL}) ==="
python benchmark_asr.py --model_id "openai/whisper-${MODEL}" --task transcribe

echo ""
echo "=== Grounding Benchmark (whisper-${MODEL}) ==="
python benchmark_grounding.py --model "${MODEL}"
