#!/usr/bin/env bash
# Run ASR and grounding benchmarks for a given Whisper model size.
# Usage: ./run_benchmark.sh [model_size] [language]
# Example: ./run_benchmark.sh base
#          ./run_benchmark.sh small fr

MODEL=${1:-tiny}
LANGUAGE=${2:-all}

echo "=== ASR Benchmark (whisper-${MODEL}, language: ${LANGUAGE}) ==="
python benchmark_asr.py --backends whisper --sizes "${MODEL}" --language "${LANGUAGE}" --task transcribe

echo ""
echo "=== Grounding Benchmark (whisper-${MODEL}) ==="
python benchmark_grounding.py --model "${MODEL}"
