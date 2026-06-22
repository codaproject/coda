#!/usr/bin/env bash
# Run ASR and grounding benchmarks for a given Whisper model size.
#
# Usage:   ./run_benchmark.sh [model_size] [language]
# Example: ./run_benchmark.sh base
#          ./run_benchmark.sh small fr
#          ./run_benchmark.sh large-v3 all
#
# model_size : tiny | base | small | medium | large-v3-turbo | large-v3
# language   : en | nl | fr | de | es | all  (default: all)

MODEL=${1:-tiny}
LANGUAGE=${2:-all}

echo "=== ASR Benchmark (whisper-${MODEL}, language: ${LANGUAGE}) ==="
python benchmark_asr.py \
    --model_id "openai/whisper-${MODEL}" \
    --language "${LANGUAGE}" \
    --task transcribe

echo ""
echo "=== Grounding Benchmark (whisper-${MODEL}) ==="
python benchmark_grounding.py --model "${MODEL}"