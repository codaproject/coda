#!/bin/bash
trap 'kill 0' EXIT

export PYTHONPATH=$PYTHONPATH:src

python -m coda.inference.agent --provider ollama --model llama3.2 &
# Alternatives:
#   python -m coda.inference.agent --provider ollama --model gpt-oss:20b  (13 GB, pull first)
#   python -m coda.inference.agent --provider openai --model gpt-5.4-mini

echo "Waiting for inference agent..."
until curl -sf http://localhost:5123/health > /dev/null 2>&1; do
    sleep 1
done
echo "Inference agent ready."

python -m coda.app &

echo "Waiting for web application..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    sleep 1
done
echo "CODA is running at http://localhost:8000"

wait
