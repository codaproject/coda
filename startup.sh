#!/bin/bash
trap 'kill 0' EXIT

export PYTHONPATH=$PYTHONPATH:src

python -m coda.inference.agent &

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
