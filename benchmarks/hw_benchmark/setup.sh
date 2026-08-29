#!/usr/bin/env bash
# One-command setup for the CODA hardware benchmark on Apple Silicon (mac config).
# Installs the benchmark's dependencies into an isolated venv and pulls the two
# Ollama models. Self-contained: no coda install or git needed. The MLX model
# downloads on first run. For the Ryzen AI config, use setup.ps1 instead.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${CODA_INBENCH_VENV:-$HOME/.virtualenvs/coda-inbench}"

echo "== CODA hardware benchmark setup (mac config) =="

if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not found. Install it once with:  brew install ollama"
    echo "(or download from ollama.com), then re-run this script."
    exit 1
fi
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama..."
    open -a Ollama 2>/dev/null || (ollama serve >/dev/null 2>&1 &)
    for _ in $(seq 1 20); do
        curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Could not start Ollama automatically. Start it (open -a Ollama) and re-run."
        exit 1
    fi
fi

echo "Pulling Ollama models..."
ollama pull qwen2.5:7b-instruct
ollama pull gpt-oss:20b

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV"
    python3 -m venv "$VENV"
fi
echo "Installing dependencies..."
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet ollama openai mlx-openai-server \
    whisperlivekit mlx-whisper numpy matplotlib

echo
echo "Setup complete. Run the benchmark with:"
echo "  $VENV/bin/python $HERE/run_bench.py"
echo "(config auto-detects as 'mac'; override with --config)"
echo
echo "First run downloads the MLX Qwen3-30B model (~17 GB)."
echo "After collecting results, chart them with:"
echo "  $VENV/bin/python $HERE/plot_results.py"
