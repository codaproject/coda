#!/bin/bash
# CODA launcher for macOS. Double-click in Finder to start the app.
#
# Brings up Docker Desktop and native Ollama if needed, starts the native
# whisperlivekit MLX transcription sidecar, launches the containerized app with
# the remote-transcription backend, and opens it in the browser. Closing this
# window (or Ctrl-C) stops the app and the sidecar.
#
# One-time setup on each Mac:
#   - Docker Desktop and the Ollama app installed
#   - pipx install whisperlivekit && pipx inject whisperlivekit mlx-whisper
set -euo pipefail

WLK_PORT="${WLK_PORT:-8765}"
WLK_BACKEND="${WLK_BACKEND:-mlx-whisper}"
WLK_MODEL="${WLK_MODEL:-small}"
APP_PORT="${APP_PORT:-8000}"

cd "$(dirname "$0")"

log() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
die() {
    printf "\n\033[1;31mError: %s\033[0m\n" "$1"
    echo "Press any key to close."
    read -r -n1
    exit 1
}

# Docker Desktop must be running for compose
if ! docker info >/dev/null 2>&1; then
    log "Starting Docker Desktop..."
    open -a Docker || die "Docker Desktop is not installed (get it from docker.com)."
    for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
    docker info >/dev/null 2>&1 || die "Docker did not become ready in time."
fi

# Native Ollama for inference
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    log "Starting Ollama..."
    open -a Ollama 2>/dev/null || true
    for _ in $(seq 1 30); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
fi
curl -s http://localhost:11434/api/tags >/dev/null 2>&1 || \
    echo "Warning: Ollama is not reachable, inference may fail."

# Native transcription sidecar on Apple GPU (MLX/Metal)
command -v whisperlivekit-server >/dev/null 2>&1 || \
    die "whisperlivekit-server not found. Install with: pipx install whisperlivekit && pipx inject whisperlivekit mlx-whisper"
if ! pgrep -f "whisperlivekit-server.*--port $WLK_PORT" >/dev/null; then
    log "Starting transcription sidecar ($WLK_BACKEND) on port $WLK_PORT..."
    whisperlivekit-server --backend "$WLK_BACKEND" --model "$WLK_MODEL" \
        --pcm-input --host 127.0.0.1 --port "$WLK_PORT" \
        >/tmp/coda-whisperlivekit.log 2>&1 &
fi

cleanup() {
    log "Shutting down CODA..."
    docker compose down 2>/dev/null || true
    pkill -f "whisperlivekit-server.*--port $WLK_PORT" 2>/dev/null || true
}
trap cleanup EXIT

# Containerized app, using the native sidecar for transcription
log "Starting CODA app..."
export CODA_DIALOGUE__TRANSCRIBER_BACKEND=whisper-livekit-remote
docker compose up -d

for _ in $(seq 1 60); do curl -s "http://localhost:$APP_PORT/health" >/dev/null 2>&1 && break; sleep 1; done
log "Opening http://localhost:$APP_PORT"
open "http://localhost:$APP_PORT"

echo
echo "CODA is running. Close this window or press Ctrl-C to stop."
docker compose logs -f
