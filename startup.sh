#!/bin/bash
set -euo pipefail

cleanup() {
    local pid
    for pid in $(jobs -pr); do
        kill "$pid" 2>/dev/null || true
    done
}

trap cleanup EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

load_env_file() {
    local env_file="$1"
    local line key value
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        case "$line" in
            ""|\#*)
                continue
                ;;
            export\ *)
                line="${line#export }"
                ;;
        esac

        key="${line%%=*}"
        value="${line#*=}"
        if [ "$key" = "$line" ]; then
            continue
        fi

        if [ -z "${!key+x}" ]; then
            export "$key=$value"
        fi
    done < "$env_file"
}

health_host_for() {
    case "$1" in
        0.0.0.0)
            printf '127.0.0.1'
            ;;
        *)
            printf '%s' "$1"
            ;;
    esac
}

if [ -f ".env" ]; then
    # Load defaults from .env without overriding variables already set in the
    # caller's shell. This preserves explicit one-off overrides like
    # `CODA_APP__PORT=8100 CODA_INFERENCE__PORT=6123 ./startup.sh`.
    load_env_file ".env"
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SCRIPT_DIR/src"
export CODA_APP__HOST="${CODA_APP__HOST:-0.0.0.0}"
export CODA_APP__PORT="${CODA_APP__PORT:-8000}"
export CODA_INFERENCE__HOST="${CODA_INFERENCE__HOST:-0.0.0.0}"
export CODA_INFERENCE__PORT="${CODA_INFERENCE__PORT:-5123}"
export CODA_INFERENCE__URL="${CODA_INFERENCE__URL:-http://127.0.0.1:${CODA_INFERENCE__PORT}}"

APP_HEALTH_HOST="$(health_host_for "$CODA_APP__HOST")"
INFERENCE_HEALTH_HOST="$(health_host_for "$CODA_INFERENCE__HOST")"

python -m coda.inference.agent &

echo "Waiting for inference agent..."
until curl -sf "http://${INFERENCE_HEALTH_HOST}:${CODA_INFERENCE__PORT}/health" > /dev/null 2>&1; do
    sleep 1
done
echo "Inference agent ready."

python -m coda.app &

echo "Waiting for web application..."
until curl -sf "http://${APP_HEALTH_HOST}:${CODA_APP__PORT}/health" > /dev/null 2>&1; do
    sleep 1
done
echo "CODA is running at http://localhost:${CODA_APP__PORT}"

wait
