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
    # `APP_PORT=8100 INFERENCE_PORT=6123 ./startup.sh`.
    load_env_file ".env"
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SCRIPT_DIR/src"
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8000}"
export INFERENCE_HOST="${INFERENCE_HOST:-0.0.0.0}"
export INFERENCE_PORT="${INFERENCE_PORT:-5123}"
export INFERENCE_URL="${INFERENCE_URL:-${CODA_INFERENCE_URL:-http://127.0.0.1:${INFERENCE_PORT}}}"

APP_HEALTH_HOST="$(health_host_for "$APP_HOST")"
INFERENCE_HEALTH_HOST="$(health_host_for "$INFERENCE_HOST")"

python -m coda.inference.agent &

echo "Waiting for inference agent..."
until curl -sf "http://${INFERENCE_HEALTH_HOST}:${INFERENCE_PORT}/health" > /dev/null 2>&1; do
    sleep 1
done
echo "Inference agent ready."

python -m coda.app &

echo "Waiting for web application..."
until curl -sf "http://${APP_HEALTH_HOST}:${APP_PORT}/health" > /dev/null 2>&1; do
    sleep 1
done
echo "CODA is running at http://localhost:${APP_PORT}"

wait
