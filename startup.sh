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

# First global IPv4 address, for the phone setup URL. Empty when none is found.
first_lan_ip() {
    local line cidr addr
    command -v ip >/dev/null 2>&1 || return 0
    while read -r line; do
        case "$line" in
            *" inet "*)
                cidr="${line##* inet }"
                addr="${cidr%% *}"
                addr="${addr%%/*}"
                case "$addr" in
                    ""|127.*|169.254.*)
                        continue
                        ;;
                    *)
                        printf '%s\n' "$addr"
                        return 0
                        ;;
                esac
                ;;
        esac
    done < <(ip -4 -o addr show scope global 2>/dev/null || true)
}

tls_enabled() {
    case "${CODA_APP__TLS,,}" in
        1|true|yes|on)
            return 0
            ;;
        *)
            return 1
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
export CODA_APP__TLS="${CODA_APP__TLS:-true}"
export CODA_APP__SETUP_PORT="${CODA_APP__SETUP_PORT:-8001}"
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
if tls_enabled; then
    until curl -sfk "https://${APP_HEALTH_HOST}:${CODA_APP__PORT}/health" > /dev/null 2>&1; do
        sleep 1
    done
    echo "CODA is running at https://localhost:${CODA_APP__PORT}"
    LAN_IP="$(first_lan_ip || true)"
    if [ -n "${LAN_IP}" ]; then
        echo "Phone setup: http://${LAN_IP}:${CODA_APP__SETUP_PORT}"
    else
        echo "Phone setup: http://<this-computer>:${CODA_APP__SETUP_PORT}"
    fi
else
    until curl -sf "http://${APP_HEALTH_HOST}:${CODA_APP__PORT}/health" > /dev/null 2>&1; do
        sleep 1
    done
    echo "CODA is running at http://localhost:${CODA_APP__PORT}"
fi

wait
