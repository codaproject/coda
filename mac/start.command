#!/bin/bash
# Start CODA on an Apple Silicon Mac, offering a newer release first
set -euo pipefail
. "$(dirname "$0")/common.sh"
check_prereqs

head="$(git rev-parse HEAD)"
update startup || true
# Rerun the updated version of this script
[ "$(git rev-parse HEAD)" = "$head" ] || exec "$PWD/mac/start.command" "$@"

sync_env || fail "Installing CODA's Python packages failed, see the \
Terminal window for details."
load_env .env mac/defaults.env

if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
    open -a Ollama
    for _ in $(seq 60); do
        curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
        sleep 1
    done
fi
if [ "${CODA_INFERENCE__LLM__PROVIDER:-}" = ollama ] && \
        ! ollama show "$CODA_INFERENCE__LLM__MODEL" >/dev/null 2>&1; then
    echo "Downloading $CODA_INFERENCE__LLM__MODEL..."
    ollama pull "$CODA_INFERENCE__LLM__MODEL"
fi

export PYTHONPATH="$PWD/src"
if [ -z "${CODA_GROUNDER__GILDA__SQLITE_DB:-}" ]; then
    export CODA_GROUNDER__GILDA__SQLITE_DB="$GILDA_DB"
    if [ ! -f "$GILDA_DB" ]; then
        echo "Preparing grounding data (first start only)..."
        mkdir -p "$DATA"
        python -m coda.grounding.build_grounding_db "$GILDA_DB.tmp"
        mv "$GILDA_DB.tmp" "$GILDA_DB"
    fi
fi

# Once the app answers, trust its certificate authority and open it
port="${CODA_APP__PORT:-8000}"
scheme=https
case "$(printf '%s' "${CODA_APP__TLS:-true}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) ;;
    *) scheme=http ;;
esac
(
    until curl -sfk "$scheme://127.0.0.1:$port/health" >/dev/null; do
        sleep 2
    done
    ca="${CODA_APP__TLS_DIR:-certs}/ca.pem"
    if [ $scheme = https ] && ! security verify-cert -c "$ca" >/dev/null 2>&1
    then
        security add-trusted-cert -r trustRoot \
            -k "$HOME/Library/Keychains/login.keychain-db" "$ca" || true
    fi
    open "$scheme://localhost:$port"
) &
exec ./startup.sh
