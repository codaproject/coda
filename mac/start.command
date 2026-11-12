#!/bin/bash
# Start CODA on an Apple Silicon Mac, offering a newer release first
set -euo pipefail
. "$(dirname "$0")/common.sh"
check_prereqs

head="$(git rev-parse HEAD)"
update startup || true
# Rerun the updated version of this script
[ "$(git rev-parse HEAD)" = "$head" ] || exec "$PWD/mac/start.command" "$@"

# Keep a log of everything shown in the Terminal window
mkdir -p "$HOME/Library/Logs/CODA"
exec > >(tee -a "$HOME/Library/Logs/CODA/coda.log") 2>&1

sync_env || fail "Installing CODA's Python packages failed, see the \
Terminal window for details."
mkdir -p "$DATA"
[ -f "$CODA_SETTINGS_FILE" ] || cp mac/settings.yaml "$CODA_SETTINGS_FILE"
setup_cases

# A site with several use cases asks which one to start
usecases="$(site_usecases "$(laptop_site)")"
if [ "$(printf '%s' "$usecases" | grep -c .)" -gt 1 ]; then
    last="$(cat "$DATA/last_usecase" 2>/dev/null || true)"
    printf '%s\n' "$usecases" | grep -qx "$last" || last="${usecases%%$'\n'*}"
    usecase="$(choose_usecase "$last" $usecases)" || exit 0
    echo "$usecase" > "$DATA/last_usecase"
    export CODA_USECASE="$usecase"
elif [ -n "$usecases" ]; then
    export CODA_USECASE="$usecases"
fi

config="$(python mac/check_config.py 2>&1)" \
    || fail "CODA's settings have a problem: $config"
read -r port tls setup_port inference_port <<< "$config"
start_ollama || fail "Ollama did not start."
python mac/prepare.py ollama || fail "Downloading CODA's models failed, \
see the Terminal window for details."
build_grounding || fail "Preparing grounding data failed, see the Terminal \
window for details."

trap 'rm -f "$URL_FILE"; kill $(jobs -p) 2>/dev/null' EXIT
# Keep the Mac from sleeping while CODA runs
caffeinate -i -w $$ &
# Wait for a service's health URL, failing if the service exits
wait_for() {
    until curl -sfk "$2" >/dev/null; do
        kill -0 "$1" 2>/dev/null || fail "CODA stopped while starting, see \
the Terminal window for details."
        sleep 1
    done
}
python -m coda.inference.agent &
wait_for $! "http://127.0.0.1:$inference_port/health"
python -m coda.app &

# Once the app answers, trust its certificate authority and open it
scheme=http
[ "$tls" = True ] && scheme=https
wait_for $! "$scheme://127.0.0.1:$port/health"
ca=certs/ca.pem
if [ $scheme = https ] && ! security verify-cert -c "$ca" >/dev/null 2>&1; then
    security add-trusted-cert -r trustRoot \
        -k "$HOME/Library/Keychains/login.keychain-db" "$ca" || true
fi
echo "$scheme://localhost:$port" > "$URL_FILE"
open "$scheme://localhost:$port"
echo "CODA is running at $scheme://localhost:$port"
ip="$(ipconfig getifaddr en0 || ipconfig getifaddr en1 || true)"
[ $scheme = http ] || [ -z "$ip" ] \
    || echo "Phone and tablet setup: http://$ip:$setup_port"
echo "Close this window to stop CODA."
wait
