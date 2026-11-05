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

# A site with several profiles asks which use case to start
site="$(laptop_site)"
profiles="$(site_profiles "$site")"
if [ "$(printf '%s' "$profiles" | grep -c .)" -gt 1 ]; then
    uses="$(printf '%s\n' "$profiles" | sed "s/^$site-//")"
    last="$(cat "$DATA/last_use" 2>/dev/null || true)"
    printf '%s\n' "$uses" | grep -qx "$last" || last="${uses%%$'\n'*}"
    use="$(osascript -e 'on run argv' \
        -e 'set choice to choose from list (rest of argv) with title "CODA" with prompt "Start CODA for:" default items {item 1 of argv}' \
        -e 'if choice is false then error number -128' \
        -e 'return item 1 of choice' -e 'end run' "$last" $uses)" \
        || exit 0
    echo "$use" > "$DATA/last_use"
    export CODA_SITE="$site-$use"
elif [ -n "$profiles" ]; then
    export CODA_SITE="$profiles"
fi

config="$(python mac/check_config.py 2>&1)" \
    || fail "CODA's settings have a problem: $config"
read -r port tls setup_port inference_port <<< "$config"
start_ollama || fail "Ollama did not start."
python mac/prepare.py ollama || fail "Downloading CODA's models failed, \
see the Terminal window for details."
build_grounding || fail "Preparing grounding data failed, see the Terminal \
window for details."

trap 'kill $(jobs -p) 2>/dev/null' EXIT
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
open "$scheme://localhost:$port"
echo "CODA is running at $scheme://localhost:$port"
ip="$(ipconfig getifaddr en0 || ipconfig getifaddr en1 || true)"
[ $scheme = http ] || [ -z "$ip" ] \
    || echo "Phone and tablet setup: http://$ip:$setup_port"
echo "Close this window to stop CODA."
wait
