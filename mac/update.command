#!/bin/bash
# Update CODA to the newest release
set -uo pipefail
. "$(dirname "$0")/common.sh"
check_prereqs
load_env .env mac/defaults.env
if nc -z 127.0.0.1 "${CODA_APP__PORT:-8000}" 2>/dev/null; then
    fail "CODA is running. Quit it by closing its Terminal window, then \
update again."
fi
update manual
