#!/bin/bash
# Update CODA to the newest release
set -uo pipefail
. "$(dirname "$0")/common.sh"
check_prereqs
if curl -sfk "$(cat "$URL_FILE" 2>/dev/null)/health" >/dev/null; then
    fail "CODA is running. Quit it by closing its Terminal window, then \
update again."
fi
update manual
