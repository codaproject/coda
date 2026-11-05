# Helpers shared by the Mac scripts, sourced from the repo root
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$PWD/.venv-mac/bin:$HOME/.local/bin:/opt/homebrew/bin:\
/usr/local/bin:/Applications/Ollama.app/Contents/Resources:$PATH"
VENV=.venv-mac
DATA="$HOME/Library/Application Support/CODA"
GILDA_DB="$DATA/grounding_terms.db"
export CODA_SETTINGS_FILE="$DATA/settings.yaml"

# Show a message in a dialog
notify() {
    osascript -e 'on run argv' \
        -e 'display dialog (item 1 of argv) with title "CODA" buttons {"OK"} default button "OK"' \
        -e 'end run' "$1" >/dev/null 2>&1 || true
}

# Ask a yes/no question in a dialog, failing on "Not now"
ask() {
    osascript -e 'on run argv' \
        -e 'display dialog (item 1 of argv) with title "CODA" buttons {"Not now", "Update"} default button "Update" cancel button "Not now"' \
        -e 'end run' "$1" >/dev/null 2>&1
}

fail() {
    echo "$1" >&2
    notify "$1"
    exit 1
}

check_prereqs() {
    [ "$(uname -sm)" = "Darwin arm64" ] \
        || fail "CODA needs a Mac with Apple Silicon."
    command -v uv >/dev/null || fail "uv is not installed, see mac/README.md."
    command -v ollama >/dev/null \
        || fail "Ollama is not installed, see mac/README.md."
}

# Install the locked Python environment if the lock changed
sync_env() {
    if [ ! -x "$VENV/bin/python" ]; then
        uv venv --managed-python --python 3.12 "$VENV" || return
    fi
    cmp -s mac/requirements.lock "$VENV/requirements.lock" && return
    uv pip sync --python "$VENV/bin/python" mac/requirements.lock || return
    # Stopword list Gilda needs
    "$VENV/bin/python" -m nltk.downloader -q stopwords || return
    cp mac/requirements.lock "$VENV/requirements.lock"
    rm -f "$GILDA_DB"
}

latest_release() {
    git tag -l '[0-9]*' --sort=-v:refname | grep -v -- - | head -1
}

current_release() {
    git describe --tags --exact-match --match '[0-9]*' HEAD 2>/dev/null
}

# Release tag, else __version__, with the commit, e.g. 1.0.0 (ab12cd)
version_label() {
    local version
    version="$(current_release)" || version="$(sed -n \
        "s/^__version__ = '\(.*\)'$/\1/p" src/coda/__init__.py)"
    printf '%s (%s)' "$version" "$(git rev-parse --short=6 HEAD)"
}

# Offer to switch to the newest release, rolling back if that fails. With
# "startup", stay quiet unless this checkout is on an older release.
update() {
    local mode="$1" current latest previous notes
    if [ "$mode" = startup ] && [ -z "$(current_release)" ]; then
        return 0
    fi
    current="$(version_label)"
    echo "Checking for CODA updates..."
    if ! GIT_TERMINAL_PROMPT=0 git -c http.lowSpeedLimit=1000 \
            -c http.lowSpeedTime=10 fetch --quiet --tags --force origin 2>/dev/null; then
        [ "$mode" = startup ] || notify "Could not reach the CODA update \
server. Check the internet connection and try again."
        return 0
    fi
    latest="$(latest_release)"
    if [ -z "$latest" ] || \
            [ "$(git rev-parse HEAD)" = "$(git rev-parse "$latest^{commit}")" ]
    then
        [ "$mode" = startup ] || notify "CODA $current is up to date."
        return 0
    fi
    # Notes of an annotated tag, empty for a lightweight one
    notes="$(git tag -l --format='%(if)%(*objectname)%(then)%(contents)%(end)' \
        "$latest")"
    [ -z "$notes" ] || notes="$notes

"
    ask "CODA $latest is available. This laptop has $current.

${notes}Update now?" || return 0
    previous="$(git rev-parse HEAD)"
    echo "Updating CODA to $latest..."
    # The new release's own sync_env installs its environment
    if GIT_LFS_SKIP_SMUDGE=1 git checkout --quiet --force "$latest" \
            && (. mac/common.sh && sync_env); then
        notify "CODA is now at $latest."
        return 0
    fi
    echo "Update failed, going back to $current..." >&2
    GIT_LFS_SKIP_SMUDGE=1 git checkout --quiet --force "$previous" \
        && (. mac/common.sh && sync_env)
    notify "The update to $latest failed, CODA stays at $current. \
The Terminal window shows the details."
    return 1
}
