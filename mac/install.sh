#!/bin/bash
# Set up CODA on a new Apple Silicon Mac for a site profile, from Terminal:
#   curl -fsSL https://raw.githubusercontent.com/codaproject/coda/main/mac/install.sh | bash -s <site>
set -euo pipefail
site="${1:?Give a site profile name from config/sites/, e.g. mac}"
dir="$HOME/coda"
export GIT_LFS_SKIP_SMUDGE=1

if ! xcode-select -p >/dev/null 2>&1; then
    xcode-select --install
    echo "Finish installing the command line tools, then run this again."
    exit 1
fi
[ -x "$HOME/.local/bin/uv" ] || command -v uv >/dev/null \
    || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -d /Applications/Ollama.app ]; then
    echo "Downloading Ollama..."
    tmp="$(mktemp -d)"
    curl -fL -o "$tmp/Ollama.zip" https://ollama.com/download/Ollama-darwin.zip
    ditto -x -k "$tmp/Ollama.zip" /Applications
fi
[ -d "$dir" ] || git clone https://github.com/codaproject/coda.git "$dir"

. "$dir/mac/common.sh"
git checkout --quiet "$(latest_release)"
[ -f "config/sites/$site.yaml" ] || fail "No site profile config/sites/$site.yaml"
mkdir -p "$DATA"
[ -f "$CODA_SETTINGS_FILE" ] \
    || sed "s/^site: .*/site: $site/" mac/settings.yaml > "$CODA_SETTINGS_FILE"
ln -sf "$PWD/mac/CODA.app" "$PWD/mac/Update CODA.app" \
    "$PWD/mac/CODA Settings.app" "$HOME/Desktop/"
echo "Starting CODA for the first time, which downloads its packages and models."
open mac/CODA.app
