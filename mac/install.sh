#!/bin/bash
# Set up CODA on a new Apple Silicon Mac for a site, downloading everything
# its profiles need. From Terminal:
#   curl -fsSL https://raw.githubusercontent.com/codaproject/coda/main/mac/install.sh | bash -s <site>
# A HF_TOKEN in the environment is saved for models that need a Hugging Face
# login.
set -euo pipefail
site="${1:?Give a site name from config/sites/}"
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
ls config/sites | grep -qE "^$site(-.*)?\.yaml$" \
    || fail "No site profile $site in config/sites/"
mkdir -p "$DATA"
[ -f "$CODA_SETTINGS_FILE" ] \
    || sed "s/^site: .*/site: $site/" mac/settings.yaml > "$CODA_SETTINGS_FILE"
ln -sf "$PWD/mac/CODA.app" "$PWD/mac/Update CODA.app" \
    "$PWD/mac/CODA Settings.app" "$HOME/Desktop/"
echo "Installing CODA's packages and models, which takes a while..."
sync_env && prepare || fail "Installing CODA failed, see above for details."
echo "CODA is installed, starting it for the first time."
open mac/CODA.app
