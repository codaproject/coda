#!/bin/bash
# Run CODA with no network dependency: a local LLM for inference, a local
# translation model, and the local grounder. Every setting is overridable, so
# `CODA_INFERENCE__LLM__MODEL=gemma4:12b ./startup_offline.sh` swaps the model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Inference through Ollama rather than a hosted API. An instruct model is the
# right choice here: the Ollama adapter constrains decoding with format=schema,
# which degrades reasoning models.
export CODA_INFERENCE__LLM__PROVIDER="${CODA_INFERENCE__LLM__PROVIDER:-ollama}"
export CODA_INFERENCE__LLM__MODEL="${CODA_INFERENCE__LLM__MODEL:-qwen2.5:7b-instruct}"

# Translate transcripts locally instead of calling out to an LLM API. The
# local backend is not present in every build, and setting the variable for a
# build without it would leave translation quietly reaching for the network.
if grep -q "^translate:" config/settings.yaml 2>/dev/null; then
    export CODA_TRANSLATE__BACKEND="${CODA_TRANSLATE__BACKEND:-ctranslate2}"
    TRANSLATION_MODE="$CODA_TRANSLATE__BACKEND"
else
    TRANSLATION_MODE="unavailable in this build, translation may use the network"
fi

# Gilda grounds against a local database; the RAG grounder needs API embeddings.
export CODA_GROUNDER__TYPE="${CODA_GROUNDER__TYPE:-gilda}"

OLLAMA_URL="${CODA_LLM__OLLAMA__BASE_URL:-http://localhost:11434}"
if ! curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo "Ollama is not reachable at ${OLLAMA_URL}. Start it with 'ollama serve'." >&2
    exit 1
fi

if ! curl -s "${OLLAMA_URL}/api/tags" | grep -q "\"${CODA_INFERENCE__LLM__MODEL}\""; then
    echo "Model ${CODA_INFERENCE__LLM__MODEL} is not pulled. Run:" >&2
    echo "    ollama pull ${CODA_INFERENCE__LLM__MODEL}" >&2
    exit 1
fi

echo "Offline mode"
echo "  inference:   ${CODA_INFERENCE__LLM__PROVIDER} / ${CODA_INFERENCE__LLM__MODEL}"
echo "  translation: ${TRANSLATION_MODE}"
echo "  grounder:    ${CODA_GROUNDER__TYPE}"

exec ./startup.sh
