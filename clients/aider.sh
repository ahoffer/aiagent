#!/usr/bin/env bash
# Launches Aider AI pair-programming assistant against local Ollama, usage: ./aider.sh [extra-args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.env"
export OLLAMA_API_BASE="${OLLAMA_HOST}"

if ! command -v aider &>/dev/null; then
    echo "Aider not found. Installing via pipx..."
    pipx install aider-chat
    echo "Installed."
fi

# Use AIDER_MODEL. Defaults to base model since -agent thinking tokens break litellm's parser.
exec aider --model "ollama/$AIDER_MODEL" "$@"
