#!/usr/bin/env bash
# Launches Goose AI agent session against local Ollama, usage: ./goose.sh [extra-args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.env"
export OLLAMA_HOST
export GOOSE_PROVIDER="${GOOSE_PROVIDER:-ollama}"
export GOOSE_MODEL
export GOOSE_CLI_THEME="${GOOSE_CLI_THEME:-light}"

if ! command -v goose &>/dev/null; then
    echo "Goose not found. Installing..."
    curl -fsSL https://github.com/block/goose/releases/latest/download/download_cli.sh | CONFIGURE=false bash
    echo "Installed."
fi

# Support both interactive session and non-interactive run modes.
# Usage: ./goose.sh              Interactive session
#        ./goose.sh run -i -     Run from stdin
#        ./goose.sh run -t "msg" Run with text
if [[ "${1:-}" == "run" ]]; then
    shift
    exec goose run --provider "$GOOSE_PROVIDER" --model "$GOOSE_MODEL" "$@"
fi

exec goose session "$@"
