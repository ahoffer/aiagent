#!/usr/bin/env bash
# Launches OpenHands AI agent against local Ollama and SearXNG, usage: ./openhands.sh [extra-args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../defaults.sh"

export LLM_API_KEY="${LLM_API_KEY:-ollama}"
export LLM_BASE_URL="${OLLAMA_HOST}"
export LLM_MODEL="${LLM_MODEL:-ollama/$OPENHANDS_MODEL}"
export SEARXNG_URL="${SEARXNG_URL:-http://bigfish:31080}"

if ! command -v openhands &>/dev/null; then
    echo "OpenHands CLI not found. Installing..."
    if command -v uv &>/dev/null; then
        uv tool install openhands --python 3.12
    else
        curl -fsSL https://install.openhands.dev/install.sh | sh
    fi
    echo "Installed."
fi

# Configure SearXNG MCP server for web search silently
openhands mcp add searxng --transport stdio \
    --env "SEARXNG_URL=$SEARXNG_URL" \
    uvx -- mcp-searxng >/dev/null 2>&1 || true

exec openhands --override-with-envs "$@"
