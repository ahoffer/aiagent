#!/usr/bin/env bash
# Launches ollmcp MCP client for Ollama with configured MCP servers, usage: ./ollmcp.sh [extra-args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.env"

if ! command -v ollmcp &>/dev/null; then
    echo "Installing mcp-client-for-ollama..."
    pipx install mcp-client-for-ollama
    echo "Installed."
fi

# ollmcp has no CLI flag for system prompt. Write a default config with
# the current date and time so the model knows when it is.
OLLMCP_CONFIG_DIR="${HOME}/.config/ollmcp"
mkdir -p "$OLLMCP_CONFIG_DIR"
cat > "$OLLMCP_CONFIG_DIR/default.json" <<EOJSON
{
  "modelConfig": {
    "system_prompt": "${AGENT_SYSTEM_PROMPT} The current date and time is $(date -Iseconds)."
  }
}
EOJSON

exec ollmcp \
    -H "$OLLAMA_HOST" \
    -m "$AGENT_MODEL" \
    -j "$SCRIPT_DIR/mcp-servers.json" \
    "$@"
