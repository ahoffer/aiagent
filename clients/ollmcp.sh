#!/usr/bin/env bash
# Launches ollmcp MCP client for Ollama with configured MCP servers, usage: ./ollmcp.sh [extra-args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_DIR/defaults.sh"

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
    "system_prompt": "${AGENT_SYSTEM_PROMPT} The current date and time is $(date -Iseconds). You have web search via the searxng tool. USE IT. For current events, sports, news, or ANY technical question about code, libraries, APIs, or documentation, search the web FIRST before answering. Never guess. Never use outdated knowledge. Search, then answer."
  }
}
EOJSON

exec ollmcp \
    -H "$OLLAMA_HOST" \
    -m "$OLLMCP_MODEL" \
    -j "$PROJECT_DIR/mcp-servers.json" \
    "$@"
