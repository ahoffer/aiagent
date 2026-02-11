# Local AI Agent Stack

Self-hosted AI development platform on Kubernetes with GPU-accelerated LLM inference, web search, vector storage, and a chat interface.

## Architecture

Four services run in the `ai-agent` namespace.

| Service | Role | Cluster DNS | NodePort |
|---------|------|-------------|----------|
| Ollama | Serves models with NVIDIA GPU passthrough | ollama:11434 | :31434 |
| SearXNG | Aggregates web search results | searxng:8080 | :31080 |
| Qdrant | Stores vector embeddings for RAG | qdrant:6333 | :31333 |
| Open WebUI | Browser chat interface connecting all three | open-webui:8080 | :31380 |

## Quick Start

Prerequisites: Kubernetes cluster with GPU support, NVIDIA device plugin, kubectl.

```bash
./install.sh
```

Verify with `./tests/test-stack.sh` (8 checks) and `./tests/test-services.sh` (19 checks), then open `http://localhost:31380` or launch any agent.

## Agent Frontends

| Frontend | Startup | Autonomy | Integrations | Description |
|----------|---------|----------|--------------|-------------|
| `aider.sh` | 13s | You approve each edit | File edit | Pair programming agent that proposes diffs. Edits code in your working directory. |
| `goose.sh` | 17s | Self-directed | None | Multi-step agent that chains tasks together. Reads .goosehints for project context. |
| `ollmcp.sh` | 4s | You drive the conversation | Web search, files, git, shell | Terminal chat with live tool calling across 5 MCP servers. |
| `opencode.sh` | 1s | You approve each edit | None | Lightweight web UI for quick code questions. Browser only, runs on port 31580. |
| `openhands.sh` | 10s | Runs unattended | Web search, file edit, shell | Sandboxed agent that plans, codes, tests, and iterates without intervention. |
| [Open WebUI](http://localhost:31380) | - | You drive the conversation | Web search, RAG, image gen | Browser chat for general questions and web research. Works like ChatGPT. |

Local models sometimes misinterpret intent. Asking a frontend to "review" code may cause the model to attempt edits instead of analysis, surfacing raw tool errors when the edit fails. Use explicit phrasing like "analyze this code, do not edit any files" to avoid this.

## MCP Tool Servers

Five servers in `mcp-servers.json` extend tool-capable frontends.

| Server | Capability |
|--------|-----------|
| searxng | Web search via SearXNG metasearch |
| filesystem | Read/write access to home directory |
| git | Repository inspection, history, branches |
| shell | Whitelisted commands: ls, grep, find, python3, node, make, kubectl |
| fetch | HTTP URL content retrieval |

## Configuration

All launcher scripts source `.env` for defaults. Override any variable at launch.

| Variable | Default | Description |
|----------|---------|-------------|
| AGENT_MODEL | qwen3:14b-agent | Default model, tuned alias with concise output |
| AIDER_MODEL | llama3.1:8b | Model for aider, llama works where qwen3 causes litellm parse errors |
| GOOSE_MODEL | qwen3:14b-agent | Model for goose |
| OLLMCP_MODEL | mistral-nemo:latest | Initial model for ollmcp, mistral is better at tool calling |
| OPENCODE_MODEL | qwen3:14b-16k | Model for opencode, uses 16k context variant |
| OPENHANDS_MODEL | mistral-nemo:latest | Model for openhands, mistral is more reliable at tool calling |

```bash
GOOSE_MODEL=qwen3:8b ./goose.sh           # use smaller model for goose
AIDER_MODEL=llama3.1:8b ./aider.sh        # try llama for aider
./switch-model.sh                          # interactive model picker
```

Cluster-side settings live in ConfigMaps in `agent.yaml`.

## Testing

`tests/test-stack.sh` validates health, inference, tool calling, and token speed (minimum 10 tok/s). `tests/test-services.sh` checks Qdrant CRUD, SearXNG search, Open WebUI connectivity, embeddings, and cross-service wiring. `tests/test-tool-calling.py` runs 12 prompts across single-tool, no-tool, and multi-tool categories. `tests/bench-ollama.sh` measures generation speed, prompt eval, time to first token, and tool call latency. `tests/bench-frontends.sh` compares wall-clock latency across all agent frontends.

```bash
./tests/test-stack.sh
./tests/test-services.sh
python3 tests/test-tool-calling.py
```

Override the target model or URL with environment variables, for example `MODEL=qwen3:8b ./tests/test-stack.sh`.

## Teardown

Remove the stack while keeping persistent data for future redeployment.

```bash
./uninstall.sh
```

Remove everything including downloaded models and vector data.

```bash
./uninstall.sh --purge
```
