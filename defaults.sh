#!/usr/bin/env bash
# Shared environment for agent launcher scripts.
# Ollama is cluster-internal only (ClusterIP). For local testing on the cluster
# node, use kubectl port-forward deploy/ollama 11434:11434 -n aiforge.
# Override any variable before sourcing or via environment.

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
OLLAMA_URL="${OLLAMA_URL:-$OLLAMA_HOST}"

# Base model pulled from Ollama registry
AGENT_BASE_MODEL="${AGENT_BASE_MODEL:-devstral:latest}"

# Default tuned alias created by the Ollama postStart hook with baked-in verbosity controls.
AGENT_MODEL="${AGENT_MODEL:-${AGENT_BASE_MODEL}-agent}"

# Response tuning parameters. Same defaults as the ConfigMap.
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.7}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-2048}"
AGENT_TOP_P="${AGENT_TOP_P:-0.9}"
AGENT_REPEAT_PENALTY="${AGENT_REPEAT_PENALTY:-1.2}"
AGENT_SYSTEM_PROMPT="${AGENT_SYSTEM_PROMPT:-Be concise and direct. Avoid filler phrases. When helping with code, ALWAYS search the web for latest documentation, API references, and code examples before answering. Do not rely on potentially outdated training data for libraries, frameworks, or technical specifications. Search first, then answer.}"

# Multi-agent system models
INTERPRETER_MODEL="${INTERPRETER_MODEL:-${AGENT_MODEL}}"
ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-${AGENT_MODEL}}"
RESEARCH_MODEL="${RESEARCH_MODEL:-${AGENT_MODEL}}"
SYNTHESIS_MODEL="${SYNTHESIS_MODEL:-${AGENT_MODEL}}"
CRITIC_MODEL="${CRITIC_MODEL:-${AGENT_MODEL}}"

# RAG configuration
QDRANT_URL="${QDRANT_URL:-http://bigfish:31333}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
SEARXNG_URL="${SEARXNG_URL:-http://bigfish:31080}"
AGENT_URL="${AGENT_URL:-http://bigfish:31400}"
