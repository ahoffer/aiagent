"""Shared configuration read once from environment variables.

All gateway modules import from here so model names and tunables
live in a single place. In k8s, values flow from ollama-config and
gateway-config ConfigMaps. Missing required vars fail fast with a
clear message rather than falling back to stale defaults.
"""

import os
import sys


def _require(name: str) -> str:
    """Return an env var or exit with a diagnostic message."""
    val = os.getenv(name)
    if not val:
        print(
            f"FATAL: required environment variable {name} is not set. "
            f"In k8s this comes from a ConfigMap. For local dev, source config.env first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return val


OLLAMA_URL = _require("OLLAMA_URL").rstrip("/")
AGENT_MODEL = _require("AGENT_MODEL")
AGENT_NUM_CTX = int(_require("AGENT_NUM_CTX"))
EMBEDDING_MODEL = _require("EMBEDDING_MODEL")
LOG_LANGGRAPH_OUTPUT = os.getenv("LOG_LANGGRAPH_OUTPUT", "true").lower() in ("1", "true")
