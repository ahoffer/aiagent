#!/usr/bin/env python3
"""AiForge MCP server providing web_search, qdrant_search, and qdrant_index.

Designed to run client-side alongside coding agents like opencode and goose,
giving them access to SearXNG web search and Qdrant vector store through
Proteus's embedding proxy.
"""

import json
import os
from uuid import uuid4

import requests
from mcp.server.fastmcp import FastMCP

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://bigfish:31080").rstrip("/")
QDRANT_URL = os.getenv("QDRANT_URL", "http://bigfish:31333").rstrip("/")
PROTEUS_URL = os.getenv("PROTEUS_URL", "http://bigfish:31400").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

mcp = FastMCP("aiforge")


def _embed(text: str) -> list[float]:
    """Get an embedding vector from Proteus's /v1/embeddings proxy."""
    resp = requests.post(
        f"{PROTEUS_URL}/v1/embeddings",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise ValueError("Empty embedding response from Proteus")
    return data[0]["embedding"]


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for current information on any topic."""
    resp = requests.get(
        f"{SEARXNG_URL}/search",
        params={"q": query, "format": "json", "language": "en"},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])[:5]

    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')}")
        lines.append(f"   {r.get('url', '')}")
        content = r.get("content", "")
        if content:
            lines.append(f"   {content[:200]}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def qdrant_search(query: str, collection: str, limit: int = 5) -> str:
    """Search indexed knowledge in Qdrant vector store.

    Args:
        query: Natural language search query
        collection: Qdrant collection name
        limit: Max number of matches, 1-10
    """
    limit = max(1, min(limit, 10))

    if not query.strip():
        return "Error: qdrant_search requires non-empty 'query'."
    if not collection.strip():
        return "Error: qdrant_search requires non-empty 'collection'."

    vector = _embed(query)

    resp = requests.post(
        f"{QDRANT_URL}/collections/{collection}/points/search",
        json={"vector": vector, "limit": limit, "with_payload": True},
        timeout=20,
    )
    resp.raise_for_status()
    results = resp.json().get("result", [])

    if not results:
        return f"No vector matches found in collection '{collection}' for query: {query}"

    lines = [f"Qdrant results for query: {query} (collection: {collection})", ""]
    for i, point in enumerate(results, 1):
        payload = point.get("payload", {})
        score = point.get("score", 0)
        lines.append(f"{i}. id={point.get('id')} score={score:.4f}")
        if payload:
            snippet = str(payload)[:400]
            lines.append(f"   payload: {snippet}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def qdrant_index(text: str, collection: str, metadata: dict | None = None) -> str:
    """Embed text and store it in a Qdrant collection.

    Args:
        text: Content to embed and store
        collection: Qdrant collection name
        metadata: Optional key-value pairs to attach to the stored point
    """
    if not text.strip():
        return "Error: qdrant_index requires non-empty 'text'."
    if not collection.strip():
        return "Error: qdrant_index requires non-empty 'collection'."

    vector = _embed(text)

    # Create collection if it does not exist
    create_resp = requests.put(
        f"{QDRANT_URL}/collections/{collection}",
        json={
            "vectors": {
                "size": len(vector),
                "distance": "Cosine",
            },
        },
        timeout=10,
    )
    # 409 means collection already exists, which is fine
    if create_resp.status_code not in (200, 409):
        create_resp.raise_for_status()

    point_id = str(uuid4())
    point_payload = {"text": text}
    if metadata:
        point_payload.update(metadata)

    upsert_resp = requests.put(
        f"{QDRANT_URL}/collections/{collection}/points",
        json={
            "points": [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": point_payload,
                }
            ]
        },
        timeout=20,
    )
    upsert_resp.raise_for_status()

    return f"Indexed point {point_id} in collection '{collection}'."


if __name__ == "__main__":
    mcp.run()
