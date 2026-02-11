"""Research node for gathering information from RAG and web search."""

import os
from typing import Any

from clients import OllamaClient, QdrantClient, SearxngClient


def research_node(state: dict) -> dict:
    """Gather relevant information from knowledge base and web.

    Args:
        state: Current state with message and entities

    Returns:
        Updated state with research_results
    """
    message = state.get("message", "")
    entities = state.get("entities", [])

    ollama = OllamaClient()
    qdrant = QdrantClient()
    searxng = SearxngClient()

    results: list[dict[str, Any]] = []

    # Search vector database for relevant context
    try:
        collections = qdrant.list_collections()
        if collections:
            # Embed the query
            embeddings = ollama.embed(message)
            if embeddings:
                query_vector = embeddings[0]

                # Search each collection
                for collection in collections[:3]:  # Limit to avoid slow queries
                    try:
                        hits = qdrant.search(
                            collection=collection,
                            query_vector=query_vector,
                            limit=3,
                            score_threshold=0.7,
                        )
                        for hit in hits:
                            results.append({
                                "source": f"rag:{collection}",
                                "content": hit.get("payload", {}).get("text", ""),
                                "score": hit.get("score", 0),
                                "url": hit.get("payload", {}).get("url", ""),
                            })
                    except Exception:
                        # Skip collections with errors
                        pass
    except Exception:
        # Continue without RAG if unavailable
        pass

    # Perform web search
    try:
        # Build search query from message and entities
        search_query = message
        if entities:
            search_query = f"{message} {' '.join(entities[:3])}"

        web_results = searxng.search(search_query)
        for r in web_results[:5]:
            results.append({
                "source": "web",
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "url": r.get("url", ""),
            })
    except Exception:
        # Continue without web search if unavailable
        pass

    # Summarize results if we have many
    summary = _summarize_results(results, message, ollama) if results else ""

    return {
        "research_results": results,
        "research_summary": summary,
    }


def _summarize_results(
    results: list[dict[str, Any]],
    query: str,
    ollama: OllamaClient,
) -> str:
    """Summarize research results into coherent context."""
    if not results:
        return ""

    # Build context from results
    context_parts = []
    for r in results[:10]:
        source = r.get("source", "unknown")
        content = r.get("content", "")
        if content:
            context_parts.append(f"[{source}] {content[:500]}")

    if not context_parts:
        return ""

    context = "\n\n".join(context_parts)

    prompt = f"""Summarize the following research results to answer this query: {query}

Results:
{context}

Provide a concise summary of the most relevant information. Focus on facts and details that directly address the query."""

    model = os.getenv("RESEARCH_MODEL", "qwen2.5:7b")
    return ollama.generate(prompt, model=model)
