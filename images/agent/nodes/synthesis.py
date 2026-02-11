"""Synthesis node for generating the final response."""

import os

from clients import OllamaClient


SYNTHESIS_PROMPT = """You are a helpful AI assistant. Generate a clear, accurate response to the user's question.

User question: {message}

Research summary:
{research_summary}

Additional context from knowledge base:
{rag_context}

Instructions:
1. Use the research and context to provide an accurate, well-informed answer.
2. If the research contains relevant URLs or sources, reference them.
3. Be concise but comprehensive.
4. If information is uncertain or incomplete, acknowledge limitations.
5. Do not make up information not present in the research.

Response:"""


def synthesis_node(state: dict) -> dict:
    """Generate final response based on research results.

    Args:
        state: Current state with message, research_results, research_summary

    Returns:
        Updated state with draft_response
    """
    message = state.get("message", "")
    research_summary = state.get("research_summary", "")
    research_results = state.get("research_results", [])

    # Extract RAG context from results
    rag_context_parts = []
    for r in research_results:
        if r.get("source", "").startswith("rag:"):
            content = r.get("content", "")
            if content:
                rag_context_parts.append(content[:500])

    rag_context = "\n".join(rag_context_parts) if rag_context_parts else "No stored context available."

    client = OllamaClient()
    model = os.getenv("SYNTHESIS_MODEL", "qwen3:14b-agent")

    prompt = SYNTHESIS_PROMPT.format(
        message=message,
        research_summary=research_summary or "No research results available.",
        rag_context=rag_context,
    )

    response = client.generate(prompt, model=model)

    return {
        "draft_response": response,
        "sources": _extract_sources(research_results),
    }


def _extract_sources(results: list[dict]) -> list[str]:
    """Extract unique URLs from research results."""
    urls = set()
    for r in results:
        url = r.get("url", "")
        if url:
            urls.add(url)
    return list(urls)[:5]  # Limit to 5 sources
