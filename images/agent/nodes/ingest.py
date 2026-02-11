"""Ingest node for crawling and indexing documentation."""

import hashlib
import os
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from clients import OllamaClient, QdrantClient


# Common documentation URL patterns
DOC_URL_PATTERNS = {
    "fastapi": "https://fastapi.tiangolo.com/",
    "requests": "https://requests.readthedocs.io/",
    "django": "https://docs.djangoproject.com/",
    "flask": "https://flask.palletsprojects.com/",
    "numpy": "https://numpy.org/doc/stable/",
    "pandas": "https://pandas.pydata.org/docs/",
    "python": "https://docs.python.org/3/",
    "langchain": "https://python.langchain.com/docs/",
    "langgraph": "https://langchain-ai.github.io/langgraph/",
}


def ingest_node(state: dict) -> dict:
    """Crawl and index documentation from a URL.

    Args:
        state: Current state with inferred_url and entities

    Returns:
        Updated state with ingestion results
    """
    inferred_url = state.get("inferred_url")
    entities = state.get("entities", [])
    message = state.get("message", "")

    # Try to determine URL
    url = inferred_url
    if not url:
        url = _infer_url_from_entities(entities, message)

    if not url:
        return {
            "ingestion_status": "failed",
            "ingestion_message": "Could not determine documentation URL to index.",
            "pages_indexed": 0,
        }

    # Create collection name from URL
    collection_name = _url_to_collection_name(url)

    # Crawl and index
    try:
        pages_indexed = _crawl_and_index(url, collection_name, max_pages=20)
        return {
            "ingestion_status": "success",
            "ingestion_message": f"Indexed {pages_indexed} pages from {url}",
            "pages_indexed": pages_indexed,
            "collection_name": collection_name,
        }
    except Exception as e:
        return {
            "ingestion_status": "failed",
            "ingestion_message": f"Failed to index {url}: {str(e)}",
            "pages_indexed": 0,
        }


def _infer_url_from_entities(entities: list[str], message: str) -> str | None:
    """Try to infer documentation URL from entities and message."""
    text = " ".join(entities + [message]).lower()

    for keyword, url in DOC_URL_PATTERNS.items():
        if keyword in text:
            return url

    return None


def _url_to_collection_name(url: str) -> str:
    """Convert URL to a valid collection name."""
    parsed = urlparse(url)
    # Use domain as base, clean up
    name = parsed.netloc.replace(".", "_").replace("-", "_")
    # Add path hash for uniqueness
    if parsed.path and parsed.path != "/":
        path_hash = hashlib.md5(parsed.path.encode()).hexdigest()[:8]
        name = f"{name}_{path_hash}"
    return name


def _crawl_and_index(base_url: str, collection_name: str, max_pages: int = 20) -> int:
    """Crawl pages and index them into Qdrant.

    Args:
        base_url: Starting URL to crawl
        collection_name: Qdrant collection name
        max_pages: Maximum number of pages to index

    Returns:
        Number of pages indexed
    """
    ollama = OllamaClient()
    qdrant = QdrantClient()

    # Create collection
    qdrant.create_collection(collection_name, vector_size=768)

    visited = set()
    to_visit = [base_url]
    indexed = 0

    while to_visit and indexed < max_pages:
        url = to_visit.pop(0)

        if url in visited:
            continue

        visited.add(url)

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract text content
            text = _extract_text(soup)
            if not text or len(text) < 100:
                continue

            # Chunk and embed
            chunks = _chunk_text(text, chunk_size=1000)
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                # Generate embedding
                embeddings = ollama.embed(chunk)
                if not embeddings:
                    continue

                # Store in Qdrant
                chunk_id = hashlib.md5(f"{url}:{i}".encode()).hexdigest()
                qdrant.upsert(
                    collection=collection_name,
                    vectors=[embeddings[0]],
                    payloads=[{"url": url, "text": chunk, "chunk_index": i}],
                    ids=[chunk_id],
                )

            indexed += 1

            # Find more links on the same domain
            base_domain = urlparse(base_url).netloc
            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)
                parsed = urlparse(full_url)

                # Only follow links on same domain
                if parsed.netloc == base_domain and full_url not in visited:
                    # Skip anchors, images, external resources
                    if not any(full_url.endswith(ext) for ext in [".png", ".jpg", ".pdf", ".zip"]):
                        to_visit.append(full_url.split("#")[0])  # Remove anchor

        except Exception:
            # Skip pages that fail
            continue

    return indexed


def _extract_text(soup: BeautifulSoup) -> str:
    """Extract readable text from HTML."""
    # Remove script and style elements
    for element in soup(["script", "style", "nav", "header", "footer"]):
        element.decompose()

    # Get text
    text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text


def _chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """Split text into chunks, trying to break at sentence boundaries."""
    chunks = []
    current_chunk = ""

    # Split by sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
