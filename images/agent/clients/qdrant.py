"""Qdrant vector database client for RAG storage and retrieval."""

import os
from typing import Any
from uuid import uuid4

import requests


class QdrantClient:
    """Client for Qdrant vector database operations."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("QDRANT_URL", "http://qdrant:6333")).rstrip("/")

    def list_collections(self) -> list[str]:
        """List all collection names."""
        resp = requests.get(f"{self.base_url}/collections", timeout=30)
        resp.raise_for_status()
        result = resp.json().get("result", {}).get("collections", [])
        return [c["name"] for c in result]

    def create_collection(self, name: str, vector_size: int = 768) -> bool:
        """Create a new collection if it does not exist.

        Args:
            name: Collection name
            vector_size: Dimension of vectors, 768 for nomic-embed-text

        Returns:
            True if created, False if already exists
        """
        if name in self.list_collections():
            return False

        resp = requests.put(
            f"{self.base_url}/collections/{name}",
            json={
                "vectors": {
                    "size": vector_size,
                    "distance": "Cosine",
                }
            },
            timeout=30,
        )
        resp.raise_for_status()
        return True

    def upsert(
        self,
        collection: str,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """Insert or update vectors in a collection.

        Args:
            collection: Collection name
            vectors: List of embedding vectors
            payloads: Optional metadata for each vector
            ids: Optional IDs, auto-generated if not provided

        Returns:
            Number of points upserted
        """
        if ids is None:
            ids = [str(uuid4()) for _ in vectors]
        if payloads is None:
            payloads = [{} for _ in vectors]

        points = [
            {"id": id_, "vector": vec, "payload": payload}
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]

        resp = requests.put(
            f"{self.base_url}/collections/{collection}/points",
            json={"points": points},
            timeout=60,
        )
        resp.raise_for_status()
        return len(points)

    def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            collection: Collection name
            query_vector: Vector to search for
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of results with id, score, and payload
        """
        payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
        }
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold

        resp = requests.post(
            f"{self.base_url}/collections/{collection}/points/search",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

        results = resp.json().get("result", [])
        return [
            {
                "id": r["id"],
                "score": r["score"],
                "payload": r.get("payload", {}),
            }
            for r in results
        ]

    def delete_collection(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if deleted
        """
        resp = requests.delete(f"{self.base_url}/collections/{name}", timeout=30)
        resp.raise_for_status()
        return True

    def count(self, collection: str) -> int:
        """Get the number of points in a collection."""
        resp = requests.get(f"{self.base_url}/collections/{collection}", timeout=30)
        resp.raise_for_status()
        return resp.json().get("result", {}).get("points_count", 0)

    def health(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            resp = requests.get(f"{self.base_url}/healthz", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
