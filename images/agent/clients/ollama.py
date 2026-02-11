"""Ollama service client for LLM inference and embeddings."""

import os
from typing import Generator

import requests


class OllamaClient:
    """Client for Ollama API providing chat and embedding capabilities."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://ollama:11434")).rstrip("/")

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """Generate a response from the model.

        Args:
            prompt: The user prompt
            model: Model name, defaults to INTERPRETER_MODEL env var
            system: Optional system prompt
            stream: If True, yields chunks instead of returning full response

        Returns:
            Complete response string or generator of chunks
        """
        model = model or os.getenv("INTERPRETER_MODEL", "qwen2.5:7b")

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system

        if stream:
            return self._stream_generate(payload)

        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _stream_generate(self, payload: dict) -> Generator[str, None, None]:
        """Stream generation response chunks."""
        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if chunk := data.get("response"):
                        yield chunk

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name, defaults to ORCHESTRATOR_MODEL env var
            stream: If True, yields chunks instead of returning full response

        Returns:
            Complete response string or generator of chunks
        """
        model = model or os.getenv("ORCHESTRATOR_MODEL", "qwen3:14b-agent")

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if stream:
            return self._stream_chat(payload)

        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    def _stream_chat(self, payload: dict) -> Generator[str, None, None]:
        """Stream chat response chunks."""
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if content := data.get("message", {}).get("content"):
                        yield content

    def embed(self, text: str | list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for text.

        Args:
            text: Single string or list of strings to embed
            model: Embedding model name, defaults to EMBEDDING_MODEL env var

        Returns:
            List of embedding vectors
        """
        model = model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

        if isinstance(text, str):
            text = [text]

        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": text},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("embeddings", [])

    def health(self) -> bool:
        """Check if Ollama is healthy."""
        try:
            resp = requests.get(f"{self.base_url}/", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
