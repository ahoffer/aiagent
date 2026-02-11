"""Interpreter node for detecting user intent from natural language."""

import json
import os
import re

from clients import OllamaClient


INTENT_PROMPT = """Analyze this user message and determine the intent.

User message: {message}

Respond with a JSON object containing:
- intent: one of "learning", "question", "followup", "command"
- confidence: float between 0 and 1
- entities: list of key entities mentioned
- inferred_url: if intent is "learning" and message references docs, infer the documentation URL

Intent definitions:
- learning: User wants to learn about or index documentation (for example "learn the fastapi docs", "index the requests library")
- question: User asks a factual or technical question
- followup: User references previous context (for example "explain that again", "what did you mean")
- command: User wants to perform a specific action

Return only valid JSON, no other text.
"""


def interpreter_node(state: dict) -> dict:
    """Interpret user message and detect intent.

    Args:
        state: Current graph state with 'message' key

    Returns:
        Updated state with intent_type, confidence, entities, and inferred_url
    """
    message = state.get("message", "")
    client = OllamaClient()

    model = os.getenv("INTERPRETER_MODEL", "qwen2.5:7b")
    prompt = INTENT_PROMPT.format(message=message)

    response = client.generate(prompt, model=model)

    # Parse JSON from response, handling possible markdown code blocks
    try:
        text = response.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?\n?", "", text)
            text = text.strip()

        data = json.loads(text)
        intent = data.get("intent", "question")
        confidence = float(data.get("confidence", 0.5))
        entities = data.get("entities", [])
        inferred_url = data.get("inferred_url")
    except (json.JSONDecodeError, ValueError):
        # Default to question intent on parse failure
        intent = "question"
        confidence = 0.5
        entities = []
        inferred_url = None

    return {
        "intent_type": intent,
        "confidence": confidence,
        "entities": entities,
        "inferred_url": inferred_url,
    }
