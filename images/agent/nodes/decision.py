"""Decision node for routing based on interpreted intent."""

import os

from clients import OllamaClient


DECISION_PROMPT = """You are deciding how to handle this user request.

User message: {message}
Detected intent: {intent_type}
Entities: {entities}
Inferred URL: {inferred_url}

Based on this analysis, decide the next action:
1. If intent is "learning" and we have an inferred URL, we should crawl and index it.
2. If intent is "followup", we need to resolve context from conversation history.
3. For questions, we should search our knowledge base and the web.
4. For commands, we should execute the appropriate action.

Respond with just the action name: crawl, resolve_context, research, or execute
"""


def decision_node(state: dict) -> dict:
    """Make routing decision based on interpreted intent.

    Args:
        state: Current state with intent_type, entities, inferred_url

    Returns:
        Updated state with next_action field
    """
    intent_type = state.get("intent_type", "question")
    entities = state.get("entities", [])
    inferred_url = state.get("inferred_url")
    message = state.get("message", "")

    # Fast path for clear intents without LLM
    if intent_type == "learning" and inferred_url:
        return {"next_action": "crawl"}

    if intent_type == "followup":
        return {"next_action": "resolve_context"}

    if intent_type == "question":
        return {"next_action": "research"}

    # For ambiguous cases, use LLM to decide
    client = OllamaClient()
    model = os.getenv("ORCHESTRATOR_MODEL", "qwen3:14b-agent")

    prompt = DECISION_PROMPT.format(
        message=message,
        intent_type=intent_type,
        entities=entities,
        inferred_url=inferred_url or "none",
    )

    response = client.generate(prompt, model=model)
    action = response.strip().lower()

    # Validate action
    valid_actions = ["crawl", "resolve_context", "research", "execute"]
    if action not in valid_actions:
        action = "research"

    return {"next_action": action}
