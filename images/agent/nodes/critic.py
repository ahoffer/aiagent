"""Critic node for validating response quality."""

import json
import os
import re

from clients import OllamaClient


CRITIC_PROMPT = """You are a quality control critic. Evaluate this AI response for accuracy and completeness.

User question: {message}

Draft response:
{draft_response}

Available research:
{research_summary}

Evaluate the response and respond with JSON:
{{
    "is_approved": true/false,
    "score": 0.0-1.0,
    "issues": ["list of issues if any"],
    "suggestions": ["improvements if not approved"]
}}

Approve if the response:
1. Directly addresses the user's question
2. Is factually consistent with the research
3. Is clear and well-structured
4. Does not contain obvious hallucinations

Only reject if there are significant issues. Minor stylistic concerns are not grounds for rejection.

Return only valid JSON."""


def critic_node(state: dict) -> dict:
    """Validate the draft response for quality.

    Args:
        state: Current state with draft_response, message, research_summary

    Returns:
        Updated state with is_approved, critique_count, and optional feedback
    """
    message = state.get("message", "")
    draft_response = state.get("draft_response", "")
    research_summary = state.get("research_summary", "")
    critique_count = state.get("critique_count", 0)

    # Increment critique count
    critique_count += 1

    # Auto-approve after max attempts
    max_critiques = 3
    if critique_count >= max_critiques:
        return {
            "is_approved": True,
            "critique_count": critique_count,
            "critique_feedback": "Auto-approved after maximum critique attempts.",
        }

    client = OllamaClient()
    model = os.getenv("CRITIC_MODEL", "qwen2.5:7b")

    prompt = CRITIC_PROMPT.format(
        message=message,
        draft_response=draft_response,
        research_summary=research_summary or "No research available.",
    )

    response = client.generate(prompt, model=model)

    # Parse the critic's response
    try:
        text = response.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?\n?", "", text)
            text = text.strip()

        data = json.loads(text)
        is_approved = data.get("is_approved", True)
        score = float(data.get("score", 0.8))
        issues = data.get("issues", [])
        suggestions = data.get("suggestions", [])

        feedback = ""
        if not is_approved:
            if issues:
                feedback += "Issues: " + "; ".join(issues) + "\n"
            if suggestions:
                feedback += "Suggestions: " + "; ".join(suggestions)

    except (json.JSONDecodeError, ValueError):
        # Default to approved on parse failure
        is_approved = True
        score = 0.7
        feedback = ""

    return {
        "is_approved": is_approved,
        "critique_count": critique_count,
        "critique_score": score,
        "critique_feedback": feedback,
    }
