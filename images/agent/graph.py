"""LangGraph workflow for autonomous multi-agent system."""

from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from nodes import (
    interpreter_node,
    decision_node,
    research_node,
    synthesis_node,
    critic_node,
    ingest_node,
)


class AgentState(TypedDict, total=False):
    """Shared state for the agent graph.

    All nodes read from and write to this state. Fields use total=False
    so nodes only need to return the fields they modify.
    """
    # Input
    message: str
    conversation_id: str

    # Interpreter output
    intent_type: str
    confidence: float
    entities: list[str]
    inferred_url: str | None

    # Decision output
    next_action: str

    # Research output
    research_results: list[dict[str, Any]]
    research_summary: str

    # Ingestion output
    ingestion_status: str
    ingestion_message: str
    pages_indexed: int
    collection_name: str

    # Synthesis output
    draft_response: str
    sources: list[str]

    # Critic output
    is_approved: bool
    critique_count: int
    critique_score: float
    critique_feedback: str

    # Final output
    final_response: str
    actions_taken: list[str]


def route_after_decision(state: AgentState) -> Literal["crawl", "research", "resolve_context"]:
    """Route based on decision node output."""
    action = state.get("next_action", "research")

    if action == "crawl":
        return "crawl"
    elif action == "resolve_context":
        # For now, resolve_context falls back to research
        return "resolve_context"
    else:
        return "research"


def route_after_critic(state: AgentState) -> Literal["answer", "revise"]:
    """Route based on critic approval."""
    is_approved = state.get("is_approved", False)
    critique_count = state.get("critique_count", 0)

    # Approve if passed or hit max attempts
    if is_approved or critique_count >= 3:
        return "answer"

    return "revise"


def resolve_context_node(state: AgentState) -> dict:
    """Resolve context for followup questions.

    For now, this falls through to research with original message.
    Future enhancement could use conversation history.
    """
    return {
        "research_summary": f"Followup context: {state.get('message', '')}",
    }


def revise_node(state: AgentState) -> dict:
    """Revise the draft response based on critic feedback.

    Re-runs synthesis with feedback incorporated.
    """
    feedback = state.get("critique_feedback", "")
    draft = state.get("draft_response", "")

    # Add feedback to research summary for synthesis to consider
    current_summary = state.get("research_summary", "")
    enhanced_summary = f"{current_summary}\n\nPrevious draft issues: {feedback}"

    return {
        "research_summary": enhanced_summary,
    }


def answer_node(state: AgentState) -> dict:
    """Finalize the response."""
    draft = state.get("draft_response", "")
    actions = []

    # Collect actions taken
    if state.get("ingestion_status") == "success":
        actions.append("crawl")
    if state.get("research_results"):
        actions.append("research")
    if state.get("critique_count", 0) > 1:
        actions.append("revise")

    actions.append("synthesize")

    return {
        "final_response": draft,
        "actions_taken": actions,
    }


def ingest_then_synthesize(state: AgentState) -> dict:
    """After ingestion, create a synthesis response about what was indexed."""
    status = state.get("ingestion_status", "")
    message = state.get("ingestion_message", "")
    pages = state.get("pages_indexed", 0)
    collection = state.get("collection_name", "")

    if status == "success":
        response = (
            f"I've indexed the documentation you requested. "
            f"Processed {pages} pages and stored them in the '{collection}' collection. "
            f"You can now ask questions about this documentation."
        )
    else:
        response = f"I encountered an issue while indexing: {message}"

    return {
        "draft_response": response,
        "is_approved": True,  # Skip critic for ingestion responses
    }


def build_graph() -> StateGraph:
    """Build and compile the agent workflow graph.

    Workflow:
    1. Interpreter detects intent
    2. Decision routes to appropriate action
    3. Action nodes: crawl, research, or resolve_context
    4. Synthesis generates response
    5. Critic validates quality
    6. Either answer or revise based on critic

    Returns:
        Compiled StateGraph ready for invocation
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("interpreter", interpreter_node)
    graph.add_node("decision", decision_node)
    graph.add_node("crawl", ingest_node)
    graph.add_node("ingest_response", ingest_then_synthesize)
    graph.add_node("research", research_node)
    graph.add_node("resolve_context", resolve_context_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("critic", critic_node)
    graph.add_node("revise", revise_node)
    graph.add_node("answer", answer_node)

    # Set entry point
    graph.set_entry_point("interpreter")

    # Add edges
    graph.add_edge("interpreter", "decision")

    # Conditional routing after decision
    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "crawl": "crawl",
            "research": "research",
            "resolve_context": "resolve_context",
        },
    )

    # Crawl path goes directly to ingest response
    graph.add_edge("crawl", "ingest_response")
    graph.add_edge("ingest_response", "answer")

    # Research and context paths go to synthesis
    graph.add_edge("research", "synthesis")
    graph.add_edge("resolve_context", "research")

    # Synthesis goes to critic
    graph.add_edge("synthesis", "critic")

    # Conditional routing after critic
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "answer": "answer",
            "revise": "revise",
        },
    )

    # Revise goes back to synthesis
    graph.add_edge("revise", "synthesis")

    # Answer is terminal
    graph.add_edge("answer", END)

    return graph.compile()


# Create the compiled graph as a module-level variable
agent_graph = build_graph()
