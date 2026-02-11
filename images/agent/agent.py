"""FastAPI entry point for the multi-agent system."""

import argparse
import sys
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from clients import OllamaClient, QdrantClient, SearxngClient
from graph import agent_graph


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    response: str
    intent: str
    actions_taken: list[str]
    confidence: float
    conversation_id: str
    sources: list[str] = []


class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str
    services: dict[str, bool]


class CollectionInfo(BaseModel):
    """Information about a collection."""
    name: str
    count: int


class CollectionsResponse(BaseModel):
    """Response body for collections endpoint."""
    collections: list[CollectionInfo]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    print("Agent starting up...")
    yield
    print("Agent shutting down...")


app = FastAPI(
    title="AI Agent",
    description="Autonomous multi-agent system with LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of the agent and its dependencies."""
    ollama = OllamaClient()
    qdrant = QdrantClient()
    searxng = SearxngClient()

    services = {
        "ollama": ollama.health(),
        "qdrant": qdrant.health(),
        "searxng": searxng.health(),
    }

    # Agent is healthy if at least Ollama is available
    overall = "healthy" if services["ollama"] else "degraded"

    return HealthResponse(status=overall, services=services)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the agent graph.

    The agent interprets the message, decides on actions, gathers
    information, synthesizes a response, and validates quality.
    """
    conversation_id = request.conversation_id or str(uuid4())

    try:
        # Run the graph
        result = agent_graph.invoke({
            "message": request.message,
            "conversation_id": conversation_id,
        })

        return ChatResponse(
            response=result.get("final_response", ""),
            intent=result.get("intent_type", "unknown"),
            actions_taken=result.get("actions_taken", []),
            confidence=result.get("confidence", 0.0),
            conversation_id=conversation_id,
            sources=result.get("sources", []),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response with progress updates.

    Uses Server-Sent Events to stream node completions and final response.
    """
    conversation_id = request.conversation_id or str(uuid4())

    async def generate():
        try:
            # Stream through graph nodes
            for output in agent_graph.stream({
                "message": request.message,
                "conversation_id": conversation_id,
            }):
                # Each output is a dict with node name as key
                for node_name, node_output in output.items():
                    # Send node completion event
                    yield f"event: node\ndata: {node_name}\n\n"

                    # If we have a final response, send it
                    if "final_response" in node_output:
                        response = node_output["final_response"]
                        yield f"event: response\ndata: {response}\n\n"

            yield "event: done\ndata: complete\n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """List all indexed collections in the vector database."""
    qdrant = QdrantClient()

    try:
        names = qdrant.list_collections()
        collections = []
        for name in names:
            try:
                count = qdrant.count(name)
                collections.append(CollectionInfo(name=name, count=count))
            except Exception:
                collections.append(CollectionInfo(name=name, count=0))

        return CollectionsResponse(collections=collections)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(description="AI Agent")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the HTTP server",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "message",
        nargs="*",
        help="Message to process (CLI mode)",
    )

    args = parser.parse_args()

    if args.serve:
        # Start HTTP server
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.message:
        # CLI mode: process message directly
        message = " ".join(args.message)
        result = agent_graph.invoke({"message": message})
        print(result.get("final_response", "No response"))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
