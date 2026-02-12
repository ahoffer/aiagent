# proteus.py
"""FastAPI entry point for Proteus.

Proteus is now a thin HTTP shim around the LangGraph agent:
- /chat and /chat/stream for Open WebUI-style usage
- /v1/chat/completions OpenAI-compatible endpoint (non-streaming + basic streaming)

All orchestration (tool loop, iteration limit, etc.) lives in LangGraph (graph.py).

Conversation memory:
- Use LangGraph checkpointer by passing config.thread_id
- /chat uses conversation_id as thread_id
- /v1/chat/completions: optionally accept a conversation_id as an extra field
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from graph import agent_graph
from clients import OllamaClient, SearxngClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("proteus")

LOG_LANGGRAPH_OUTPUT = os.getenv("LOG_LANGGRAPH_OUTPUT", "true").lower() in ("1", "true")


# ----------------------------
# Native request/response models
# ----------------------------

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []
    search_count: int = 0
    conversation_id: str


class HealthResponse(BaseModel):
    status: str
    services: dict[str, bool] = {}


# ----------------------------
# OpenAI-compatible models
# ----------------------------

class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string per OpenAI spec


class OpenAIToolCall(BaseModel):
    id: str
    type: str = "function"
    function: OpenAIFunctionCall


class OpenAIMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class OpenAIChatRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = "proteus"
    messages: list[OpenAIMessage] = []
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


def _msg_to_dict(msg) -> dict:
    """Convert an OpenAIMessage (or dict) to a plain dict message."""
    if isinstance(msg, dict):
        return msg
    d = {"role": msg.role}
    if msg.content is not None:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id is not None:
        d["tool_call_id"] = msg.tool_call_id
    if msg.name is not None:
        d["name"] = msg.name
    return d


def _thread_id_from_openai_request(request: OpenAIChatRequest) -> str:
    """
    Prefer a caller-supplied conversation_id (extra field), else create a new one.
    This lets OpenAI clients opt into server-side memory by sending conversation_id.
    """
    try:
        data = request.model_dump()  # includes extras because extra=allow
    except Exception:
        data = {}
    conv_id = data.get("conversation_id")
    if isinstance(conv_id, str) and conv_id.strip():
        return conv_id.strip()
    return uuid4().hex


# ----------------------------
# App setup
# ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Proteus starting up")
    yield
    log.info("Proteus shutting down")


app = FastAPI(
    title="Proteus",
    description="Thin HTTP shim over LangGraph agent",
    version="4.0.0",
    lifespan=lifespan,
)


def _dependency_health() -> dict[str, bool]:
    return {
        "ollama": OllamaClient().health(),
        "searxng": SearxngClient().health(),
    }


def _overall_health_status(services: dict[str, bool]) -> str:
    if services and all(services.values()):
        return "healthy"
    if services and any(services.values()):
        return "degraded"
    return "unhealthy"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    services = await asyncio.to_thread(_dependency_health)
    status = _overall_health_status(services)
    return HealthResponse(status=status, services=services)


@app.get("/health/live", response_model=HealthResponse)
async def health_live():
    # Liveness should not depend on upstream services.
    return HealthResponse(status="healthy", services={})


@app.get("/health/ready", response_model=HealthResponse)
async def health_ready():
    services = await asyncio.to_thread(_dependency_health)
    status = _overall_health_status(services)
    status_code = 200 if status == "healthy" else 503
    body = HealthResponse(status=status, services=services).model_dump()
    return JSONResponse(status_code=status_code, content=body)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    log.info("POST /chat conversation_id=%s", request.conversation_id)
    conversation_id = request.conversation_id or str(uuid4())

    try:
        result = await asyncio.to_thread(
            agent_graph.invoke,
            {"message": request.message, "conversation_id": conversation_id},
            {"configurable": {"thread_id": conversation_id}},
        )

        if LOG_LANGGRAPH_OUTPUT:
            preview = (result.get("final_response", "") or "")[:200].replace("\n", " ")
            log.info(
                "LangGraph /chat result conversation_id=%s keys=%s search_count=%s final_preview=%r",
                conversation_id,
                sorted(result.keys()),
                result.get("search_count", 0),
                preview,
            )

        return ChatResponse(
            response=result.get("final_response", "") or "",
            sources=result.get("sources", []) or [],
            search_count=int(result.get("search_count", 0) or 0),
            conversation_id=conversation_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    log.info("POST /chat/stream conversation_id=%s", request.conversation_id)
    conversation_id = request.conversation_id or str(uuid4())

    async def generate():
        try:
            graph_input = {"message": request.message, "conversation_id": conversation_id}
            outputs = await asyncio.to_thread(
                lambda: list(
                    agent_graph.stream(
                        graph_input,
                        config={"configurable": {"thread_id": conversation_id}},
                    )
                )
            )

            for output in outputs:
                for node_name, node_output in output.items():
                    keys = sorted(node_output.keys()) if isinstance(node_output, dict) else []
                    preview = ""
                    if isinstance(node_output, dict):
                        preview = (node_output.get("final_response", "") or "")[:160].replace("\n", " ")
                    if LOG_LANGGRAPH_OUTPUT:
                        log.info(
                            "LangGraph /chat/stream node=%s conversation_id=%s keys=%s final_preview=%r",
                            node_name,
                            conversation_id,
                            keys,
                            preview,
                        )

                    yield f"event: node\ndata: {node_name}\n\n"

                    if isinstance(node_output, dict) and "final_response" in node_output:
                        response = node_output.get("final_response") or ""
                        yield f"event: response\ndata: {response}\n\n"

            yield "event: done\ndata: complete\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "proteus", "object": "model", "created": 0, "owned_by": "local"}
        ],
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    log.info("POST /v1/chat/completions messages=%d stream=%s", len(request.messages), request.stream)

    if request.stream:
        return await _openai_streaming(request)

    return await _openai_non_streaming(request)


async def _openai_non_streaming(request: OpenAIChatRequest):
    completion_id = f"chatcmpl-{uuid4().hex[:12]}"
    created = int(time.time())

    thread_id = _thread_id_from_openai_request(request)
    messages = [_msg_to_dict(m) for m in request.messages]

    try:
        result = await asyncio.to_thread(
            agent_graph.invoke,
            {"messages": messages, "tools": request.tools or []},
            {"configurable": {"thread_id": thread_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    content = result.get("final_response", "") or ""
    resp = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model or "proteus",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    return resp


async def _openai_streaming(request: OpenAIChatRequest):
    completion_id = f"chatcmpl-{uuid4().hex[:12]}"
    created = int(time.time())

    thread_id = _thread_id_from_openai_request(request)
    messages = [_msg_to_dict(m) for m in request.messages]

    async def generate():
        try:
            result = await asyncio.to_thread(
                agent_graph.invoke,
                {"messages": messages, "tools": request.tools or []},
                {"configurable": {"thread_id": thread_id}},
            )
            content = result.get("final_response", "") or ""

            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model or "proteus",
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            done = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model or "proteus",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = {"error": str(e)}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def main():
    parser = argparse.ArgumentParser(description="Proteus")
    parser.add_argument("--serve", action="store_true", help="Start the HTTP server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("message", nargs="*", help="Message to process (CLI mode)")
    args = parser.parse_args()

    if args.serve:
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.message:
        message = " ".join(args.message)
        # CLI: single-shot unless you add a conversation_id and pass thread_id.
        result = agent_graph.invoke({"message": message})
        print(result.get("final_response", "No response"))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
