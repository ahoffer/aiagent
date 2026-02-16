"""Adapter for qwen3 family.

qwen3 supports native tool calling, but with 6+ tool schemas it falls
back to XML in the content field instead of structured tool_calls.
The XML follows Hermes conventions:
  <function=NAME><parameter=KEY>VALUE</parameter></function>

This adapter recovers those XML tool calls into OpenAI-compatible
tool_calls, strips <think> reasoning tags, and delegates to the base
class hallucination filter when structured calls are already present.
"""

import logging
import re

from langchain_core.messages import AIMessage

from adapters.base import ModelAdapter


log = logging.getLogger(__name__)

# <think>...</think> blocks that qwen3 may prepend to reasoning
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Hermes-style XML tool calls emitted when 6+ tools are in context
_FUNCTION_BLOCK = re.compile(
    r"<function=([^>]+)>(.*?)</function>",
    re.DOTALL,
)
_PARAMETER = re.compile(
    r"<parameter=([^>]+)>(.*?)</parameter>",
    re.DOTALL,
)


def _parse_xml_tool_calls(text: str) -> tuple[list[dict], str]:
    """Extract Hermes-style XML tool calls from content text.

    Returns a list of parsed tool call dicts and the content with
    all XML tool call markup removed.
    """
    calls = []
    for match in _FUNCTION_BLOCK.finditer(text):
        name = match.group(1).strip()
        body = match.group(2)
        args = {}
        for param in _PARAMETER.finditer(body):
            args[param.group(1).strip()] = param.group(2)
        calls.append({"name": name, "arguments": args})

    cleaned = _FUNCTION_BLOCK.sub("", text).strip()
    return calls, cleaned


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from content."""
    return _THINK_BLOCK.sub("", text).strip()


class Qwen3Adapter(ModelAdapter):
    """Recovers XML tool calls and strips think tags from qwen3 output."""

    def normalize_tool_calls(self, ollama_msg: dict,
                             valid_names: set[str]) -> dict:
        """Recover tool_calls from XML when qwen3 falls back to content."""
        content = ollama_msg.get("content", "") or ""
        tool_calls = ollama_msg.get("tool_calls") or []

        # Structured calls present means fewer than 6 tools were sent,
        # so qwen3 used native JSON. Just filter hallucinated names.
        if tool_calls:
            result = dict(ollama_msg)
            result["tool_calls"] = self._filter_hallucinated(
                tool_calls, valid_names)
            result["content"] = _strip_think_tags(content)
            return result

        # Try XML recovery from content
        if content:
            content = _strip_think_tags(content)
            parsed, cleaned = _parse_xml_tool_calls(content)
            if parsed:
                recovered = []
                for call in parsed:
                    if call["name"] in valid_names:
                        recovered.append({
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            }
                        })
                    else:
                        log.debug("XML tool name=%r not in valid set, skipping",
                                  call["name"])
                if recovered:
                    log.info("recovered %d XML tool call(s) from content",
                             len(recovered))
                    result = dict(ollama_msg)
                    result["tool_calls"] = recovered
                    result["content"] = cleaned
                    return result

        # No recovery needed, just strip think tags
        if content != (ollama_msg.get("content", "") or ""):
            result = dict(ollama_msg)
            result["content"] = content
            return result
        return ollama_msg

    def normalize_ai_message(self, message: AIMessage) -> AIMessage:
        """Recover XML tool calls from AIMessage content on the agent path."""
        content = message.content or ""

        # If structured tool_calls already present, just strip think tags
        if message.tool_calls:
            stripped = _strip_think_tags(content)
            if stripped != content:
                return AIMessage(
                    content=stripped,
                    tool_calls=message.tool_calls,
                )
            return message

        if not content:
            return message

        content = _strip_think_tags(content)
        parsed, cleaned = _parse_xml_tool_calls(content)
        if not parsed:
            # No XML found but think tags may have been stripped
            if content != (message.content or ""):
                return AIMessage(content=content)
            return message

        tool_calls = []
        for call in parsed:
            tool_calls.append({
                "name": call["name"],
                "args": call["arguments"],
                "id": f"recovered_{len(tool_calls)}",
            })

        log.info("recovered %d XML tool call(s) from AIMessage content",
                 len(tool_calls))
        return AIMessage(content=cleaned, tool_calls=tool_calls)
