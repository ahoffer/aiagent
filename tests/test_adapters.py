"""Tests for the model adapter layer."""

import json
import sys
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from adapters import select_adapter
from adapters.base import ModelAdapter
from adapters.qwen25coder import Qwen25CoderAdapter, _extract_tool_objects
from adapters.qwen3 import Qwen3Adapter, _parse_xml_tool_calls, _strip_think_tags


# -- Adapter selection --

class TestSelectAdapter:

    def test_qwen25coder_prefix_match(self):
        adapter = select_adapter("qwen2.5-coder:14b")
        assert isinstance(adapter, Qwen25CoderAdapter)

    def test_qwen25coder_variant_match(self):
        adapter = select_adapter("qwen2.5-coder:7b")
        assert isinstance(adapter, Qwen25CoderAdapter)

    def test_qwen3_prefix_match(self):
        adapter = select_adapter("qwen3:14b")
        assert isinstance(adapter, Qwen3Adapter)

    def test_unknown_model_gets_base(self):
        adapter = select_adapter("llama3:8b")
        assert type(adapter) is ModelAdapter

    def test_results_are_cached(self):
        a1 = select_adapter("test-cache-model:1b")
        a2 = select_adapter("test-cache-model:1b")
        assert a1 is a2


# -- Base adapter passthrough --

class TestBaseAdapter:

    def setup_method(self):
        self.adapter = ModelAdapter()

    def test_normalize_tool_calls_passthrough(self):
        msg = {"content": "hello", "tool_calls": None}
        assert self.adapter.normalize_tool_calls(msg, set()) is msg

    def test_normalize_ai_message_passthrough(self):
        ai = AIMessage(content="hi")
        assert self.adapter.normalize_ai_message(ai) is ai

    def test_inject_tool_guidance_passthrough(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert self.adapter.inject_tool_guidance(msgs, None) is msgs

    def test_filter_hallucinated_removes_bad_names(self):
        calls = [
            {"function": {"name": "web_search", "arguments": {}}},
            {"function": {"name": "invented_tool", "arguments": {}}},
        ]
        kept = self.adapter._filter_hallucinated(calls, {"web_search"})
        assert len(kept) == 1
        assert kept[0]["function"]["name"] == "web_search"

    def test_filter_hallucinated_noop_without_valid_names(self):
        calls = [{"function": {"name": "anything", "arguments": {}}}]
        assert self.adapter._filter_hallucinated(calls, set()) is calls

    def test_normalize_tool_calls_filters_structured_calls(self):
        """Base adapter filters hallucinated names from structured tool_calls."""
        msg = {
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {}}},
                {"function": {"name": "hallucinated", "arguments": {}}},
            ],
        }
        result = self.adapter.normalize_tool_calls(msg, {"read_file"})
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "read_file"


# -- Tool object extraction --

class TestExtractToolObjects:

    def test_fenced_json_block(self):
        text = 'Here is the call:\n```json\n{"name": "web_search", "arguments": {"query": "test"}}\n```'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 1
        assert candidates[0][1]["name"] == "web_search"

    def test_fenced_block_without_json_label(self):
        text = 'Call:\n```\n{"name": "read_file", "arguments": {"path": "/tmp"}}\n```'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 1
        assert candidates[0][1]["name"] == "read_file"

    def test_bare_json_object(self):
        text = 'I will call {"name": "run_command", "arguments": {"command": "ls"}} now.'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 1
        assert candidates[0][1]["name"] == "run_command"

    def test_invalid_json_ignored(self):
        text = '```json\n{broken json}\n```'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 0

    def test_no_name_key_ignored(self):
        text = '```json\n{"tool": "web_search", "args": {}}\n```'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 0

    def test_multiple_candidates(self):
        text = (
            '```json\n{"name": "read_file", "arguments": {"path": "/a"}}\n```\n'
            'Also: {"name": "web_search", "arguments": {"query": "test"}}'
        )
        candidates = _extract_tool_objects(text)
        names = {c[1]["name"] for c in candidates}
        assert names == {"read_file", "web_search"}

    def test_fenced_not_duplicated_as_bare(self):
        """A JSON object inside a fenced block should not also match as bare."""
        text = '```json\n{"name": "web_search", "arguments": {"query": "test"}}\n```'
        candidates = _extract_tool_objects(text)
        assert len(candidates) == 1


# -- Qwen25Coder proxy path normalization --

class TestQwen25CoderNormalizeToolCalls:

    def setup_method(self):
        self.adapter = Qwen25CoderAdapter()
        self.valid = {"web_search", "read_file", "run_command"}

    def test_structured_calls_filtered_only(self):
        """When Ollama returns structured tool_calls, just filter names."""
        msg = {
            "content": "",
            "tool_calls": [
                {"function": {"name": "web_search", "arguments": {"query": "hi"}}},
                {"function": {"name": "fake_tool", "arguments": {}}},
            ],
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_recovery_from_fenced_json(self):
        msg = {
            "content": 'Here:\n```json\n{"name": "web_search", "arguments": {"query": "python"}}\n```',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "web_search"
        assert result["tool_calls"][0]["function"]["arguments"] == {"query": "python"}
        # JSON stripped from content
        assert "web_search" not in result["content"]

    def test_recovery_from_bare_json(self):
        msg = {
            "content": 'I will call {"name": "run_command", "arguments": {"command": "ls -la"}} to list files.',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "run_command"

    def test_invalid_name_not_recovered(self):
        """Tool calls with names not in valid_names are skipped."""
        msg = {
            "content": '```json\n{"name": "nonexistent_tool", "arguments": {}}\n```',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert result.get("tool_calls") is None

    def test_plain_json_in_conversation_not_fabricated(self):
        """JSON objects without a name+arguments structure are not recovered."""
        msg = {
            "content": 'The config is {"database": "postgres", "port": 5432}.',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert result.get("tool_calls") is None


# -- Qwen25Coder agent path normalization --

class TestQwen25CoderNormalizeAIMessage:

    def setup_method(self):
        self.adapter = Qwen25CoderAdapter()

    def test_passthrough_when_tool_calls_present(self):
        """If AIMessage already has tool_calls, no recovery needed."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}],
        )
        result = self.adapter.normalize_ai_message(msg)
        assert result is msg

    def test_recovery_from_content(self):
        msg = AIMessage(
            content='```json\n{"name": "web_search", "arguments": {"query": "test"}}\n```',
        )
        result = self.adapter.normalize_ai_message(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "web_search"
        assert result.tool_calls[0]["args"] == {"query": "test"}
        assert "web_search" not in result.content

    def test_no_recovery_from_plain_text(self):
        msg = AIMessage(content="The answer is 42.")
        result = self.adapter.normalize_ai_message(msg)
        assert result is msg
        assert not result.tool_calls

    def test_empty_content_passthrough(self):
        msg = AIMessage(content="")
        result = self.adapter.normalize_ai_message(msg)
        assert result is msg


# -- Qwen25Coder tool guidance --

class TestQwen25CoderToolGuidance:

    def setup_method(self):
        self.adapter = Qwen25CoderAdapter()

    def test_injects_system_message(self):
        msgs = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "web_search"}}]
        result = self.adapter.inject_tool_guidance(msgs, tools)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "web_search" in result[0]["content"]

    def test_no_tools_passthrough(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = self.adapter.inject_tool_guidance(msgs, None)
        assert result is msgs

    def test_empty_tools_passthrough(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = self.adapter.inject_tool_guidance(msgs, [])
        assert result is msgs


# -- Qwen3 XML parsing helpers --

class TestQwen3XmlParsing:

    def test_single_function_parsed(self):
        text = '<function=web_search><parameter=query>python docs</parameter></function>'
        calls, cleaned = _parse_xml_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "web_search"
        assert calls[0]["arguments"] == {"query": "python docs"}
        assert cleaned == ""

    def test_multiple_functions_parsed(self):
        text = (
            '<function=web_search><parameter=query>test</parameter></function>'
            '<function=read_file><parameter=path>/tmp/a.txt</parameter></function>'
        )
        calls, cleaned = _parse_xml_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "web_search"
        assert calls[1]["name"] == "read_file"

    def test_multiple_parameters(self):
        text = '<function=run_command><parameter=command>ls -la</parameter><parameter=cwd>/tmp</parameter></function>'
        calls, _ = _parse_xml_tool_calls(text)
        assert calls[0]["arguments"] == {"command": "ls -la", "cwd": "/tmp"}

    def test_surrounding_text_preserved(self):
        text = 'I will search now. <function=web_search><parameter=query>test</parameter></function> Done.'
        calls, cleaned = _parse_xml_tool_calls(text)
        assert len(calls) == 1
        assert "I will search now." in cleaned
        assert "Done." in cleaned
        assert "<function" not in cleaned

    def test_no_xml_returns_empty(self):
        calls, cleaned = _parse_xml_tool_calls("Just plain text.")
        assert calls == []
        assert cleaned == "Just plain text."

    def test_strip_think_tags(self):
        text = "<think>Let me reason about this.</think>The answer is 42."
        assert _strip_think_tags(text) == "The answer is 42."

    def test_strip_think_tags_multiline(self):
        text = "<think>\nStep 1: consider\nStep 2: decide\n</think>\nHere is my answer."
        assert _strip_think_tags(text) == "Here is my answer."

    def test_strip_think_tags_no_think(self):
        text = "No thinking here."
        assert _strip_think_tags(text) == "No thinking here."


# -- Qwen3 proxy path normalization --

class TestQwen3NormalizeToolCalls:

    def setup_method(self):
        self.adapter = Qwen3Adapter()
        self.valid = {"web_search", "read_file", "run_command"}

    def test_single_xml_function_recovered(self):
        msg = {
            "content": '<function=web_search><parameter=query>python docs</parameter></function>',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "web_search"
        assert result["tool_calls"][0]["function"]["arguments"] == {"query": "python docs"}
        assert "<function" not in result["content"]

    def test_multiple_xml_functions_recovered(self):
        msg = {
            "content": (
                '<function=web_search><parameter=query>test</parameter></function>'
                '<function=read_file><parameter=path>/tmp/a.txt</parameter></function>'
            ),
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert len(result["tool_calls"]) == 2
        names = {tc["function"]["name"] for tc in result["tool_calls"]}
        assert names == {"web_search", "read_file"}

    def test_mixed_content_and_xml(self):
        """Text before and after XML is preserved, XML is stripped."""
        msg = {
            "content": 'Let me search. <function=web_search><parameter=query>test</parameter></function> Done.',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert len(result["tool_calls"]) == 1
        assert "Let me search." in result["content"]
        assert "Done." in result["content"]
        assert "<function" not in result["content"]

    def test_structured_calls_passed_through(self):
        """When structured tool_calls exist, filter but do not parse XML."""
        msg = {
            "content": "",
            "tool_calls": [
                {"function": {"name": "web_search", "arguments": {"query": "hi"}}},
                {"function": {"name": "fake_tool", "arguments": {}}},
            ],
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_invalid_names_filtered(self):
        """XML tool calls with unknown names are dropped."""
        msg = {
            "content": '<function=invented_tool><parameter=x>1</parameter></function>',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert not result.get("tool_calls")

    def test_think_tags_stripped(self):
        msg = {
            "content": '<think>Reasoning here.</think>The answer is 42.',
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert "<think>" not in result["content"]
        assert "The answer is 42." in result["content"]

    def test_think_tags_stripped_with_xml_calls(self):
        msg = {
            "content": (
                '<think>Let me think.</think>'
                '<function=web_search><parameter=query>test</parameter></function>'
            ),
            "tool_calls": None,
        }
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert len(result["tool_calls"]) == 1
        assert "<think>" not in result["content"]

    def test_no_xml_passthrough(self):
        """Plain text content without XML passes through unchanged."""
        msg = {"content": "Hello world.", "tool_calls": None}
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert result is msg

    def test_empty_content_passthrough(self):
        msg = {"content": "", "tool_calls": None}
        result = self.adapter.normalize_tool_calls(msg, self.valid)
        assert result is msg


# -- Qwen3 agent path normalization --

class TestQwen3NormalizeAIMessage:

    def setup_method(self):
        self.adapter = Qwen3Adapter()

    def test_xml_recovery_from_content(self):
        msg = AIMessage(
            content='<function=web_search><parameter=query>test</parameter></function>',
        )
        result = self.adapter.normalize_ai_message(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "web_search"
        assert result.tool_calls[0]["args"] == {"query": "test"}
        assert "<function" not in result.content

    def test_passthrough_when_tool_calls_present(self):
        msg = AIMessage(
            content="some text",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}],
        )
        result = self.adapter.normalize_ai_message(msg)
        assert result is msg

    def test_think_tags_stripped(self):
        msg = AIMessage(content="<think>Internal reasoning.</think>The answer is 42.")
        result = self.adapter.normalize_ai_message(msg)
        assert "<think>" not in result.content
        assert "The answer is 42." in result.content

    def test_think_tags_stripped_with_existing_tool_calls(self):
        msg = AIMessage(
            content="<think>Reasoning.</think>Calling tool.",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}],
        )
        result = self.adapter.normalize_ai_message(msg)
        assert "<think>" not in result.content
        assert "Calling tool." in result.content
        assert len(result.tool_calls) == 1

    def test_plain_text_passthrough(self):
        msg = AIMessage(content="The answer is 42.")
        result = self.adapter.normalize_ai_message(msg)
        assert result is msg

    def test_empty_content_passthrough(self):
        msg = AIMessage(content="")
        result = self.adapter.normalize_ai_message(msg)
        assert result is msg

    def test_multiple_xml_calls_recovered(self):
        msg = AIMessage(
            content=(
                '<function=web_search><parameter=query>a</parameter></function>'
                '<function=read_file><parameter=path>/b</parameter></function>'
            ),
        )
        result = self.adapter.normalize_ai_message(msg)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "web_search"
        assert result.tool_calls[1]["name"] == "read_file"
