"""Tests for the unified LLM type system."""

import pytest

from attractor.llm.types import (
    ContentKind,
    ContentPart,
    FinishReason,
    Message,
    ModelInfo,
    Role,
    Tool,
    ToolCall,
    ToolChoice,
    Usage,
    get_model_info,
    list_models,
)


def test_message_system():
    m = Message.system("You are helpful.")
    assert m.role == Role.SYSTEM
    assert m.text == "You are helpful."


def test_message_user():
    m = Message.user("Hello")
    assert m.role == Role.USER
    assert m.text == "Hello"


def test_message_assistant():
    m = Message.assistant("Hi there")
    assert m.role == Role.ASSISTANT
    assert m.text == "Hi there"


def test_message_tool_result():
    m = Message.tool_result("call_1", "result data", is_error=False)
    assert m.role == Role.TOOL
    assert m.tool_call_id == "call_1"
    assert m.content[0].kind == ContentKind.TOOL_RESULT
    assert m.content[0].tool_result.content == "result data"


def test_usage_addition():
    a = Usage(input_tokens=10, output_tokens=5, total_tokens=15, reasoning_tokens=2)
    b = Usage(input_tokens=20, output_tokens=10, total_tokens=30, reasoning_tokens=3)
    c = a + b
    assert c.input_tokens == 30
    assert c.output_tokens == 15
    assert c.total_tokens == 45
    assert c.reasoning_tokens == 5


def test_usage_addition_none_fields():
    a = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
    b = Usage(input_tokens=20, output_tokens=10, total_tokens=30)
    c = a + b
    assert c.reasoning_tokens is None


def test_tool_valid_name():
    t = Tool(name="read_file", description="Read a file", parameters={"type": "object"})
    assert t.name == "read_file"


def test_tool_invalid_name():
    with pytest.raises(ValueError, match="Invalid tool name"):
        Tool(name="123bad", description="Bad name", parameters={})


def test_tool_choice_modes():
    assert ToolChoice.auto().mode == "auto"
    assert ToolChoice.none().mode == "none"
    assert ToolChoice.required().mode == "required"
    named = ToolChoice.named("read_file")
    assert named.mode == "named"
    assert named.tool_name == "read_file"


def test_model_catalog():
    info = get_model_info("claude-opus-4-6")
    assert info is not None
    assert info.provider == "anthropic"
    assert info.supports_tools is True

    models = list_models("anthropic")
    assert len(models) >= 1
    assert all(m.provider == "anthropic" for m in models)

    assert get_model_info("nonexistent-model") is None
