"""Unified LLM type definitions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class ContentKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"


class StreamEventType(str, Enum):
    STREAM_START = "stream_start"
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"
    REASONING_START = "reasoning_start"
    REASONING_DELTA = "reasoning_delta"
    REASONING_END = "reasoning_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    FINISH = "finish"
    ERROR = "error"
    PROVIDER_EVENT = "provider_event"


# ---------------------------------------------------------------------------
# Media data
# ---------------------------------------------------------------------------

@dataclass
class ImageData:
    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    detail: str | None = None


@dataclass
class AudioData:
    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None


@dataclass
class DocumentData:
    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    file_name: str | None = None


# ---------------------------------------------------------------------------
# Tool primitives
# ---------------------------------------------------------------------------

@dataclass
class ToolCallData:
    id: str
    name: str
    arguments: dict | str
    type: str = "function"


@dataclass
class ToolResultData:
    tool_call_id: str
    content: str | dict | list
    is_error: bool = False
    image_data: bytes | None = None
    image_media_type: str | None = None


@dataclass
class ThinkingData:
    text: str
    signature: str | None = None
    redacted: bool = False


# ---------------------------------------------------------------------------
# Content & Message
# ---------------------------------------------------------------------------

@dataclass
class ContentPart:
    kind: ContentKind | str
    text: str | None = None
    image: ImageData | None = None
    audio: AudioData | None = None
    document: DocumentData | None = None
    tool_call: ToolCallData | None = None
    tool_result: ToolResultData | None = None
    thinking: ThinkingData | None = None


@dataclass
class Message:
    role: Role
    content: list[ContentPart]
    name: str | None = None
    tool_call_id: str | None = None

    # -- Convenience constructors ------------------------------------------------

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role=Role.SYSTEM, content=[ContentPart(kind=ContentKind.TEXT, text=text)])

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role=Role.USER, content=[ContentPart(kind=ContentKind.TEXT, text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role=Role.ASSISTANT, content=[ContentPart(kind=ContentKind.TEXT, text=text)])

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str | dict, is_error: bool = False) -> Message:
        result_data = ToolResultData(tool_call_id=tool_call_id, content=content, is_error=is_error)
        return cls(
            role=Role.TOOL,
            content=[ContentPart(kind=ContentKind.TOOL_RESULT, tool_result=result_data)],
            tool_call_id=tool_call_id,
        )

    @property
    def text(self) -> str:
        return "".join(part.text or "" for part in self.content if part.kind == ContentKind.TEXT)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

_TOOL_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    execute: Callable | None = None

    def __post_init__(self) -> None:
        if not _TOOL_NAME_RE.match(self.name):
            raise ValueError(f"Invalid tool name: {self.name}")


@dataclass
class ToolChoice:
    mode: str  # "auto" | "none" | "required" | "named"
    tool_name: str | None = None

    @staticmethod
    def auto() -> ToolChoice:
        return ToolChoice(mode="auto")

    @staticmethod
    def none() -> ToolChoice:
        return ToolChoice(mode="none")

    @staticmethod
    def required() -> ToolChoice:
        return ToolChoice(mode="required")

    @staticmethod
    def named(tool_name: str) -> ToolChoice:
        return ToolChoice(mode="named", tool_name=tool_name)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str | None = None


@dataclass
class ToolResult:
    tool_call_id: str
    content: str | dict | list
    is_error: bool = False


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

@dataclass
class ResponseFormat:
    type: str  # "text" | "json" | "json_schema"
    json_schema: dict | None = None
    strict: bool = False


@dataclass
class Request:
    model: str
    messages: list[Message]
    provider: str | None = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    response_format: ResponseFormat | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    reasoning_effort: str | None = None
    metadata: dict[str, str] | None = None
    provider_options: dict | None = None


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    raw: dict | None = None

    def __add__(self, other: Usage) -> Usage:
        def _sum(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=_sum(self.reasoning_tokens, other.reasoning_tokens),
            cache_read_tokens=_sum(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=_sum(self.cache_write_tokens, other.cache_write_tokens),
        )


@dataclass
class FinishReason:
    reason: str  # "stop" | "length" | "tool_calls" | "content_filter" | "error" | "other"
    raw: str | None = None


@dataclass
class Warning:
    message: str
    code: str | None = None


@dataclass
class RateLimitInfo:
    requests_remaining: int | None = None
    requests_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_limit: int | None = None
    reset_at: str | None = None


@dataclass
class Response:
    id: str
    model: str
    provider: str
    message: Message
    finish_reason: FinishReason
    usage: Usage
    raw: dict | None = None
    warnings: list[Warning] | None = None
    rate_limit: RateLimitInfo | None = None

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for part in self.message.content:
            if part.kind == ContentKind.TOOL_CALL and part.tool_call is not None:
                args = part.tool_call.arguments
                if isinstance(args, str):
                    parsed = json.loads(args)
                    raw = args
                else:
                    parsed = args
                    raw = None
                calls.append(ToolCall(id=part.tool_call.id, name=part.tool_call.name, arguments=parsed, raw_arguments=raw))
        return calls

    @property
    def reasoning(self) -> str | None:
        parts = []
        for part in self.message.content:
            if part.kind in (ContentKind.THINKING, ContentKind.REDACTED_THINKING) and part.thinking:
                parts.append(part.thinking.text)
        return "".join(parts) or None


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    type: StreamEventType | str
    delta: str | None = None
    text_id: str | None = None
    reasoning_delta: str | None = None
    tool_call: ToolCall | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    response: Response | None = None
    error: Any | None = None
    raw: dict | None = None


# ---------------------------------------------------------------------------
# High-level result types
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    text: str
    reasoning: str | None
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    finish_reason: FinishReason
    usage: Usage
    response: Response
    warnings: list[Warning] | None = None


@dataclass
class GenerateResult:
    text: str
    reasoning: str | None
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    finish_reason: FinishReason
    usage: Usage
    total_usage: Usage
    steps: list[StepResult]
    response: Response
    output: Any | None = None


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    id: str
    provider: str
    display_name: str
    context_window: int
    max_output: int | None = None
    supports_tools: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    aliases: list[str] | None = None


MODELS: list[ModelInfo] = [
    ModelInfo(id="claude-opus-4-6", provider="anthropic", display_name="Claude Opus 4.6",
             context_window=200_000, supports_tools=True, supports_vision=True, supports_reasoning=True),
    ModelInfo(id="claude-sonnet-4-5-20250929", provider="anthropic", display_name="Claude Sonnet 4.5",
             context_window=200_000, supports_tools=True, supports_vision=True, supports_reasoning=True),
    ModelInfo(id="gpt-4o", provider="openai", display_name="GPT-4o",
             context_window=128_000, supports_tools=True, supports_vision=True, supports_reasoning=True),
    ModelInfo(id="gpt-4o-mini", provider="openai", display_name="GPT-4o Mini",
             context_window=128_000, supports_tools=True, supports_vision=True, supports_reasoning=True),
    ModelInfo(id="gemini-2.0-flash", provider="gemini", display_name="Gemini 2.0 Flash",
             context_window=1_048_576, supports_tools=True, supports_vision=True, supports_reasoning=True),
]


def get_model_info(model_id: str) -> ModelInfo | None:
    return next((m for m in MODELS if m.id == model_id), None)


def list_models(provider: str | None = None) -> list[ModelInfo]:
    return [m for m in MODELS if provider is None or m.provider == provider]
