"""Unified LLM Client – provider-agnostic interface for language model interaction."""

from attractor.llm.types import (
    ContentKind,
    ContentPart,
    FinishReason,
    Message,
    Request,
    Response,
    ResponseFormat,
    Role,
    StreamEvent,
    StreamEventType,
    Tool,
    ToolCall,
    ToolCallData,
    ToolChoice,
    ToolResult,
    ToolResultData,
    Usage,
    Warning,
    RateLimitInfo,
    ModelInfo,
)
from attractor.llm.errors import (
    SDKError,
    ProviderError,
    AuthenticationError,
    AccessDeniedError,
    NotFoundError,
    InvalidRequestError,
    RateLimitError,
    ServerError,
    ContentFilterError,
    ContextLengthError,
    RequestTimeoutError,
    AbortError,
    NetworkError,
    StreamError,
    InvalidToolCallError,
    ConfigurationError,
)
from attractor.llm.adapter import ProviderAdapter
from attractor.llm.client import Client
from attractor.llm.api import generate, stream

__all__ = [
    "Role", "ContentKind", "ContentPart", "Message",
    "Tool", "ToolChoice", "ToolCall", "ToolCallData", "ToolResult", "ToolResultData",
    "Request", "Response", "ResponseFormat", "Usage", "FinishReason",
    "Warning", "RateLimitInfo", "ModelInfo",
    "StreamEvent", "StreamEventType",
    "SDKError", "ProviderError", "AuthenticationError", "AccessDeniedError",
    "NotFoundError", "InvalidRequestError", "RateLimitError", "ServerError",
    "ContentFilterError", "ContextLengthError", "RequestTimeoutError",
    "AbortError", "NetworkError", "StreamError", "InvalidToolCallError",
    "ConfigurationError",
    "ProviderAdapter", "Client",
    "generate", "stream",
]
