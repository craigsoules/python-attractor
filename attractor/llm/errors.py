"""Unified LLM error hierarchy."""

from __future__ import annotations


class SDKError(Exception):
    """Base error for the unified LLM SDK."""

    retryable: bool = False

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        self.message = message
        self.cause = cause
        super().__init__(message)


# ---------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------

class ProviderError(SDKError):
    """Provider returned an error."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        raw: dict | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.raw = raw


class AuthenticationError(ProviderError):
    """401 – invalid API key or token."""


class AccessDeniedError(ProviderError):
    """403 – insufficient permissions."""


class NotFoundError(ProviderError):
    """404 – model or endpoint not found."""


class InvalidRequestError(ProviderError):
    """400/422 – malformed request."""


class RateLimitError(ProviderError):
    """429 – rate limit exceeded."""

    retryable = True


class ServerError(ProviderError):
    """500-599 – provider internal error."""

    retryable = True


class ContentFilterError(ProviderError):
    """Response blocked by safety filter."""


class ContextLengthError(ProviderError):
    """Input + output exceeds context window."""


class QuotaExceededError(ProviderError):
    """Billing quota exhausted."""


# ---------------------------------------------------------------------------
# SDK-level errors
# ---------------------------------------------------------------------------

class RequestTimeoutError(SDKError):
    retryable = True


class AbortError(SDKError):
    pass


class NetworkError(SDKError):
    retryable = True


class StreamError(SDKError):
    retryable = True


class InvalidToolCallError(SDKError):
    pass


class NoObjectGeneratedError(SDKError):
    pass


class ConfigurationError(SDKError):
    pass


class UnsupportedToolChoiceError(SDKError):
    pass
