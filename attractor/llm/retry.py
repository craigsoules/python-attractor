"""Retry policy with exponential backoff."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from attractor.llm.errors import SDKError

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_retries: int = 2
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    on_retry: Callable[[SDKError, int, float], None] | None = None

    def calculate_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (self.backoff_multiplier ** attempt), self.max_delay)
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay


async def retry_call(
    coro_factory: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
) -> T:
    """Execute an async callable with exponential backoff retry."""
    pol = policy or RetryPolicy()
    last_error: SDKError | None = None

    for attempt in range(pol.max_retries + 1):
        try:
            return await coro_factory()
        except SDKError as exc:
            last_error = exc
            if not getattr(exc, "retryable", False) or attempt >= pol.max_retries:
                raise

            delay = pol.calculate_delay(attempt)

            # Honour Retry-After header if present
            retry_after = getattr(exc, "retry_after", None)
            if retry_after and retry_after <= pol.max_delay:
                delay = retry_after
            elif retry_after:
                raise

            if pol.on_retry:
                pol.on_retry(exc, attempt, delay)

            await asyncio.sleep(delay)

    # Should not reach here but satisfy type checker
    assert last_error is not None
    raise last_error
