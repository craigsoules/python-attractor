"""Core LLM client – routing, middleware, provider resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import AsyncIterator

from attractor.llm.adapter import AnthropicAdapter, OpenAIAdapter, ProviderAdapter
from attractor.llm.errors import ConfigurationError
from attractor.llm.types import Request, Response, StreamEvent


@dataclass
class TimeoutConfig:
    total: float | None = None
    per_step: float | None = None


class Client:
    """Main orchestration layer for LLM providers."""

    def __init__(
        self,
        providers: dict[str, ProviderAdapter] | None = None,
        default_provider: str | None = None,
        timeout: TimeoutConfig | None = None,
    ) -> None:
        self.providers: dict[str, ProviderAdapter] = providers or {}
        self.default_provider = default_provider
        self.timeout = timeout or TimeoutConfig()

    @classmethod
    def from_env(cls) -> Client:
        """Auto-detect providers from environment variables."""
        providers: dict[str, ProviderAdapter] = {}

        if os.getenv("ANTHROPIC_API_KEY"):
            providers["anthropic"] = AnthropicAdapter()

        if os.getenv("OPENAI_API_KEY"):
            providers["openai"] = OpenAIAdapter()

        if not providers:
            raise ConfigurationError("No provider API keys found in environment (set ANTHROPIC_API_KEY or OPENAI_API_KEY)")

        default = next(iter(providers))
        return cls(providers=providers, default_provider=default)

    def _resolve_provider(self, provider_name: str | None) -> ProviderAdapter:
        name = provider_name or self.default_provider
        if not name:
            raise ConfigurationError("No provider specified and no default set")
        if name not in self.providers:
            raise ConfigurationError(f"Unknown provider: {name}")
        return self.providers[name]

    async def complete(self, request: Request) -> Response:
        provider = self._resolve_provider(request.provider)
        return await provider.complete(request)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        provider = self._resolve_provider(request.provider)
        async for event in provider.stream(request):
            yield event

    async def close(self) -> None:
        for adapter in self.providers.values():
            await adapter.close()
