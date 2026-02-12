"""LLM backend for pipeline codergen nodes – calls a real LLM provider."""

from __future__ import annotations

import asyncio
from typing import Any

from attractor.llm.client import Client
from attractor.llm.types import Message, Request
from attractor.pipeline.graph import Context, Node


class LLMBackend:
    """Connects CodergenHandler to a real LLM via the unified client."""

    def __init__(
        self,
        client: Client | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        provider: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _get_client(self) -> Client:
        if self.client is None:
            self.client = Client.from_env()
        return self.client

    def run(self, node: Node, prompt: str, context: Context) -> str:
        """Synchronous entry point called by CodergenHandler."""
        return asyncio.run(self._run_async(node, prompt, context))

    async def _run_async(self, node: Node, prompt: str, context: Context) -> str:
        client = self._get_client()

        # Resolve model: node-level > backend default
        model = node.llm_model or self.model
        provider = node.llm_provider or self.provider

        # Build messages with context from prior stages
        messages: list[Message] = []
        messages.append(Message.system(
            "You are a precise coding assistant working inside an automated pipeline. "
            "Produce only the requested output — no conversational filler. "
            "When writing code, output the code directly."
        ))

        # Include accumulated context from prior stages
        prior = self._build_prior_context(context)
        if prior:
            messages.append(Message.user(f"Here is context from prior pipeline stages:\n\n{prior}"))
            messages.append(Message.assistant("Understood. I have the context from prior stages. What should I do next?"))

        messages.append(Message.user(prompt))

        request = Request(
            model=model,
            messages=messages,
            provider=provider,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        response = await client.complete(request)
        return response.text

    @staticmethod
    def _build_prior_context(context: Context) -> str:
        """Gather output from prior stages stored in context.

        When iteration keys (stage.X.iter_N.response) exist, builds a
        chronological conversation history across loop iterations.
        Falls back to plain stage.X.response keys for non-loop pipelines.
        """
        import re

        snapshot = context.snapshot()
        parts: list[str] = []

        # Check for iteration-indexed keys
        iter_pattern = re.compile(r"^stage\.(.+?)\.iter_(\d+)\.response$")
        iter_entries: list[tuple[int, str, str]] = []
        for key, value in snapshot.items():
            m = iter_pattern.match(key)
            if m:
                stage_name = m.group(1)
                iteration = int(m.group(2))
                iter_entries.append((iteration, stage_name, str(value)))

        if iter_entries:
            # Sort by iteration number, then by stage name for determinism
            iter_entries.sort(key=lambda e: (e[0], e[1]))
            for iteration, stage_name, value in iter_entries:
                parts.append(f"## Stage: {stage_name} (iteration {iteration})\n{value}")
        else:
            # Non-loop fallback: plain stage.X.response keys
            for key, value in sorted(snapshot.items()):
                if key.startswith("stage.") and key.endswith(".response"):
                    stage = key.removeprefix("stage.").removesuffix(".response")
                    parts.append(f"## Stage: {stage}\n{value}")

        # Also include the rolling last_response if nothing else found
        if not parts:
            last_stage = snapshot.get("last_stage")
            last_resp = snapshot.get("last_response")
            if last_stage and last_resp:
                parts.append(f"## Stage: {last_stage}\n{last_resp}")

        return "\n\n".join(parts)
