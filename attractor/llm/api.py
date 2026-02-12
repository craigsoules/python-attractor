"""High-level generate / stream API."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from attractor.llm.client import Client, TimeoutConfig
from attractor.llm.retry import RetryPolicy, retry_call
from attractor.llm.types import (
    GenerateResult,
    Message,
    Request,
    Response,
    ResponseFormat,
    StepResult,
    StreamEvent,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    Usage,
)

# ---------------------------------------------------------------------------
# Module-level default client
# ---------------------------------------------------------------------------

_default_client: Client | None = None


def set_default_client(client: Client) -> None:
    global _default_client
    _default_client = client


def _get_default_client() -> Client:
    global _default_client
    if _default_client is None:
        _default_client = Client.from_env()
    return _default_client


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

async def _execute_tools_parallel(tools: list[Tool], tool_calls: list[ToolCall]) -> list[ToolResult]:
    async def _run_one(call: ToolCall) -> ToolResult:
        tool = next((t for t in tools if t.name == call.name), None)
        if not tool:
            return ToolResult(tool_call_id=call.id, content=f"Unknown tool: {call.name}", is_error=True)
        if not tool.execute:
            return ToolResult(tool_call_id=call.id, content=f"Tool {call.name} has no execute handler", is_error=True)
        try:
            result = tool.execute(**call.arguments)
            if asyncio.iscoroutine(result):
                result = await result
            return ToolResult(tool_call_id=call.id, content=result, is_error=False)
        except Exception as exc:
            return ToolResult(tool_call_id=call.id, content=str(exc), is_error=True)

    results = await asyncio.gather(*[_run_one(c) for c in tool_calls])
    return list(results)


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

async def generate(
    model: str,
    prompt: str | None = None,
    messages: list[Message] | None = None,
    system: str | None = None,
    tools: list[Tool] | None = None,
    tool_choice: ToolChoice | None = None,
    max_tool_rounds: int = 1,
    stop_when: Callable[[list[StepResult]], bool] | None = None,
    response_format: ResponseFormat | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    provider_options: dict | None = None,
    max_retries: int = 2,
    timeout: TimeoutConfig | float | None = None,
    client: Client | None = None,
) -> GenerateResult:
    """Generate text with optional multi-round tool execution loop."""
    if prompt and messages:
        raise ValueError("Provide either prompt or messages, not both")

    c = client or _get_default_client()

    msg_list: list[Message] = list(messages) if messages else []
    if prompt:
        msg_list = [Message.user(prompt)]
    if system:
        msg_list.insert(0, Message.system(system))

    conversation = list(msg_list)
    steps: list[StepResult] = []
    policy = RetryPolicy(max_retries=max_retries)

    for round_num in range(max_tool_rounds + 1):
        request = Request(
            model=model,
            messages=conversation,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            reasoning_effort=reasoning_effort,
            provider=provider,
            provider_options=provider_options,
        )

        response: Response = await retry_call(lambda: c.complete(request), policy)
        tcalls = response.tool_calls
        tresults: list[ToolResult] = []

        if tcalls and tools:
            tresults = await _execute_tools_parallel(tools, tcalls)

        step = StepResult(
            text=response.text,
            reasoning=response.reasoning,
            tool_calls=tcalls,
            tool_results=tresults,
            finish_reason=response.finish_reason,
            usage=response.usage,
            response=response,
        )
        steps.append(step)

        if not tcalls or response.finish_reason.reason != "tool_calls":
            break
        if round_num >= max_tool_rounds:
            break
        if stop_when and stop_when(steps):
            break

        # Continue conversation
        conversation.append(response.message)
        for tr in tresults:
            conversation.append(Message.tool_result(tool_call_id=tr.tool_call_id, content=tr.content, is_error=tr.is_error))

    total_usage = steps[0].usage
    for s in steps[1:]:
        total_usage = total_usage + s.usage

    final = steps[-1]
    return GenerateResult(
        text=final.text,
        reasoning=final.reasoning,
        tool_calls=final.tool_calls,
        tool_results=final.tool_results,
        finish_reason=final.finish_reason,
        usage=final.usage,
        total_usage=total_usage,
        steps=steps,
        response=final.response,
    )


# ---------------------------------------------------------------------------
# stream() – thin wrapper, delegates to client
# ---------------------------------------------------------------------------

async def stream(
    model: str,
    prompt: str | None = None,
    messages: list[Message] | None = None,
    system: str | None = None,
    provider: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    client: Client | None = None,
):
    """Stream generation with incremental events."""
    if prompt and messages:
        raise ValueError("Provide either prompt or messages, not both")

    c = client or _get_default_client()
    msg_list: list[Message] = list(messages) if messages else []
    if prompt:
        msg_list = [Message.user(prompt)]
    if system:
        msg_list.insert(0, Message.system(system))

    request = Request(model=model, messages=msg_list, provider=provider, max_tokens=max_tokens, temperature=temperature)
    async for event in c.stream(request):
        yield event
