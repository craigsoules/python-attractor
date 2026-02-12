"""Core agentic loop – the heart of the coding agent."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from attractor.agent.environment import ExecutionEnvironment
from attractor.agent.events import EventKind
from attractor.agent.session import (
    AssistantTurn,
    Session,
    SessionState,
    SteeringTurn,
    ToolResultsTurn,
    UserTurn,
)
from attractor.agent.truncation import truncate_tool_output
from attractor.llm.client import Client
from attractor.llm.types import (
    ContentKind,
    ContentPart,
    Message,
    Request,
    Response,
    Role,
    Tool,
    ToolCall,
    ToolCallData,
    ToolChoice,
    ToolResult,
    ToolResultData,
    Usage,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# History → Messages conversion
# ---------------------------------------------------------------------------

def _convert_history(history: list[Any]) -> list[Message]:
    """Convert session history turns into LLM Message objects."""
    messages: list[Message] = []
    for turn in history:
        if isinstance(turn, UserTurn):
            messages.append(Message.user(turn.content))
        elif isinstance(turn, AssistantTurn):
            parts: list[ContentPart] = []
            if turn.content:
                parts.append(ContentPart(kind=ContentKind.TEXT, text=turn.content))
            for tc in turn.tool_calls:
                parts.append(ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(id=tc.id, name=tc.name, arguments=tc.arguments),
                ))
            messages.append(Message(role=Role.ASSISTANT, content=parts))
        elif isinstance(turn, ToolResultsTurn):
            for tr in turn.results:
                messages.append(Message.tool_result(
                    tool_call_id=tr.tool_call_id,
                    content=tr.content,
                    is_error=tr.is_error,
                ))
        elif isinstance(turn, SteeringTurn):
            messages.append(Message.user(f"[SYSTEM]: {turn.content}"))
    return messages


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(session: Session) -> str:
    env = session.execution_env
    parts = [
        "You are an autonomous coding agent. You help users by reading, writing, and editing code.",
        "Use the provided tools to accomplish tasks. Be concise and accurate.",
        "",
        f"Working directory: {env.working_directory()}",
        f"Platform: {env.platform()}",
        f"OS: {env.os_version()}",
        f"Model: {session.model}",
    ]

    # Tool descriptions
    parts.append("\nAvailable tools:")
    for td in session.tool_registry.definitions():
        parts.append(f"  - {td.name}: {td.description}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _tool_defs_for_llm(session: Session) -> list[Tool]:
    """Convert registry tools into LLM Tool objects (without executors)."""
    return [
        Tool(name=td.name, description=td.description, parameters=td.parameters)
        for td in session.tool_registry.definitions()
    ]


def _execute_tool_call(session: Session, tc: ToolCall) -> ToolResult:
    registered = session.tool_registry.get(tc.name)
    if not registered:
        return ToolResult(tool_call_id=tc.id, content=f"Unknown tool: {tc.name}", is_error=True)
    try:
        session.emit(EventKind.TOOL_CALL_START, tool_name=tc.name, call_id=tc.id)
        raw_output = registered.executor(tc.arguments, session.execution_env)
        truncated = truncate_tool_output(raw_output, tc.name, session.config.tool_output_limits)
        session.emit(EventKind.TOOL_CALL_END, call_id=tc.id, output=raw_output)
        return ToolResult(tool_call_id=tc.id, content=truncated, is_error=False)
    except Exception as exc:
        msg = f"Tool error ({tc.name}): {exc}"
        session.emit(EventKind.TOOL_CALL_END, call_id=tc.id, error=msg)
        return ToolResult(tool_call_id=tc.id, content=msg, is_error=True)


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

def _extract_tool_signatures(history: list[Any], last: int) -> list[str]:
    """Extract last N tool call signatures from history."""
    sigs: list[str] = []
    for turn in reversed(history):
        if isinstance(turn, AssistantTurn):
            for tc in turn.tool_calls:
                sigs.append(f"{tc.name}({json.dumps(tc.arguments, sort_keys=True)})")
                if len(sigs) >= last:
                    return list(reversed(sigs))
    return list(reversed(sigs))


def _detect_loop(history: list[Any], window: int) -> bool:
    sigs = _extract_tool_signatures(history, window)
    if len(sigs) < window:
        return False
    for pattern_len in (1, 2, 3):
        if window % pattern_len != 0:
            continue
        pattern = sigs[:pattern_len]
        if all(sigs[i:i + pattern_len] == pattern for i in range(pattern_len, window, pattern_len)):
            return True
    return False


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------

def _drain_steering(session: Session) -> None:
    while not session.steering_queue.empty():
        msg = session.steering_queue.get()
        session.history.append(SteeringTurn(content=msg, timestamp=_now()))
        session.emit(EventKind.STEERING_INJECTED, content=msg)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def process_input(session: Session, user_input: str) -> None:
    """Main agent loop: LLM call → tool execution → repeat until done."""
    if session.llm_client is None:
        session.llm_client = Client.from_env()

    session.state = SessionState.PROCESSING
    session.history.append(UserTurn(content=user_input, timestamp=_now()))
    session.emit(EventKind.USER_INPUT, content=user_input)
    _drain_steering(session)

    round_count = 0

    while True:
        # Check limits
        if round_count >= session.config.max_tool_rounds_per_input:
            session.emit(EventKind.TURN_LIMIT, round=round_count)
            break

        if session.abort_signaled:
            break

        # Build request
        build_prompt = session.system_prompt_builder or _build_system_prompt
        system_prompt = build_prompt(session)
        messages = [Message.system(system_prompt)] + _convert_history(session.history)
        tool_defs = _tool_defs_for_llm(session)

        request = Request(
            model=session.model,
            messages=messages,
            tools=tool_defs if tool_defs else None,
            tool_choice=ToolChoice.auto() if tool_defs else None,
            reasoning_effort=session.config.reasoning_effort,
            provider=session.provider,
        )

        # Call LLM
        response: Response = await session.llm_client.complete(request)

        # Record assistant turn
        tcalls = response.tool_calls
        assistant_turn = AssistantTurn(
            content=response.text,
            tool_calls=tcalls,
            reasoning=response.reasoning,
            usage=response.usage,
            response_id=response.id,
            timestamp=_now(),
        )
        session.history.append(assistant_turn)
        session.total_usage = session.total_usage + response.usage
        session.emit(EventKind.ASSISTANT_TEXT_END, text=response.text, reasoning=response.reasoning)

        # Natural completion
        if not tcalls:
            break

        # Execute tools
        round_count += 1
        results = [_execute_tool_call(session, tc) for tc in tcalls]
        session.history.append(ToolResultsTurn(results=results, timestamp=_now()))

        # Drain steering
        _drain_steering(session)

        # Loop detection
        if session.config.enable_loop_detection:
            if _detect_loop(session.history, session.config.loop_detection_window):
                warning = f"Loop detected: last {session.config.loop_detection_window} tool calls repeat. Try a different approach."
                session.history.append(SteeringTurn(content=warning, timestamp=_now()))
                session.emit(EventKind.LOOP_DETECTION, message=warning)

    # Handle follow-ups
    if not session.followup_queue.empty():
        next_input = session.followup_queue.get()
        await process_input(session, next_input)
        return

    session.state = SessionState.IDLE
    session.emit(EventKind.SESSION_END)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def submit(session: Session, user_input: str) -> None:
    """Synchronous entry point – runs the async loop."""
    asyncio.run(process_input(session, user_input))


def steer(session: Session, message: str) -> None:
    """Queue a steering message for injection after the current tool round."""
    session.steering_queue.put(message)


def follow_up(session: Session, message: str) -> None:
    """Queue a message to process after the current input completes."""
    session.followup_queue.put(message)
