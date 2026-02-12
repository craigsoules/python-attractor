"""Session event system."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable


class EventKind(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_INPUT = "user_input"
    ASSISTANT_TEXT_START = "assistant_text_start"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_END = "assistant_text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_OUTPUT_DELTA = "tool_call_output_delta"
    TOOL_CALL_END = "tool_call_end"
    STEERING_INJECTED = "steering_injected"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTION = "loop_detection"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class SessionEvent:
    kind: EventKind
    timestamp: datetime
    session_id: str
    data: dict[str, Any]


class EventEmitter:
    """Typed event emitter with sync listeners and async streaming."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[SessionEvent], None]] = []
        self._queues: list[asyncio.Queue[SessionEvent]] = []

    def on(self, callback: Callable[[SessionEvent], None]) -> None:
        self._listeners.append(callback)

    def emit(self, session_id: str, kind: EventKind, **data: Any) -> None:
        event = SessionEvent(
            kind=kind,
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            data=data,
        )
        for listener in self._listeners:
            listener(event)
        for q in self._queues:
            q.put_nowait(event)

    async def stream(self) -> AsyncIterator[SessionEvent]:
        q: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._queues.append(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._queues.remove(q)
