"""Agent session management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from queue import Queue
from typing import Any, Callable

from attractor.agent.environment import ExecutionEnvironment, LocalExecutionEnvironment
from attractor.agent.events import EventEmitter
from attractor.agent.tools import ToolRegistry, create_default_registry
from attractor.llm.client import Client
from attractor.llm.types import Usage


class SessionState(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"


@dataclass
class SessionConfig:
    max_turns: int = 0
    max_tool_rounds_per_input: int = 200
    default_command_timeout_ms: int = 10_000
    max_command_timeout_ms: int = 600_000
    reasoning_effort: str | None = None
    tool_output_limits: dict[str, int] = field(default_factory=dict)
    enable_loop_detection: bool = True
    loop_detection_window: int = 10
    max_subagent_depth: int = 1


# -- Turn types ---------------------------------------------------------------

@dataclass
class UserTurn:
    content: str
    timestamp: datetime


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[Any]
    reasoning: str | None
    usage: Usage | None
    response_id: str | None
    timestamp: datetime


@dataclass
class ToolResultsTurn:
    results: list[Any]
    timestamp: datetime


@dataclass
class SystemTurn:
    content: str
    timestamp: datetime


@dataclass
class SteeringTurn:
    content: str
    timestamp: datetime


Turn = UserTurn | AssistantTurn | ToolResultsTurn | SystemTurn | SteeringTurn


# -- Session -------------------------------------------------------------------

class Session:
    """A single agent session with history, tools, and event emitter."""

    def __init__(
        self,
        *,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-5-20250929",
        llm_client: Client | None = None,
        execution_env: ExecutionEnvironment | None = None,
        config: SessionConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        system_prompt_builder: Callable[..., str] | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.provider = provider
        self.model = model
        self.llm_client = llm_client
        self.execution_env = execution_env or LocalExecutionEnvironment()
        self.config = config or SessionConfig()
        self.tool_registry = tool_registry or create_default_registry()
        self.history: list[Turn] = []
        self.events = EventEmitter()
        self.state = SessionState.IDLE
        self.abort_signaled = False
        self.steering_queue: Queue[str] = Queue()
        self.followup_queue: Queue[str] = Queue()
        self.system_prompt_builder = system_prompt_builder
        self.total_usage = Usage(input_tokens=0, output_tokens=0, total_tokens=0)

    def emit(self, kind: Any, **data: Any) -> None:
        self.events.emit(self.id, kind, **data)
