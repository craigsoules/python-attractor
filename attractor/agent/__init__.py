"""Coding Agent Loop – autonomous coding agent with tool execution."""

from attractor.agent.session import Session, SessionConfig, SessionState
from attractor.agent.loop import process_input, submit
from attractor.agent.environment import ExecutionEnvironment, LocalExecutionEnvironment
from attractor.agent.tools import ToolRegistry, RegisteredTool, ToolDefinition
from attractor.agent.events import EventKind, SessionEvent, EventEmitter

__all__ = [
    "Session", "SessionConfig", "SessionState",
    "process_input", "submit",
    "ExecutionEnvironment", "LocalExecutionEnvironment",
    "ToolRegistry", "RegisteredTool", "ToolDefinition",
    "EventKind", "SessionEvent", "EventEmitter",
]
