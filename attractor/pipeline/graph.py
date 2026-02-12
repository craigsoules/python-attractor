"""Core pipeline data structures."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RETRY = "retry"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass
class Outcome:
    status: StageStatus
    preferred_label: str = ""
    suggested_next_ids: list[str] = field(default_factory=list)
    context_updates: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    failure_reason: str = ""


@dataclass
class Node:
    id: str
    label: str = ""
    shape: str = "box"
    type: str = ""
    prompt: str = ""
    max_retries: int = 0
    goal_gate: bool = False
    retry_target: str = ""
    fallback_retry_target: str = ""
    fidelity: str = ""
    thread_id: str = ""
    classes: list[str] = field(default_factory=list)
    timeout: str | None = None
    llm_model: str = ""
    llm_provider: str = ""
    reasoning_effort: str = "high"
    auto_status: bool = False
    allow_partial: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    from_node: str
    to_node: str
    label: str = ""
    condition: str = ""
    weight: int = 0
    fidelity: str = ""
    thread_id: str = ""
    loop_restart: bool = False


@dataclass
class Graph:
    name: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    goal: str = ""
    label: str = ""
    model_stylesheet: str = ""
    default_max_retry: int = 50
    default_fidelity: str = ""
    retry_target: str = ""
    fallback_retry_target: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.from_node == node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.to_node == node_id]


class Context:
    """Thread-safe key-value store for pipeline state."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.logs: list[str] = []
        self._lock = threading.RLock()

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.values.get(key, default)

    def get_string(self, key: str, default: str = "") -> str:
        val = self.get(key)
        return str(val) if val is not None else default

    def append_log(self, entry: str) -> None:
        with self._lock:
            self.logs.append(entry)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.values)

    def clone(self) -> Context:
        with self._lock:
            ctx = Context()
            ctx.values = dict(self.values)
            ctx.logs = list(self.logs)
            return ctx

    def apply_updates(self, updates: dict[str, Any]) -> None:
        with self._lock:
            self.values.update(updates)
