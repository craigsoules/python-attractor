"""Node handlers – the executable logic behind each pipeline stage."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable

from attractor.pipeline.graph import Context, Edge, Graph, Node, Outcome, StageStatus
from attractor.pipeline.interviewer import (
    Answer,
    Interviewer,
    Option,
    Question,
    QuestionType,
)


class Handler(ABC):
    """Base interface for node handlers."""

    @abstractmethod
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        ...


class HandlerRegistry:
    """Registry mapping handler types to handler instances."""

    SHAPE_TO_TYPE: dict[str, str] = {
        "Mdiamond": "start",
        "Msquare": "exit",
        "box": "codergen",
        "hexagon": "wait.human",
        "diamond": "conditional",
        "component": "parallel",
        "tripleoctagon": "parallel.fan_in",
        "parallelogram": "tool",
        "house": "stack.manager_loop",
    }

    def __init__(self) -> None:
        self.handlers: dict[str, Handler] = {}
        self.default_handler: Handler | None = None

    def register(self, type_string: str, handler: Handler) -> None:
        self.handlers[type_string] = handler

    def set_default(self, handler: Handler) -> None:
        self.default_handler = handler

    def resolve(self, node: Node) -> Handler:
        # 1. Explicit type attribute
        if node.type and node.type in self.handlers:
            return self.handlers[node.type]
        # 2. Shape-based
        handler_type = self.SHAPE_TO_TYPE.get(node.shape)
        if handler_type and handler_type in self.handlers:
            return self.handlers[handler_type]
        # 3. Default
        if self.default_handler:
            return self.default_handler
        raise ValueError(f"No handler found for node '{node.id}' (shape={node.shape}, type={node.type})")


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------

class StartHandler(Handler):
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)


class ExitHandler(Handler):
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)


class ConditionalHandler(Handler):
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS, notes=f"Conditional: {node.id}")


class CodergenHandler(Handler):
    """LLM task handler – calls a backend or produces simulated output."""

    def __init__(self, backend: Any | None = None) -> None:
        self.backend = backend

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        prompt = node.prompt or node.label or ""
        prompt = prompt.replace("$goal", graph.goal)

        stage_dir = os.path.join(logs_root, node.id)
        os.makedirs(stage_dir, exist_ok=True)

        with open(os.path.join(stage_dir, "prompt.md"), "w") as f:
            f.write(prompt)

        response_text = ""
        if self.backend:
            try:
                result = self.backend.run(node, prompt, context)
                if isinstance(result, Outcome):
                    return result
                response_text = str(result)
            except Exception as exc:
                return Outcome(status=StageStatus.FAIL, failure_reason=str(exc))
        else:
            response_text = f"[Simulated] Response for stage: {node.id}"

        with open(os.path.join(stage_dir, "response.md"), "w") as f:
            f.write(response_text)

        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Stage completed: {node.id}",
            context_updates={
                "last_stage": node.id,
                "last_response": response_text[:200],
                f"stage.{node.id}.response": response_text,
            },
        )


class WaitForHumanHandler(Handler):
    """Wait for human selection at a gate node."""

    def __init__(self, interviewer: Interviewer) -> None:
        self.interviewer = interviewer

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        edges = graph.outgoing_edges(node.id)
        if not edges:
            return Outcome(status=StageStatus.FAIL, failure_reason="No outgoing edges for human gate")

        options: list[Option] = []
        for edge in edges:
            label = edge.label or edge.to_node
            key = self._extract_key(label)
            options.append(Option(key=key, label=label))

        question = Question(
            text=node.label or "Select an option:",
            type=QuestionType.MULTIPLE_CHOICE,
            options=options,
            stage=node.id,
        )

        answer = self.interviewer.ask(question)

        selected = next((o for o in options if o.key == answer.value), options[0] if options else None)
        target_edge = next(
            (e for e in edges if e.label == (selected.label if selected else "") or e.to_node == answer.value),
            edges[0] if edges else None,
        )

        return Outcome(
            status=StageStatus.SUCCESS,
            suggested_next_ids=[target_edge.to_node] if target_edge else [],
            context_updates={
                "human.gate.selected": selected.key if selected else "",
                "human.gate.label": selected.label if selected else "",
            },
        )

    @staticmethod
    def _extract_key(label: str) -> str:
        match = re.search(r"[\[\(]?([A-Za-z0-9])", label)
        if match:
            return match.group(1).upper()
        return label[0].upper() if label else ""


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------

def create_default_handler_registry(
    interviewer: Interviewer | None = None,
    llm_backend: Any | None = None,
) -> HandlerRegistry:
    from attractor.pipeline.interviewer import AutoApproveInterviewer

    registry = HandlerRegistry()
    registry.register("start", StartHandler())
    registry.register("exit", ExitHandler())
    registry.register("codergen", CodergenHandler(backend=llm_backend))
    registry.register("conditional", ConditionalHandler())
    registry.register("wait.human", WaitForHumanHandler(interviewer or AutoApproveInterviewer()))
    registry.set_default(CodergenHandler(backend=llm_backend))
    return registry
