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


class FreeformHumanHandler(Handler):
    """Wait for freeform human text input."""

    _DONE_WORDS = {"done", "exit", "quit", "finish"}

    def __init__(self, interviewer: Interviewer) -> None:
        self.interviewer = interviewer

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        question = Question(
            text=node.label or "Enter your input:",
            type=QuestionType.FREEFORM,
            stage=node.id,
        )

        answer = self.interviewer.ask(question)
        text = answer.text or answer.value or ""
        is_done = text.strip().lower() in self._DONE_WORDS

        return Outcome(
            status=StageStatus.SUCCESS,
            preferred_label="done" if is_done else "",
            context_updates={
                f"stage.{node.id}.response": text,
                "human.input": text,
                "last_stage": node.id,
                "last_response": text[:200],
            },
        )


class ToolHandler(Handler):
    """Execute a tool function referenced by node attributes."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., tuple[bool, str]]] = {
            "validate_dot": self._validate_dot,
            "write_file": self._write_file,
            "read_file": self._read_file,
        }

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        tool_name = node.attributes.get("tool", "")
        if not tool_name:
            return Outcome(status=StageStatus.FAIL, failure_reason="No 'tool' attribute on node")

        tool_fn = self._tools.get(tool_name)
        if not tool_fn:
            return Outcome(status=StageStatus.FAIL, failure_reason=f"Unknown tool: {tool_name}")

        try:
            success, result_text = tool_fn(node, context)
        except Exception as exc:
            return Outcome(status=StageStatus.FAIL, failure_reason=str(exc))

        status = StageStatus.SUCCESS if success else StageStatus.FAIL
        return Outcome(
            status=status,
            notes=result_text[:200],
            failure_reason="" if success else result_text,
            context_updates={
                f"stage.{node.id}.response": result_text,
                "last_stage": node.id,
                "last_response": result_text[:200],
            },
        )

    @staticmethod
    def _validate_dot(node: Node, context: Context) -> tuple[bool, str]:
        from attractor.pipeline.parser import DotParser
        from attractor.pipeline.validation import Severity, Validator

        dot_text = context.get("last_response", "")
        # Try to find full DOT content from the most recent stage response
        snapshot = context.snapshot()
        for key in sorted(snapshot.keys(), reverse=True):
            if key.startswith("stage.") and key.endswith(".response"):
                val = str(snapshot[key])
                if "digraph" in val:
                    dot_text = val
                    break

        if not dot_text or "digraph" not in dot_text:
            return False, "No DOT content found in context to validate."

        # Extract DOT block if embedded in markdown or other text
        import re
        match = re.search(r"(digraph\s+\w+\s*\{.*\})", dot_text, re.DOTALL)
        if match:
            dot_text = match.group(1)

        try:
            graph = DotParser().parse(dot_text)
        except Exception as exc:
            return False, f"Parse error: {exc}"

        diags = Validator().validate(graph)
        errors = [d for d in diags if d.severity == Severity.ERROR]
        if errors:
            msg = "\n".join(f"- {d.message}" for d in errors)
            return False, f"Validation errors:\n{msg}"

        warnings = [d for d in diags if d.severity == Severity.WARNING]
        if warnings:
            msg = "\n".join(f"- {d.message}" for d in warnings)
            return True, f"Valid (with warnings):\n{msg}"

        return True, "Valid: no errors or warnings."

    @staticmethod
    def _write_file(node: Node, context: Context) -> tuple[bool, str]:
        path = node.attributes.get("path", "")
        if not path:
            # Try to derive from context
            path = context.get("output_path", "")
        if not path:
            return False, "No 'path' attribute on node and no 'output_path' in context."

        # Find the most recent DOT content
        content = ""
        snapshot = context.snapshot()
        for key in sorted(snapshot.keys(), reverse=True):
            if key.startswith("stage.") and key.endswith(".response"):
                val = str(snapshot[key])
                if "digraph" in val:
                    content = val
                    break

        if not content:
            return False, "No content found to write."

        # Extract DOT block if embedded
        import re
        match = re.search(r"(digraph\s+\w+\s*\{.*\})", content, re.DOTALL)
        if match:
            content = match.group(1)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

        return True, f"Written to {path}"

    @staticmethod
    def _read_file(node: Node, context: Context) -> tuple[bool, str]:
        path = node.attributes.get("path", "")
        if not path:
            return False, "No 'path' attribute on node."

        if not os.path.exists(path):
            return False, f"File not found: {path}"

        with open(path, "r") as f:
            content = f.read()

        return True, content


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
    registry.register("wait.human.freeform", FreeformHumanHandler(interviewer or AutoApproveInterviewer()))
    registry.register("tool", ToolHandler())
    registry.set_default(CodergenHandler(backend=llm_backend))
    return registry
