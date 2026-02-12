"""Validation and linting for pipeline graphs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from attractor.pipeline.conditions import ConditionParser
from attractor.pipeline.graph import Edge, Graph, Node


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Diagnostic:
    rule: str
    severity: Severity
    message: str
    node_id: str | None = None
    edge: tuple[str, str] | None = None
    fix: str = ""


# ---------------------------------------------------------------------------
# Lint rule base
# ---------------------------------------------------------------------------

class LintRule(ABC):
    name: str
    severity: Severity

    @abstractmethod
    def apply(self, graph: Graph) -> list[Diagnostic]:
        ...


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------

class StartNodeRule(LintRule):
    name = "start_node"
    severity = Severity.ERROR

    def apply(self, graph: Graph) -> list[Diagnostic]:
        starts = [n for n in graph.nodes.values() if n.shape == "Mdiamond"]
        if len(starts) != 1:
            return [Diagnostic(rule=self.name, severity=self.severity,
                               message=f"Expected 1 start node (shape=Mdiamond), found {len(starts)}",
                               fix="Add exactly one node with shape=Mdiamond")]
        return []


class ExitNodeRule(LintRule):
    name = "terminal_node"
    severity = Severity.ERROR

    def apply(self, graph: Graph) -> list[Diagnostic]:
        exits = [n for n in graph.nodes.values() if n.shape == "Msquare"]
        if not exits:
            return [Diagnostic(rule=self.name, severity=self.severity,
                               message="No exit node found (shape=Msquare)",
                               fix="Add a node with shape=Msquare")]
        return []


class ReachabilityRule(LintRule):
    name = "reachability"
    severity = Severity.ERROR

    def apply(self, graph: Graph) -> list[Diagnostic]:
        start = next((n for n in graph.nodes.values() if n.shape == "Mdiamond"), None)
        if not start:
            return []
        visited: set[str] = set()
        queue: deque[str] = deque([start.id])
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            for e in graph.outgoing_edges(nid):
                queue.append(e.to_node)
        unreachable = [nid for nid in graph.nodes if nid not in visited]
        if unreachable:
            return [Diagnostic(rule=self.name, severity=self.severity,
                               message=f"Unreachable nodes: {', '.join(unreachable)}",
                               node_id=unreachable[0])]
        return []


class EdgeTargetRule(LintRule):
    name = "edge_target_exists"
    severity = Severity.ERROR

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for e in graph.edges:
            if e.from_node not in graph.nodes:
                diags.append(Diagnostic(rule=self.name, severity=self.severity,
                                        message=f"Edge source '{e.from_node}' not found",
                                        edge=(e.from_node, e.to_node)))
            if e.to_node not in graph.nodes:
                diags.append(Diagnostic(rule=self.name, severity=self.severity,
                                        message=f"Edge target '{e.to_node}' not found",
                                        edge=(e.from_node, e.to_node)))
        return diags


class StartNoIncomingRule(LintRule):
    name = "start_no_incoming"
    severity = Severity.ERROR

    def apply(self, graph: Graph) -> list[Diagnostic]:
        start = next((n for n in graph.nodes.values() if n.shape == "Mdiamond"), None)
        if not start:
            return []
        if graph.incoming_edges(start.id):
            return [Diagnostic(rule=self.name, severity=self.severity,
                               message="Start node must not have incoming edges", node_id=start.id)]
        return []


class ExitNoOutgoingRule(LintRule):
    name = "exit_no_outgoing"
    severity = Severity.ERROR

    def apply(self, graph: Graph) -> list[Diagnostic]:
        for n in graph.nodes.values():
            if n.shape == "Msquare" and graph.outgoing_edges(n.id):
                return [Diagnostic(rule=self.name, severity=self.severity,
                                   message="Exit node must not have outgoing edges", node_id=n.id)]
        return []


class ConditionSyntaxRule(LintRule):
    name = "condition_syntax"
    severity = Severity.ERROR

    def apply(self, graph: Graph) -> list[Diagnostic]:
        parser = ConditionParser()
        diags: list[Diagnostic] = []
        for e in graph.edges:
            if not e.condition:
                continue
            try:
                parser.parse(e.condition)
            except Exception as exc:
                diags.append(Diagnostic(rule=self.name, severity=self.severity,
                                        message=f"Invalid condition: {exc}",
                                        edge=(e.from_node, e.to_node)))
        return diags


class RetryTargetExistsRule(LintRule):
    name = "retry_target_exists"
    severity = Severity.WARNING

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for n in graph.nodes.values():
            if n.retry_target and n.retry_target not in graph.nodes:
                diags.append(Diagnostic(rule=self.name, severity=self.severity,
                                        message=f"Retry target '{n.retry_target}' not found", node_id=n.id))
        return diags


class GoalGateHasRetryRule(LintRule):
    name = "goal_gate_has_retry"
    severity = Severity.WARNING

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for n in graph.nodes.values():
            if n.goal_gate and not n.retry_target and not n.fallback_retry_target:
                diags.append(Diagnostic(rule=self.name, severity=self.severity,
                                        message="Goal gate node should have a retry_target", node_id=n.id))
        return diags


class PromptOnLlmNodesRule(LintRule):
    name = "prompt_on_llm_nodes"
    severity = Severity.WARNING

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for n in graph.nodes.values():
            if n.shape == "box" and not n.prompt and not n.label:
                diags.append(Diagnostic(rule=self.name, severity=self.severity,
                                        message="LLM node should have prompt or label", node_id=n.id))
        return diags


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_BUILTIN_RULES: list[LintRule] = [
    StartNodeRule(),
    ExitNodeRule(),
    ReachabilityRule(),
    EdgeTargetRule(),
    StartNoIncomingRule(),
    ExitNoOutgoingRule(),
    ConditionSyntaxRule(),
    RetryTargetExistsRule(),
    GoalGateHasRetryRule(),
    PromptOnLlmNodesRule(),
]


class Validator:
    """Run validation / lint rules against a pipeline graph."""

    SHAPE_TO_HANDLER: dict[str, str] = {
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

    def validate(self, graph: Graph, extra_rules: list[LintRule] | None = None) -> list[Diagnostic]:
        rules = list(_BUILTIN_RULES)
        if extra_rules:
            rules.extend(extra_rules)
        diags: list[Diagnostic] = []
        for rule in rules:
            diags.extend(rule.apply(graph))
        return diags

    def validate_or_raise(self, graph: Graph, extra_rules: list[LintRule] | None = None) -> list[Diagnostic]:
        diags = self.validate(graph, extra_rules)
        errors = [d for d in diags if d.severity == Severity.ERROR]
        if errors:
            msg = "\n".join(d.message for d in errors)
            raise ValueError(f"Validation failed:\n{msg}")
        return diags
