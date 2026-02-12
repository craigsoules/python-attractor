"""Pipeline execution engine."""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any

from attractor.pipeline.conditions import ConditionParser
from attractor.pipeline.graph import Context, Edge, Graph, Node, Outcome, StageStatus
from attractor.pipeline.handlers import Handler, HandlerRegistry
from attractor.pipeline.stylesheet import StylesheetApplier


class PipelineEngine:
    """Execute a DOT pipeline graph from start to exit."""

    def __init__(
        self,
        registry: HandlerRegistry,
        on_stage_complete: Callable[[Node, Outcome, Context], None] | None = None,
    ) -> None:
        self.registry = registry
        self.on_stage_complete = on_stage_complete
        self.condition_parser = ConditionParser()

    def run(self, graph: Graph, logs_root: str = "./logs") -> Outcome:
        # Apply stylesheet
        StylesheetApplier().apply(graph)

        # Initialize context
        context = Context()
        context.set("graph.goal", graph.goal)
        context.set("graph.label", graph.label)
        Path(logs_root).mkdir(parents=True, exist_ok=True)

        completed: list[str] = []
        node_outcomes: dict[str, Outcome] = {}
        node_retries: dict[str, int] = {}
        node_visits: dict[str, int] = {}

        current = self._find_start(graph)
        if not current:
            raise ValueError("No start node found in graph")

        while True:
            node = graph.nodes[current]

            # Terminal node
            if self._is_terminal(node):
                gate_ok, failed = self._check_goal_gates(graph, node_outcomes)
                if not gate_ok and failed:
                    retry = self._get_retry_target(failed, graph)
                    if retry:
                        current = retry
                        continue
                    raise ValueError(f"Goal gate '{failed.id}' unsatisfied with no retry target")
                break

            # Execute with retry
            retry_policy = self._retry_policy(node, graph)
            outcome = self._execute_with_retry(node, context, graph, logs_root, retry_policy, node_retries)

            completed.append(node.id)
            node_outcomes[node.id] = outcome

            # Apply context updates
            context.apply_updates(outcome.context_updates)
            context.set("outcome", outcome.status.value)
            if outcome.preferred_label:
                context.set("preferred_label", outcome.preferred_label)

            # Track iteration-indexed responses for loop awareness
            visit = node_visits.get(node.id, 0) + 1
            node_visits[node.id] = visit
            response_key = f"stage.{node.id}.response"
            response_val = context.get(response_key)
            if response_val is not None:
                context.set(f"stage.{node.id}.iter_{visit}.response", response_val)

            # Notify caller
            if self.on_stage_complete:
                self.on_stage_complete(node, outcome, context)

            # Checkpoint
            self._checkpoint(logs_root, context, current, completed, node_retries)

            # Select next edge
            next_edge = self._select_edge(node, outcome, context, graph)
            if not next_edge:
                if outcome.status == StageStatus.FAIL:
                    raise ValueError(f"Stage '{node.id}' failed with no outgoing edge")
                break

            if next_edge.loop_restart:
                return outcome

            current = next_edge.to_node

        return Outcome(status=StageStatus.SUCCESS, notes=f"Pipeline '{graph.name}' completed")

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _find_start(graph: Graph) -> str | None:
        for n in graph.nodes.values():
            if n.shape == "Mdiamond":
                return n.id
        return None

    @staticmethod
    def _is_terminal(node: Node) -> bool:
        return node.shape == "Msquare"

    @staticmethod
    def _check_goal_gates(graph: Graph, outcomes: dict[str, Outcome]) -> tuple[bool, Node | None]:
        for nid, outcome in outcomes.items():
            node = graph.nodes[nid]
            if node.goal_gate and outcome.status not in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS):
                return False, node
        return True, None

    @staticmethod
    def _get_retry_target(node: Node, graph: Graph) -> str | None:
        if node.retry_target and node.retry_target in graph.nodes:
            return node.retry_target
        if node.fallback_retry_target and node.fallback_retry_target in graph.nodes:
            return node.fallback_retry_target
        return None

    @staticmethod
    def _retry_policy(node: Node, graph: Graph) -> dict[str, Any]:
        max_retries = node.max_retries or graph.default_max_retry
        return {"max_attempts": max_retries + 1, "initial_delay_ms": 200, "backoff_factor": 2.0, "max_delay_ms": 60_000, "jitter": True}

    def _execute_with_retry(
        self, node: Node, context: Context, graph: Graph, logs_root: str,
        policy: dict[str, Any], node_retries: dict[str, int],
    ) -> Outcome:
        for attempt in range(policy["max_attempts"]):
            try:
                handler = self.registry.resolve(node)
                outcome = handler.execute(node, context, graph, logs_root)

                if outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS):
                    node_retries[node.id] = 0
                    return outcome

                if outcome.status == StageStatus.RETRY:
                    if attempt < policy["max_attempts"] - 1:
                        node_retries[node.id] = attempt + 1
                        time.sleep(self._delay(attempt + 1, policy) / 1000)
                        continue
                    if node.allow_partial:
                        return Outcome(status=StageStatus.PARTIAL_SUCCESS, notes="Retries exhausted, partial accepted")
                    return Outcome(status=StageStatus.FAIL, failure_reason="Max retries exceeded")

                if outcome.status == StageStatus.FAIL:
                    return outcome

            except Exception as exc:
                if attempt < policy["max_attempts"] - 1:
                    time.sleep(self._delay(attempt + 1, policy) / 1000)
                    continue
                return Outcome(status=StageStatus.FAIL, failure_reason=str(exc))

        return Outcome(status=StageStatus.FAIL, failure_reason="Max retries exceeded")

    @staticmethod
    def _delay(attempt: int, policy: dict[str, Any]) -> float:
        delay = policy["initial_delay_ms"] * (policy["backoff_factor"] ** (attempt - 1))
        delay = min(delay, policy["max_delay_ms"])
        if policy.get("jitter"):
            delay *= random.uniform(0.5, 1.5)
        return delay

    @staticmethod
    def _checkpoint(logs_root: str, context: Context, current: str, completed: list[str], retries: dict[str, int]) -> None:
        cp = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_node": current,
            "completed_nodes": completed,
            "node_retries": retries,
            "context": context.snapshot(),
        }
        with open(f"{logs_root}/checkpoint.json", "w") as f:
            json.dump(cp, f, indent=2, default=str)

    def _select_edge(self, node: Node, outcome: Outcome, context: Context, graph: Graph) -> Edge | None:
        edges = graph.outgoing_edges(node.id)
        if not edges:
            return None

        # 1. Condition-matching
        cond_matched = [e for e in edges if e.condition and self.condition_parser.parse(e.condition)(outcome, context)]
        if cond_matched:
            return self._best_weight(cond_matched)

        # 2. Preferred label
        if outcome.preferred_label:
            norm = outcome.preferred_label.lower().strip()
            for e in edges:
                if e.label.lower().strip() == norm:
                    return e

        # 3. Suggested next IDs
        for sid in outcome.suggested_next_ids:
            for e in edges:
                if e.to_node == sid:
                    return e

        # 4/5. Unconditional by weight
        unconditional = [e for e in edges if not e.condition]
        if unconditional:
            return self._best_weight(unconditional)
        return self._best_weight(edges)

    @staticmethod
    def _best_weight(edges: list[Edge]) -> Edge:
        return sorted(edges, key=lambda e: (-e.weight, e.to_node))[0]
