"""Condition expression language for edge routing."""

from __future__ import annotations

from typing import Callable

from attractor.pipeline.graph import Context, Outcome


class ConditionParser:
    """Parse and evaluate condition expressions.

    Supported syntax:
        outcome=success
        context.key!=value
        clause1 && clause2
    """

    def parse(self, condition: str) -> Callable[[Outcome, Context], bool]:
        if not condition:
            return lambda o, c: True

        clauses = [c.strip() for c in condition.split("&&")]
        evaluators = [self._parse_clause(c) for c in clauses]
        return lambda o, c: all(ev(o, c) for ev in evaluators)

    def _parse_clause(self, clause: str) -> Callable[[Outcome, Context], bool]:
        if "!=" in clause:
            key, value = clause.split("!=", 1)
            key, value = key.strip(), value.strip()
            return lambda o, c, k=key, v=value: self._resolve(k, o, c) != v
        if "=" in clause:
            key, value = clause.split("=", 1)
            key, value = key.strip(), value.strip()
            return lambda o, c, k=key, v=value: self._resolve(k, o, c) == v
        # Bare key → truthy check
        key = clause.strip()
        return lambda o, c, k=key: bool(self._resolve(k, o, c))

    @staticmethod
    def _resolve(key: str, outcome: Outcome, context: Context) -> str:
        if key == "outcome":
            return outcome.status.value
        if key == "preferred_label":
            return outcome.preferred_label
        if key.startswith("context."):
            val = context.get(key[8:])
            return str(val) if val is not None else ""
        val = context.get(key)
        return str(val) if val is not None else ""
