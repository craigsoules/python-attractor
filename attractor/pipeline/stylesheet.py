"""Model stylesheet – CSS-like syntax for centralized LLM configuration."""

from __future__ import annotations

import re
from typing import Any

from attractor.pipeline.graph import Graph, Node


class StylesheetParser:
    """Parse CSS-like stylesheet for LLM model configuration."""

    def parse(self, stylesheet: str) -> list[dict[str, Any]]:
        if not stylesheet:
            return []
        rules: list[dict[str, Any]] = []
        for match in re.finditer(r"([\w\.\#\*-]+)\s*\{([^}]+)\}", stylesheet):
            selector, declarations = match.group(1), match.group(2)
            properties: dict[str, str] = {}
            for decl in declarations.split(";"):
                decl = decl.strip()
                if not decl or ":" not in decl:
                    continue
                prop, value = decl.split(":", 1)
                prop, value = prop.strip(), value.strip()
                if prop in ("llm_model", "llm_provider", "reasoning_effort"):
                    properties[prop] = value
            if properties:
                rules.append({"selector": selector, "properties": properties, "specificity": self._specificity(selector)})
        return rules

    @staticmethod
    def _specificity(selector: str) -> int:
        if selector == "*":
            return 0
        if selector.startswith("#"):
            return 2
        if selector.startswith("."):
            return 1
        return 0


class StylesheetApplier:
    """Apply stylesheet rules to graph nodes (lower-specificity first, higher wins)."""

    def apply(self, graph: Graph) -> None:
        if not graph.model_stylesheet:
            return
        rules = StylesheetParser().parse(graph.model_stylesheet)
        rules.sort(key=lambda r: r["specificity"])
        for node in graph.nodes.values():
            for rule in rules:
                if self._matches(rule["selector"], node):
                    self._apply_props(rule["properties"], node)

    @staticmethod
    def _matches(selector: str, node: Node) -> bool:
        if selector == "*":
            return True
        if selector.startswith("#"):
            return node.id == selector[1:]
        if selector.startswith("."):
            return selector[1:] in node.classes
        return False

    @staticmethod
    def _apply_props(props: dict[str, str], node: Node) -> None:
        if "llm_model" in props:
            node.llm_model = props["llm_model"]
        if "llm_provider" in props:
            node.llm_provider = props["llm_provider"]
        if "reasoning_effort" in props:
            node.reasoning_effort = props["reasoning_effort"]
