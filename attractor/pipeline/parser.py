"""DOT DSL parser for pipeline graphs."""

from __future__ import annotations

import re
from typing import Any

from attractor.pipeline.graph import Edge, Graph, Node


class DotParser:
    """Parse a DOT digraph source string into a Graph."""

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.pos = 0

    def parse(self, source: str) -> Graph:
        source = self._strip_comments(source)
        self.tokens = self._tokenize(source)
        self.pos = 0
        return self._parse_graph()

    # -- Lexing ---------------------------------------------------------------

    @staticmethod
    def _strip_comments(source: str) -> str:
        source = re.sub(r"//.*?$", "", source, flags=re.MULTILINE)
        source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
        return source

    @staticmethod
    def _tokenize(source: str) -> list[str]:
        pattern = r'("(?:[^"\\]|\\.)*"|\w+|[{};,\[\]=]|->)'
        return re.findall(pattern, source)

    # -- Token helpers --------------------------------------------------------

    def _cur(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, expected: str | None = None) -> str:
        token = self._cur()
        if expected is not None and token != expected:
            raise SyntaxError(f"Expected '{expected}', got '{token}' at token {self.pos}")
        self.pos += 1
        return token  # type: ignore[return-value]

    def _consume_optional(self, token: str) -> bool:
        if self._cur() == token:
            self.pos += 1
            return True
        return False

    def _peek(self, offset: int = 1) -> str | None:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else None

    # -- Parsing --------------------------------------------------------------

    def _parse_graph(self) -> Graph:
        self._consume("digraph")
        name = self._consume()
        self._consume("{")

        graph = Graph(name=name)
        default_node: dict[str, str] = {}
        default_edge: dict[str, str] = {}

        while self._cur() != "}":
            token = self._cur()

            if token == "graph":
                self._consume("graph")
                attrs = self._parse_attr_block()
                for k, v in attrs.items():
                    if k == "goal":
                        graph.goal = v
                    elif k == "label":
                        graph.label = v
                    elif k == "model_stylesheet":
                        graph.model_stylesheet = v
                    elif k == "default_max_retry":
                        graph.default_max_retry = int(v)
                    else:
                        graph.attributes[k] = v
                self._consume_optional(";")

            elif token == "node":
                self._consume("node")
                default_node = self._parse_attr_block()
                self._consume_optional(";")

            elif token == "edge":
                self._consume("edge")
                default_edge = self._parse_attr_block()
                self._consume_optional(";")

            elif token == "subgraph":
                self._consume("subgraph")
                if self._cur() != "{":
                    self._consume()  # subgraph name
                self._consume("{")
                while self._cur() != "}":
                    self._parse_statement(graph, default_node, default_edge)
                self._consume("}")

            else:
                self._parse_statement(graph, default_node, default_edge)

        self._consume("}")
        return graph

    def _parse_statement(self, graph: Graph, default_node: dict[str, str], default_edge: dict[str, str]) -> None:
        node_id = self._consume()
        if node_id is None:
            return
        self._consume_optional(";")

        if self._cur() == "->":
            self._parse_edge_chain(graph, node_id, default_edge)
        else:
            attrs: dict[str, str] = {}
            if self._cur() == "[":
                attrs = self._parse_attr_block()
            merged = {**default_node, **attrs}
            node = Node(id=node_id, **self._normalize_node(merged))
            graph.nodes[node_id] = node
            self._consume_optional(";")

    def _parse_edge_chain(self, graph: Graph, from_node: str, default_edge: dict[str, str]) -> None:
        chain = [from_node]
        while self._cur() == "->":
            self._consume("->")
            chain.append(self._consume())
        attrs: dict[str, str] = {}
        if self._cur() == "[":
            attrs = self._parse_attr_block()
        merged = {**default_edge, **attrs}
        for i in range(len(chain) - 1):
            edge = Edge(from_node=chain[i], to_node=chain[i + 1], **self._normalize_edge(merged))
            graph.edges.append(edge)
        self._consume_optional(";")

    def _parse_attr_block(self) -> dict[str, str]:
        self._consume("[")
        attrs: dict[str, str] = {}
        while self._cur() != "]":
            key = self._consume()
            self._consume("=")
            value = self._parse_value()
            attrs[key] = value
            if self._cur() == ",":
                self._consume(",")
        self._consume("]")
        return attrs

    def _parse_value(self) -> str:
        token = self._cur()
        if token is None:
            raise SyntaxError("Unexpected end of input")
        self._consume()
        if token.startswith('"') and token.endswith('"'):
            return token[1:-1].replace('\\"', '"')
        return token

    # -- Normalization --------------------------------------------------------

    @staticmethod
    def _normalize_node(attrs: dict[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        mapping = {
            "label": "label", "shape": "shape", "type": "type", "prompt": "prompt",
            "fidelity": "fidelity", "thread_id": "thread_id", "timeout": "timeout",
            "llm_model": "llm_model", "llm_provider": "llm_provider", "reasoning_effort": "reasoning_effort",
            "retry_target": "retry_target", "fallback_retry_target": "fallback_retry_target",
        }
        for k, v in attrs.items():
            if k in mapping:
                result[mapping[k]] = v
            elif k == "max_retries":
                result["max_retries"] = int(v)
            elif k == "goal_gate":
                result["goal_gate"] = v.lower() == "true"
            elif k == "auto_status":
                result["auto_status"] = v.lower() == "true"
            elif k == "allow_partial":
                result["allow_partial"] = v.lower() == "true"
            elif k == "class":
                result["classes"] = [c.strip() for c in v.split(",")]
            else:
                extra[k] = v
        if extra:
            result["attributes"] = extra
        return result

    @staticmethod
    def _normalize_edge(attrs: dict[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in attrs.items():
            if k == "label":
                result["label"] = v
            elif k == "condition":
                result["condition"] = v
            elif k == "weight":
                result["weight"] = int(v)
            elif k == "fidelity":
                result["fidelity"] = v
            elif k == "thread_id":
                result["thread_id"] = v
            elif k == "loop_restart":
                result["loop_restart"] = v.lower() == "true"
        return result
