"""Pipeline-specific tools for the DOT builder agent."""

from __future__ import annotations

from attractor.agent.environment import ExecutionEnvironment
from attractor.agent.tools import RegisteredTool, ToolDefinition, ToolRegistry, create_default_registry
from attractor.pipeline.handlers import HandlerRegistry
from attractor.pipeline.parser import DotParser
from attractor.pipeline.validation import Severity, Validator


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

def validate_pipeline_executor(args: dict, env: ExecutionEnvironment) -> str:
    """Parse and validate a .dot pipeline file."""
    path = args["file_path"]
    source = env.read_file(path)

    try:
        graph = DotParser().parse(source)
    except Exception as exc:
        return f"Parse error: {exc}"

    diags = Validator().validate(graph)
    if not diags:
        return f"Valid. Pipeline '{graph.name}' has {len(graph.nodes)} nodes and {len(graph.edges)} edges. 0 diagnostics."

    lines = []
    errors = 0
    warnings = 0
    for d in diags:
        prefix = "ERROR" if d.severity == Severity.ERROR else "WARN" if d.severity == Severity.WARNING else "INFO"
        loc = f" ({d.node_id})" if d.node_id else ""
        lines.append(f"[{prefix}] {d.rule}{loc}: {d.message}")
        if d.severity == Severity.ERROR:
            errors += 1
        elif d.severity == Severity.WARNING:
            warnings += 1

    lines.append(f"\n{errors} error(s), {warnings} warning(s).")
    return "\n".join(lines)


def preview_pipeline_executor(args: dict, env: ExecutionEnvironment) -> str:
    """Parse a .dot file and return a structured summary."""
    path = args["file_path"]
    source = env.read_file(path)

    try:
        graph = DotParser().parse(source)
    except Exception as exc:
        return f"Parse error: {exc}"

    shape_map = HandlerRegistry.SHAPE_TO_TYPE

    lines = [
        f"Pipeline: {graph.name}",
        f"Goal: {graph.goal or '(not set)'}",
        f"Label: {graph.label or '(not set)'}",
        f"Default max retries: {graph.default_max_retry}",
        "",
        f"Nodes ({len(graph.nodes)}):",
    ]

    for node in graph.nodes.values():
        handler = shape_map.get(node.shape, "unknown")
        has_prompt = "yes" if node.prompt else "no"
        extras = []
        if node.goal_gate:
            extras.append("goal_gate")
        if node.retry_target:
            extras.append(f"retry_target={node.retry_target}")
        if node.llm_model:
            extras.append(f"model={node.llm_model}")
        extra_str = f" [{', '.join(extras)}]" if extras else ""
        lines.append(f"  {node.id} (shape={node.shape}, handler={handler}, prompt={has_prompt}){extra_str}")

    lines.append("")
    lines.append(f"Edges ({len(graph.edges)}):")
    for edge in graph.edges:
        parts = [f"  {edge.from_node} -> {edge.to_node}"]
        if edge.label:
            parts.append(f'label="{edge.label}"')
        if edge.condition:
            parts.append(f'condition="{edge.condition}"')
        if edge.weight:
            parts.append(f"weight={edge.weight}")
        lines.append(" ".join(parts))

    # Validation
    diags = Validator().validate(graph)
    if diags:
        lines.append("")
        lines.append("Diagnostics:")
        for d in diags:
            prefix = "ERROR" if d.severity == Severity.ERROR else "WARN"
            loc = f" ({d.node_id})" if d.node_id else ""
            lines.append(f"  [{prefix}] {d.rule}{loc}: {d.message}")
    else:
        lines.append("\nValidation: OK")

    return "\n".join(lines)


def create_pipeline_executor(args: dict, env: ExecutionEnvironment) -> str:
    """Validate DOT content and write to file only if valid."""
    path = args["file_path"]
    content = args["content"]

    # Parse
    try:
        graph = DotParser().parse(content)
    except Exception as exc:
        return f"Refused to write — parse error: {exc}\n\nFix the DOT syntax and try again."

    # Validate
    diags = Validator().validate(graph)
    errors = [d for d in diags if d.severity == Severity.ERROR]

    if errors:
        lines = ["Refused to write — validation errors:"]
        for d in errors:
            loc = f" ({d.node_id})" if d.node_id else ""
            lines.append(f"  [ERROR] {d.rule}{loc}: {d.message}")
            if d.fix:
                lines.append(f"         Fix: {d.fix}")
        lines.append("\nFix the errors and try again.")
        return "\n".join(lines)

    # Write
    env.write_file(path, content)

    warnings = [d for d in diags if d.severity == Severity.WARNING]
    result = f"Pipeline '{graph.name}' written to {path} ({len(graph.nodes)} nodes, {len(graph.edges)} edges)."
    if warnings:
        warning_lines = []
        for d in warnings:
            loc = f" ({d.node_id})" if d.node_id else ""
            warning_lines.append(f"  [WARN] {d.rule}{loc}: {d.message}")
        result += "\n\nWarnings:\n" + "\n".join(warning_lines)
    return result


def update_pipeline_executor(args: dict, env: ExecutionEnvironment) -> str:
    """Apply a search-and-replace edit to a .dot file, validating the result before writing."""
    path = args["file_path"]
    old_string = args["old_string"]
    new_string = args["new_string"]

    # Read current content
    content = env.read_file(path)

    if old_string not in content:
        return f"old_string not found in {path}. Read the file first to see current contents."

    count = content.count(old_string)
    if count > 1:
        return f"old_string appears {count} times in {path}. Provide more context to make it unique."

    new_content = content.replace(old_string, new_string, 1)

    # Parse the result
    try:
        graph = DotParser().parse(new_content)
    except Exception as exc:
        return f"Refused to write — the edit would create a parse error: {exc}\n\nOriginal file unchanged."

    # Validate
    diags = Validator().validate(graph)
    errors = [d for d in diags if d.severity == Severity.ERROR]

    if errors:
        lines = ["Refused to write — the edit would introduce validation errors:"]
        for d in errors:
            loc = f" ({d.node_id})" if d.node_id else ""
            lines.append(f"  [ERROR] {d.rule}{loc}: {d.message}")
        lines.append("\nOriginal file unchanged. Fix the errors and try again.")
        return "\n".join(lines)

    # Write
    env.write_file(path, new_content)

    warnings = [d for d in diags if d.severity == Severity.WARNING]
    result = f"Pipeline '{graph.name}' updated in {path}."
    if warnings:
        warning_lines = []
        for d in warnings:
            loc = f" ({d.node_id})" if d.node_id else ""
            warning_lines.append(f"  [WARN] {d.rule}{loc}: {d.message}")
        result += "\n\nWarnings:\n" + "\n".join(warning_lines)
    return result


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def create_dot_agent_registry() -> ToolRegistry:
    """Create a tool registry with standard tools + pipeline-specific tools."""
    registry = create_default_registry()

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="validate_pipeline",
            description="Parse and validate a .dot pipeline file. Returns diagnostics (errors/warnings) or confirms validity.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the .dot file"},
                },
                "required": ["file_path"],
            },
        ),
        executor=validate_pipeline_executor,
    ))

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="preview_pipeline",
            description="Parse a .dot file and return a structured summary: nodes, edges, goal, handler types, and validation status.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the .dot file"},
                },
                "required": ["file_path"],
            },
        ),
        executor=preview_pipeline_executor,
    ))

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="create_pipeline",
            description="Write a new .dot pipeline file. The content is parsed and validated BEFORE writing — if there are errors, the file is NOT written and diagnostics are returned. Always use this instead of write_file for .dot files.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path for the new .dot file"},
                    "content": {"type": "string", "description": "Complete DOT digraph content"},
                },
                "required": ["file_path", "content"],
            },
        ),
        executor=create_pipeline_executor,
    ))

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="update_pipeline",
            description="Apply a search-and-replace edit to an existing .dot file. The result is parsed and validated BEFORE writing — if the edit would introduce errors, the file is NOT modified. Always use this instead of edit_file for .dot files.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the .dot file"},
                    "old_string": {"type": "string", "description": "Text to find (must be unique in the file)"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        ),
        executor=update_pipeline_executor,
    ))

    return registry
