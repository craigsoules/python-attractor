"""Tool output truncation strategies."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TOOL_CHAR_LIMITS: dict[str, int] = {
    "read_file": 50_000,
    "shell": 30_000,
    "grep": 20_000,
    "glob": 20_000,
    "edit_file": 10_000,
    "write_file": 1_000,
    "spawn_agent": 20_000,
}

DEFAULT_TRUNCATION_MODES: dict[str, str] = {
    "read_file": "head_tail",
    "shell": "head_tail",
    "grep": "tail",
    "glob": "tail",
    "edit_file": "tail",
    "write_file": "tail",
}

DEFAULT_LINE_LIMITS: dict[str, int] = {
    "shell": 256,
    "grep": 200,
    "glob": 500,
}


def truncate_by_chars(output: str, max_chars: int, mode: str = "head_tail") -> str:
    if len(output) <= max_chars:
        return output
    removed = len(output) - max_chars
    if mode == "head_tail":
        half = max_chars // 2
        return (
            output[:half]
            + f"\n\n[WARNING: Output truncated. {removed} characters removed from middle.]\n\n"
            + output[-half:]
        )
    # tail
    return (
        f"[WARNING: Output truncated. First {removed} characters removed.]\n\n"
        + output[-max_chars:]
    )


def truncate_by_lines(output: str, max_lines: int) -> str:
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output
    head_count = max_lines // 2
    tail_count = max_lines - head_count
    omitted = len(lines) - head_count - tail_count
    return (
        "\n".join(lines[:head_count])
        + f"\n[... {omitted} lines omitted ...]\n"
        + "\n".join(lines[-tail_count:])
    )


def truncate_tool_output(output: str, tool_name: str, overrides: dict[str, int] | None = None) -> str:
    """Character truncation first, then line-based truncation."""
    limits = dict(DEFAULT_TOOL_CHAR_LIMITS)
    if overrides:
        limits.update(overrides)

    max_chars = limits.get(tool_name, 10_000)
    mode = DEFAULT_TRUNCATION_MODES.get(tool_name, "tail")
    result = truncate_by_chars(output, max_chars, mode)

    max_lines = DEFAULT_LINE_LIMITS.get(tool_name)
    if max_lines:
        result = truncate_by_lines(result, max_lines)

    return result
