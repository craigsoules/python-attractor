"""Tool registry and built-in tool definitions."""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

from attractor.agent.environment import ExecutionEnvironment


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict


@dataclass
class RegisteredTool:
    definition: ToolDefinition
    executor: Callable[[dict, ExecutionEnvironment], str]


class ToolRegistry:
    """Registry mapping tool names to handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.definition.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# Built-in tool executors
# ---------------------------------------------------------------------------

def read_file_executor(args: dict, env: ExecutionEnvironment) -> str:
    path = args["file_path"]
    offset = args.get("offset", 1)
    limit = args.get("limit", 2000)
    content = env.read_file(path, offset=offset, limit=limit)
    lines = content.split("\n")
    numbered = [f"{i + offset:4d} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


def write_file_executor(args: dict, env: ExecutionEnvironment) -> str:
    path = args["file_path"]
    content = args["content"]
    env.write_file(path, content)
    return f"Wrote {len(content)} bytes to {path}"


def edit_file_executor(args: dict, env: ExecutionEnvironment) -> str:
    path = args["file_path"]
    old_string = args["old_string"]
    new_string = args["new_string"]
    replace_all = args.get("replace_all", False)

    content = env.read_file(path)
    if old_string not in content:
        raise ValueError(f"old_string not found in {path}")

    if replace_all:
        count = content.count(old_string)
        new_content = content.replace(old_string, new_string)
    else:
        count = content.count(old_string)
        if count > 1:
            raise ValueError(f"old_string appears {count} times in {path}, provide more context or use replace_all")
        new_content = content.replace(old_string, new_string, 1)
        count = 1

    env.write_file(path, new_content)
    return f"Replaced {count} occurrence(s) in {path}"


def shell_executor(args: dict, env: ExecutionEnvironment) -> str:
    command = args["command"]
    timeout_ms = args.get("timeout_ms", 10_000)
    result = env.exec_command(command=command, timeout_ms=timeout_ms)

    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.timed_out:
        output += f"\n[ERROR: Command timed out after {timeout_ms}ms. Partial output shown above.]"
    return f"Exit code: {result.exit_code}\n{output}"


def grep_executor(args: dict, env: ExecutionEnvironment) -> str:
    pattern = args["pattern"]
    path = args.get("path", env.working_directory())
    options = {
        "glob_filter": args.get("glob_filter"),
        "case_insensitive": args.get("case_insensitive", False),
        "max_results": args.get("max_results", 100),
    }
    return env.grep(pattern, path, options)


def glob_executor(args: dict, env: ExecutionEnvironment) -> str:
    pattern = args["pattern"]
    path = args.get("path", env.working_directory())
    matches = env.glob(pattern, path)
    return "\n".join(matches)


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------

def create_default_registry() -> ToolRegistry:
    """Create a registry populated with all built-in tools."""
    registry = ToolRegistry()

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="read_file",
            description="Read a file from disk with line numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "offset": {"type": "integer", "description": "Line number to start from (1-based)", "default": 1},
                    "limit": {"type": "integer", "description": "Max lines to read", "default": 2000},
                },
                "required": ["file_path"],
            },
        ),
        executor=read_file_executor,
    ))

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="write_file",
            description="Write content to a file, creating parent directories as needed.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "content": {"type": "string", "description": "File content to write"},
                },
                "required": ["file_path", "content"],
            },
        ),
        executor=write_file_executor,
    ))

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="edit_file",
            description="Search-and-replace within a file. old_string must be unique unless replace_all is true.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        ),
        executor=edit_file_executor,
    ))

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="shell",
            description="Execute a shell command and return stdout/stderr.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds", "default": 10000},
                },
                "required": ["command"],
            },
        ),
        executor=shell_executor,
    ))

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="grep",
            description="Search file contents by regex pattern.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory or file to search"},
                    "glob_filter": {"type": "string", "description": "Glob to filter files, e.g. '*.py'"},
                    "case_insensitive": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "default": 100},
                },
                "required": ["pattern"],
            },
        ),
        executor=grep_executor,
    ))

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="glob",
            description="Find files matching a glob pattern.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
                    "path": {"type": "string", "description": "Base directory"},
                },
                "required": ["pattern"],
            },
        ),
        executor=glob_executor,
    ))

    return registry
