"""Tests for the coding agent loop."""

import os
import tempfile

import pytest

from attractor.agent.environment import LocalExecutionEnvironment
from attractor.agent.events import EventEmitter, EventKind
from attractor.agent.session import Session, SessionConfig
from attractor.agent.tools import ToolRegistry, create_default_registry
from attractor.agent.truncation import truncate_by_chars, truncate_by_lines, truncate_tool_output


# ---------------------------------------------------------------------------
# Execution environment
# ---------------------------------------------------------------------------

def test_local_env_read_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("test.txt", "hello\nworld\n")
        content = env.read_file("test.txt")
        assert "hello" in content
        assert "world" in content


def test_local_env_read_offset_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("lines.txt", "line1\nline2\nline3\nline4\n")
        content = env.read_file("lines.txt", offset=2, limit=2)
        assert "line2" in content
        assert "line3" in content
        assert "line1" not in content


def test_local_env_file_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("exists.txt", "yes")
        assert env.file_exists("exists.txt")
        assert not env.file_exists("nope.txt")


def test_local_env_exec_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        result = env.exec_command("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout


def test_local_env_exec_timeout():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        result = env.exec_command("sleep 10", timeout_ms=500)
        assert result.timed_out is True


def test_local_env_glob():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("a.py", "pass")
        env.write_file("b.py", "pass")
        env.write_file("c.txt", "text")
        matches = env.glob("*.py", tmpdir)
        assert len(matches) == 2
        assert all(m.endswith(".py") for m in matches)


def test_local_env_grep():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("search.txt", "foo\nbar\nbaz\n")
        result = env.grep("bar", tmpdir, {"max_results": 10})
        assert "bar" in result


def test_local_env_platform():
    env = LocalExecutionEnvironment()
    assert env.platform() in ("darwin", "linux", "windows", "unknown")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def test_default_registry():
    reg = create_default_registry()
    names = reg.names()
    assert "read_file" in names
    assert "write_file" in names
    assert "edit_file" in names
    assert "shell" in names
    assert "grep" in names
    assert "glob" in names


def test_registry_register_unregister():
    reg = ToolRegistry()
    from attractor.agent.tools import RegisteredTool, ToolDefinition
    tool = RegisteredTool(
        definition=ToolDefinition(name="custom", description="Custom", parameters={"type": "object"}),
        executor=lambda args, env: "ok",
    )
    reg.register(tool)
    assert reg.get("custom") is not None
    reg.unregister("custom")
    assert reg.get("custom") is None


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def test_truncate_by_chars_no_op():
    assert truncate_by_chars("short", 1000) == "short"


def test_truncate_by_chars_head_tail():
    big = "x" * 10_000
    result = truncate_by_chars(big, 1_000, "head_tail")
    assert "truncated" in result.lower() or "WARNING" in result
    assert len(result) < len(big)


def test_truncate_by_chars_tail():
    big = "y" * 5_000
    result = truncate_by_chars(big, 1_000, "tail")
    assert "WARNING" in result
    assert result.endswith("y" * 100)


def test_truncate_by_lines_no_op():
    text = "a\nb\nc"
    assert truncate_by_lines(text, 10) == text


def test_truncate_by_lines():
    lines = "\n".join(f"line{i}" for i in range(100))
    result = truncate_by_lines(lines, 10)
    assert "omitted" in result
    assert "line0" in result
    assert "line99" in result


def test_truncate_tool_output():
    big = "z" * 100_000
    result = truncate_tool_output(big, "shell")
    assert len(result) < len(big)


# ---------------------------------------------------------------------------
# Event emitter
# ---------------------------------------------------------------------------

def test_event_emitter():
    emitter = EventEmitter()
    events = []
    emitter.on(lambda e: events.append(e))
    emitter.emit("sess-1", EventKind.SESSION_START, info="test")
    assert len(events) == 1
    assert events[0].kind == EventKind.SESSION_START
    assert events[0].session_id == "sess-1"
    assert events[0].data["info"] == "test"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def test_session_creation():
    s = Session()
    assert s.state.value == "idle"
    assert len(s.history) == 0
    assert s.tool_registry.names()
