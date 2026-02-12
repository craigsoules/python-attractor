"""Tests for the dot_agent package."""

import os
import tempfile

import pytest

from attractor.agent.environment import LocalExecutionEnvironment
from attractor.agent.session import Session, SessionConfig
from dot_agent.prompt import build_system_prompt
from dot_agent.tools import (
    create_dot_agent_registry,
    create_pipeline_executor,
    preview_pipeline_executor,
    update_pipeline_executor,
    validate_pipeline_executor,
)

VALID_DOT = '''\
digraph test {
    graph [goal = "Test goal"];
    start [shape = Mdiamond, label = "Begin"];
    work [shape = box, label = "Do work", prompt = "Do the thing"];
    done [shape = Msquare, label = "End"];
    start -> work;
    work -> done;
}
'''

INVALID_DOT_NO_START = '''\
digraph bad {
    work [shape = box, label = "Work"];
    done [shape = Msquare];
    work -> done;
}
'''

INVALID_DOT_SYNTAX = 'this is not valid dot'


# ---------------------------------------------------------------------------
# validate_pipeline
# ---------------------------------------------------------------------------

def test_validate_valid_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("test.dot", VALID_DOT)
        result = validate_pipeline_executor({"file_path": "test.dot"}, env)
        assert "Valid" in result
        assert "0 diagnostics" in result


def test_validate_invalid_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("bad.dot", INVALID_DOT_NO_START)
        result = validate_pipeline_executor({"file_path": "bad.dot"}, env)
        assert "ERROR" in result
        assert "error(s)" in result


def test_validate_parse_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("broken.dot", INVALID_DOT_SYNTAX)
        result = validate_pipeline_executor({"file_path": "broken.dot"}, env)
        assert "Parse error" in result


# ---------------------------------------------------------------------------
# preview_pipeline
# ---------------------------------------------------------------------------

def test_preview_valid_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("test.dot", VALID_DOT)
        result = preview_pipeline_executor({"file_path": "test.dot"}, env)
        assert "Pipeline: test" in result
        assert "Goal: Test goal" in result
        assert "Nodes (3)" in result
        assert "Edges (2)" in result
        assert "start" in result
        assert "work" in result
        assert "done" in result
        assert "Validation: OK" in result


def test_preview_invalid_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("bad.dot", INVALID_DOT_NO_START)
        result = preview_pipeline_executor({"file_path": "bad.dot"}, env)
        assert "Diagnostics:" in result
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# create_pipeline
# ---------------------------------------------------------------------------

def test_create_valid_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        path = os.path.join(tmpdir, "new.dot")
        result = create_pipeline_executor({"file_path": path, "content": VALID_DOT}, env)
        assert "written to" in result
        assert os.path.exists(path)
        # Verify contents
        written = env.read_file(path)
        assert "digraph test" in written


def test_create_invalid_pipeline_not_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        path = os.path.join(tmpdir, "bad.dot")
        result = create_pipeline_executor({"file_path": path, "content": INVALID_DOT_NO_START}, env)
        assert "Refused to write" in result
        assert not os.path.exists(path)


def test_create_parse_error_not_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        path = os.path.join(tmpdir, "broken.dot")
        result = create_pipeline_executor({"file_path": path, "content": INVALID_DOT_SYNTAX}, env)
        assert "Refused to write" in result
        assert "parse error" in result.lower()
        assert not os.path.exists(path)


def test_create_with_warnings():
    dot_with_warning = '''\
digraph warn {
    start [shape = Mdiamond];
    work [shape = box];
    done [shape = Msquare];
    start -> work;
    work -> done;
}
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        path = os.path.join(tmpdir, "warn.dot")
        result = create_pipeline_executor({"file_path": path, "content": dot_with_warning}, env)
        # Should write (warnings are not blocking)
        assert "written to" in result
        assert os.path.exists(path)
        # Should report warnings
        assert "WARN" in result


# ---------------------------------------------------------------------------
# update_pipeline
# ---------------------------------------------------------------------------

def test_update_valid_edit():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("test.dot", VALID_DOT)
        result = update_pipeline_executor({
            "file_path": "test.dot",
            "old_string": 'label = "Do work"',
            "new_string": 'label = "Do better work"',
        }, env)
        assert "updated" in result
        content = env.read_file("test.dot")
        assert "Do better work" in content


def test_update_old_string_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("test.dot", VALID_DOT)
        result = update_pipeline_executor({
            "file_path": "test.dot",
            "old_string": "nonexistent string",
            "new_string": "replacement",
        }, env)
        assert "not found" in result


def test_update_would_break_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(tmpdir)
        env.write_file("test.dot", VALID_DOT)
        # Remove the start node shape — will break validation
        result = update_pipeline_executor({
            "file_path": "test.dot",
            "old_string": "shape = Mdiamond",
            "new_string": "shape = box",
        }, env)
        assert "Refused to write" in result
        # Original file should be unchanged
        content = env.read_file("test.dot")
        assert "Mdiamond" in content


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_dot_agent_registry():
    reg = create_dot_agent_registry()
    names = reg.names()
    # Standard tools
    assert "read_file" in names
    assert "write_file" in names
    assert "shell" in names
    # Pipeline tools
    assert "validate_pipeline" in names
    assert "preview_pipeline" in names
    assert "create_pipeline" in names
    assert "update_pipeline" in names


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def test_system_prompt_contains_spec():
    session = Session(tool_registry=create_dot_agent_registry())
    prompt = build_system_prompt(session)
    assert "pipeline architect" in prompt
    assert "DOT Pipeline Specification" in prompt
    assert "Mdiamond" in prompt
    assert "validate_pipeline" in prompt
    assert "create_pipeline" in prompt


def test_system_prompt_builder_hook():
    """Verify the session accepts and uses a custom system_prompt_builder."""
    session = Session(system_prompt_builder=build_system_prompt)
    assert session.system_prompt_builder is build_system_prompt
