"""Tests for the pipeline engine: parser, validator, conditions, engine."""

import json
import os
import tempfile

import pytest

from attractor.pipeline.conditions import ConditionParser
from attractor.pipeline.engine import PipelineEngine
from attractor.pipeline.graph import Context, Edge, Graph, Node, Outcome, StageStatus
from attractor.pipeline.handlers import (
    CodergenHandler,
    ConditionalHandler,
    ExitHandler,
    HandlerRegistry,
    StartHandler,
    create_default_handler_registry,
)
from attractor.pipeline.interviewer import AutoApproveInterviewer
from attractor.pipeline.parser import DotParser
from attractor.pipeline.stylesheet import StylesheetApplier, StylesheetParser
from attractor.pipeline.validation import Diagnostic, Severity, Validator


# ---------------------------------------------------------------------------
# DOT parser
# ---------------------------------------------------------------------------

SIMPLE_DOT = '''\
digraph test {
    graph [goal = "Test goal"];
    start [shape = Mdiamond, label = "Begin"];
    work [shape = box, label = "Do work", prompt = "Do the thing"];
    done [shape = Msquare, label = "End"];
    start -> work;
    work -> done;
}
'''


def test_parser_simple():
    g = DotParser().parse(SIMPLE_DOT)
    assert g.name == "test"
    assert g.goal == "Test goal"
    assert "start" in g.nodes
    assert "work" in g.nodes
    assert "done" in g.nodes
    assert g.nodes["start"].shape == "Mdiamond"
    assert g.nodes["done"].shape == "Msquare"
    assert g.nodes["work"].prompt == "Do the thing"
    assert len(g.edges) == 2


def test_parser_edge_chain():
    dot = 'digraph chain { a [shape=Mdiamond]; b [shape=box]; c [shape=box]; d [shape=Msquare]; a -> b -> c -> d; }'
    g = DotParser().parse(dot)
    assert len(g.edges) == 3
    assert g.edges[0].from_node == "a"
    assert g.edges[0].to_node == "b"
    assert g.edges[2].to_node == "d"


def test_parser_edge_attributes():
    dot = '''digraph cond {
        a [shape=Mdiamond]; b [shape=box]; c [shape=Msquare];
        a -> b [condition = "outcome=success", weight = 10];
        a -> c [label = "fallback"];
    }'''
    g = DotParser().parse(dot)
    assert g.edges[0].condition == "outcome=success"
    assert g.edges[0].weight == 10
    assert g.edges[1].label == "fallback"


def test_parser_with_comments():
    dot = '''// A comment
    digraph commented {
        /* block comment */
        start [shape=Mdiamond];
        end [shape=Msquare];
        start -> end;
    }'''
    g = DotParser().parse(dot)
    assert len(g.nodes) == 2


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def test_validator_valid():
    g = DotParser().parse(SIMPLE_DOT)
    diags = Validator().validate(g)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 0


def test_validator_no_start():
    dot = 'digraph bad { a [shape=box]; b [shape=Msquare]; a -> b; }'
    g = DotParser().parse(dot)
    diags = Validator().validate(g)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert any("start" in d.message.lower() for d in errors)


def test_validator_no_exit():
    dot = 'digraph bad { a [shape=Mdiamond]; b [shape=box]; a -> b; }'
    g = DotParser().parse(dot)
    diags = Validator().validate(g)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert any("exit" in d.message.lower() for d in errors)


def test_validator_unreachable():
    dot = '''digraph unreach {
        start [shape=Mdiamond]; end [shape=Msquare]; orphan [shape=box];
        start -> end;
    }'''
    g = DotParser().parse(dot)
    diags = Validator().validate(g)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert any("unreachable" in d.message.lower() for d in errors)


def test_validate_or_raise():
    dot = 'digraph bad { a [shape=box]; }'
    g = DotParser().parse(dot)
    with pytest.raises(ValueError, match="Validation failed"):
        Validator().validate_or_raise(g)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def test_condition_equality():
    parser = ConditionParser()
    ev = parser.parse("outcome=success")
    outcome = Outcome(status=StageStatus.SUCCESS)
    ctx = Context()
    assert ev(outcome, ctx) is True


def test_condition_inequality():
    ev = ConditionParser().parse("outcome!=fail")
    assert ev(Outcome(status=StageStatus.SUCCESS), Context()) is True
    assert ev(Outcome(status=StageStatus.FAIL), Context()) is False


def test_condition_and():
    ev = ConditionParser().parse("outcome=success && context.ready=true")
    ctx = Context()
    ctx.set("ready", "true")
    assert ev(Outcome(status=StageStatus.SUCCESS), ctx) is True


def test_condition_context_lookup():
    ev = ConditionParser().parse("context.mode=fast")
    ctx = Context()
    ctx.set("mode", "fast")
    assert ev(Outcome(status=StageStatus.SUCCESS), ctx) is True


def test_condition_empty():
    ev = ConditionParser().parse("")
    assert ev(Outcome(status=StageStatus.FAIL), Context()) is True


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

def test_stylesheet_parse():
    css = '* { llm_model: gpt-4o; reasoning_effort: medium; } #special { llm_model: claude-opus-4-6; }'
    rules = StylesheetParser().parse(css)
    assert len(rules) == 2
    assert rules[0]["selector"] == "*"
    assert rules[1]["selector"] == "#special"


def test_stylesheet_apply():
    g = DotParser().parse(SIMPLE_DOT)
    g.model_stylesheet = '* { llm_model: gpt-4o; } #work { llm_model: claude-opus-4-6; }'
    StylesheetApplier().apply(g)
    # #work has higher specificity, so it wins
    assert g.nodes["work"].llm_model == "claude-opus-4-6"
    # start gets the wildcard
    assert g.nodes["start"].llm_model == "gpt-4o"


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def test_engine_simple_pipeline():
    g = DotParser().parse(SIMPLE_DOT)
    registry = create_default_handler_registry()
    engine = PipelineEngine(registry)
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = engine.run(g, logs_root=tmpdir)
    assert outcome.status == StageStatus.SUCCESS


def test_engine_conditional():
    dot = '''digraph cond {
        start [shape=Mdiamond];
        step [shape=box, prompt="Do thing"];
        check [shape=diamond];
        done [shape=Msquare];
        start -> step;
        step -> check;
        check -> done [condition="outcome=success"];
    }'''
    g = DotParser().parse(dot)
    registry = create_default_handler_registry()
    engine = PipelineEngine(registry)
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = engine.run(g, logs_root=tmpdir)
    assert outcome.status == StageStatus.SUCCESS


def test_engine_checkpoint_written():
    g = DotParser().parse(SIMPLE_DOT)
    registry = create_default_handler_registry()
    engine = PipelineEngine(registry)
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.run(g, logs_root=tmpdir)
        cp_path = os.path.join(tmpdir, "checkpoint.json")
        assert os.path.exists(cp_path)
        cp = json.loads(open(cp_path).read())
        assert "completed_nodes" in cp
        assert "work" in cp["completed_nodes"]


def test_engine_logs_written():
    g = DotParser().parse(SIMPLE_DOT)
    registry = create_default_handler_registry()
    engine = PipelineEngine(registry)
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.run(g, logs_root=tmpdir)
        prompt_file = os.path.join(tmpdir, "work", "prompt.md")
        assert os.path.exists(prompt_file)
        content = open(prompt_file).read()
        assert "Do the thing" in content


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def test_context_basics():
    ctx = Context()
    ctx.set("key", "value")
    assert ctx.get("key") == "value"
    assert ctx.get("missing", "default") == "default"
    assert ctx.get_string("key") == "value"


def test_context_clone():
    ctx = Context()
    ctx.set("a", 1)
    cloned = ctx.clone()
    cloned.set("a", 2)
    assert ctx.get("a") == 1
    assert cloned.get("a") == 2


def test_context_snapshot():
    ctx = Context()
    ctx.set("x", 10)
    ctx.set("y", 20)
    snap = ctx.snapshot()
    assert snap == {"x": 10, "y": 20}
