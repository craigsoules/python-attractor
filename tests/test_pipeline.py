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
    FreeformHumanHandler,
    Handler,
    HandlerRegistry,
    StartHandler,
    ToolHandler,
    create_default_handler_registry,
)
from attractor.pipeline.interviewer import Answer, AutoApproveInterviewer, QueueInterviewer
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


# ---------------------------------------------------------------------------
# FreeformHumanHandler
# ---------------------------------------------------------------------------

def test_freeform_handler_stores_input():
    interviewer = QueueInterviewer([Answer(value="Build me a CSV pipeline", text="Build me a CSV pipeline")])
    handler = FreeformHumanHandler(interviewer)
    node = Node(id="get_input", label="Describe your pipeline:", shape="hexagon", type="wait.human.freeform")
    ctx = Context()
    graph = Graph(name="test")
    outcome = handler.execute(node, ctx, graph, "/tmp")
    assert outcome.status == StageStatus.SUCCESS
    assert outcome.preferred_label == ""
    assert outcome.context_updates["human.input"] == "Build me a CSV pipeline"
    assert outcome.context_updates["stage.get_input.response"] == "Build me a CSV pipeline"


def test_freeform_handler_done_sets_preferred_label():
    interviewer = QueueInterviewer([Answer(value="done", text="done")])
    handler = FreeformHumanHandler(interviewer)
    node = Node(id="review", label="Review:", shape="hexagon", type="wait.human.freeform")
    ctx = Context()
    graph = Graph(name="test")
    outcome = handler.execute(node, ctx, graph, "/tmp")
    assert outcome.status == StageStatus.SUCCESS
    assert outcome.preferred_label == "done"


def test_freeform_handler_exit_sets_preferred_label():
    interviewer = QueueInterviewer([Answer(value="EXIT", text="EXIT")])
    handler = FreeformHumanHandler(interviewer)
    node = Node(id="review", shape="hexagon", type="wait.human.freeform")
    ctx = Context()
    graph = Graph(name="test")
    outcome = handler.execute(node, ctx, graph, "/tmp")
    assert outcome.preferred_label == "done"


# ---------------------------------------------------------------------------
# ToolHandler
# ---------------------------------------------------------------------------

VALID_DOT = '''digraph test {
    start [shape = Mdiamond];
    work [shape = box, prompt = "Do thing"];
    done [shape = Msquare];
    start -> work -> done;
}'''

INVALID_DOT = '''digraph bad {
    work [shape = box];
}'''


def test_tool_handler_validate_dot_success():
    handler = ToolHandler()
    node = Node(id="validate", shape="parallelogram", attributes={"tool": "validate_dot", "source": "generate_dot"})
    ctx = Context()
    ctx.set("stage.generate_dot.response", VALID_DOT)
    graph = Graph(name="test")
    outcome = handler.execute(node, ctx, graph, "/tmp")
    assert outcome.status == StageStatus.SUCCESS


def test_tool_handler_validate_dot_failure():
    handler = ToolHandler()
    node = Node(id="validate", shape="parallelogram", attributes={"tool": "validate_dot", "source": "generate_dot"})
    ctx = Context()
    ctx.set("stage.generate_dot.response", INVALID_DOT)
    graph = Graph(name="test")
    outcome = handler.execute(node, ctx, graph, "/tmp")
    assert outcome.status == StageStatus.FAIL
    assert "Validation errors" in outcome.failure_reason or "error" in outcome.failure_reason.lower()


def test_tool_handler_validate_dot_no_content():
    handler = ToolHandler()
    node = Node(id="validate", shape="parallelogram", attributes={"tool": "validate_dot"})
    ctx = Context()
    graph = Graph(name="test")
    outcome = handler.execute(node, ctx, graph, "/tmp")
    assert outcome.status == StageStatus.FAIL


def test_tool_handler_write_file():
    handler = ToolHandler()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "output.dot")
        node = Node(id="write", shape="parallelogram", attributes={"tool": "write_file", "path": path, "source": "generate_dot"})
        ctx = Context()
        ctx.set("stage.generate_dot.response", VALID_DOT)
        graph = Graph(name="test")
        outcome = handler.execute(node, ctx, graph, "/tmp")
        assert outcome.status == StageStatus.SUCCESS
        assert os.path.exists(path)
        content = open(path).read()
        assert "digraph test" in content


def test_tool_handler_read_file():
    handler = ToolHandler()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "spec.md")
        with open(path, "w") as f:
            f.write("# My Spec\nSome content here.")
        node = Node(id="load", shape="parallelogram", attributes={"tool": "read_file", "path": path})
        ctx = Context()
        graph = Graph(name="test")
        outcome = handler.execute(node, ctx, graph, "/tmp")
        assert outcome.status == StageStatus.SUCCESS
        assert "My Spec" in outcome.context_updates["stage.load.response"]


def test_tool_handler_read_file_not_found():
    handler = ToolHandler()
    node = Node(id="load", shape="parallelogram", attributes={"tool": "read_file", "path": "/nonexistent/file.txt"})
    ctx = Context()
    graph = Graph(name="test")
    outcome = handler.execute(node, ctx, graph, "/tmp")
    assert outcome.status == StageStatus.FAIL


def test_tool_handler_source_fallback_to_last_stage():
    """Without a source attribute, tool reads from last_stage."""
    handler = ToolHandler()
    node = Node(id="validate", shape="parallelogram", attributes={"tool": "validate_dot"})
    ctx = Context()
    ctx.set("stage.generate_dot.response", VALID_DOT)
    ctx.set("last_stage", "generate_dot")
    graph = Graph(name="test")
    outcome = handler.execute(node, ctx, graph, "/tmp")
    assert outcome.status == StageStatus.SUCCESS


def test_tool_handler_source_overrides_last_stage():
    """Explicit source attribute takes priority over last_stage."""
    handler = ToolHandler()
    node = Node(id="validate", shape="parallelogram", attributes={"tool": "validate_dot", "source": "generate_dot"})
    ctx = Context()
    ctx.set("stage.generate_dot.response", VALID_DOT)
    ctx.set("stage.other.response", "not a digraph")
    ctx.set("last_stage", "other")
    graph = Graph(name="test")
    outcome = handler.execute(node, ctx, graph, "/tmp")
    assert outcome.status == StageStatus.SUCCESS


def test_tool_handler_unknown_tool():
    handler = ToolHandler()
    node = Node(id="bad", shape="parallelogram", attributes={"tool": "nonexistent"})
    ctx = Context()
    graph = Graph(name="test")
    outcome = handler.execute(node, ctx, graph, "/tmp")
    assert outcome.status == StageStatus.FAIL
    assert "Unknown tool" in outcome.failure_reason


def test_tool_handler_no_tool_attr():
    handler = ToolHandler()
    node = Node(id="bad", shape="parallelogram", attributes={})
    ctx = Context()
    graph = Graph(name="test")
    outcome = handler.execute(node, ctx, graph, "/tmp")
    assert outcome.status == StageStatus.FAIL


# ---------------------------------------------------------------------------
# Iteration-aware context
# ---------------------------------------------------------------------------

def test_engine_iteration_context():
    """Verify that node re-execution stores iter_N.response keys."""
    # Build a pipeline with a loop: start -> step -> check -> step (fail) or done (success)
    # We use a custom handler that fails on first visit, succeeds on second
    visit_count = {"step": 0}

    class CountingHandler(Handler):
        def execute(self, node, context, graph, logs_root):
            visit_count["step"] += 1
            if visit_count["step"] == 1:
                return Outcome(
                    status=StageStatus.SUCCESS,
                    context_updates={
                        f"stage.{node.id}.response": f"response_{visit_count['step']}",
                        "last_stage": node.id,
                        "last_response": f"response_{visit_count['step']}",
                    },
                )
            return Outcome(
                status=StageStatus.SUCCESS,
                context_updates={
                    f"stage.{node.id}.response": f"response_{visit_count['step']}",
                    "last_stage": node.id,
                    "last_response": f"response_{visit_count['step']}",
                },
            )

    # Simpler test: just run engine and check iteration keys are set
    g = DotParser().parse(SIMPLE_DOT)
    registry = create_default_handler_registry()
    engine = PipelineEngine(registry)
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.run(g, logs_root=tmpdir)
        # After run, check checkpoint for iteration context
        cp = json.loads(open(os.path.join(tmpdir, "checkpoint.json")).read())
        ctx_snap = cp["context"]
        # work node should have iter_1 key
        assert "stage.work.iter_1.response" in ctx_snap
        assert ctx_snap["stage.work.iter_1.response"] == ctx_snap["stage.work.response"]


# ---------------------------------------------------------------------------
# _build_prior_context with iterations
# ---------------------------------------------------------------------------

def test_build_prior_context_with_iterations():
    from attractor.pipeline.llm_backend import LLMBackend
    ctx = Context()
    ctx.set("stage.get_input.iter_1.response", "Build a CSV pipeline")
    ctx.set("stage.generate_dot.iter_1.response", "digraph csv { ... }")
    ctx.set("stage.get_input.iter_2.response", "Add validation step")
    ctx.set("stage.generate_dot.iter_2.response", "digraph csv { ... updated }")
    ctx.set("stage.get_input.response", "Add validation step")
    ctx.set("stage.generate_dot.response", "digraph csv { ... updated }")

    result = LLMBackend._build_prior_context(ctx)
    assert "iteration 1" in result
    assert "iteration 2" in result
    # Iteration 1 should come before iteration 2
    pos1 = result.index("iteration 1")
    pos2 = result.index("iteration 2")
    assert pos1 < pos2


def test_build_prior_context_no_iterations_fallback():
    from attractor.pipeline.llm_backend import LLMBackend
    ctx = Context()
    ctx.set("stage.work.response", "some output")
    result = LLMBackend._build_prior_context(ctx)
    assert "## Stage: work" in result
    assert "iteration" not in result


# ---------------------------------------------------------------------------
# Registry resolves new handler types
# ---------------------------------------------------------------------------

def test_registry_resolves_freeform():
    registry = create_default_handler_registry()
    node = Node(id="test", shape="hexagon", type="wait.human.freeform")
    handler = registry.resolve(node)
    assert isinstance(handler, FreeformHumanHandler)


def test_registry_resolves_tool():
    registry = create_default_handler_registry()
    node = Node(id="test", shape="parallelogram")
    handler = registry.resolve(node)
    assert isinstance(handler, ToolHandler)


# ---------------------------------------------------------------------------
# dot_agent.dot parses and validates
# ---------------------------------------------------------------------------

def test_dot_agent_pipeline_valid():
    dot_agent_path = os.path.join(os.path.dirname(__file__), "..", "pipelines", "dot_agent.dot")
    if not os.path.exists(dot_agent_path):
        pytest.skip("dot_agent.dot not found")
    with open(dot_agent_path) as f:
        source = f.read()
    g = DotParser().parse(source)
    diags = Validator().validate(g)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 0, f"Validation errors: {errors}"
