"""Attractor CLI – run DOT pipelines or start an interactive coding agent session."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from attractor.pipeline.parser import DotParser
from attractor.pipeline.validation import Validator, Severity
from attractor.pipeline.handlers import create_default_handler_registry
from attractor.pipeline.engine import PipelineEngine
from attractor.pipeline.graph import Context, Node, Outcome, StageStatus
from attractor.pipeline.interviewer import AutoApproveInterviewer, ConsoleInterviewer


def _run_pipeline(args: argparse.Namespace) -> int:
    dot_source = Path(args.file).read_text()
    parser = DotParser()
    graph = parser.parse(dot_source)

    # Validate
    validator = Validator()
    diags = validator.validate(graph)
    for d in diags:
        prefix = {"error": "ERROR", "warning": "WARN", "info": "INFO"}.get(d.severity.value, "INFO")
        loc = f" ({d.node_id})" if d.node_id else ""
        print(f"  [{prefix}] {d.rule}{loc}: {d.message}")

    errors = [d for d in diags if d.severity == Severity.ERROR]
    if errors:
        print(f"\nValidation failed with {len(errors)} error(s). Aborting.")
        return 1

    # Build handler registry with optional LLM backend
    interviewer = ConsoleInterviewer() if args.interactive else AutoApproveInterviewer()
    llm_backend = None
    if not args.simulate:
        try:
            from attractor.llm.client import Client
            from attractor.pipeline.llm_backend import LLMBackend
            client = Client.from_env()
            llm_backend = LLMBackend(
                client=client,
                model=args.model or "claude-sonnet-4-5-20250929",
            )
            print("  Using real LLM backend")
        except Exception as exc:
            print(f"  No LLM backend available ({exc}), using simulated mode")
    else:
        print("  Using simulated mode (--simulate)")
    registry = create_default_handler_registry(interviewer, llm_backend=llm_backend)

    # Run
    logs_root = args.logs or "./logs"

    def _print_stage_output(node: Node, outcome: Outcome, context: Context) -> None:
        if node.shape in ("Mdiamond", "Msquare"):
            return
        label = node.label or node.id
        status = outcome.status.value
        print(f"\n  [{label}] {status}")
        response = context.get(f"stage.{node.id}.response")
        if response and len(str(response)) > 20:
            print(f"  {response}")

    engine = PipelineEngine(registry, on_stage_complete=_print_stage_output)
    print(f"\nRunning pipeline '{graph.name}' ...")
    try:
        outcome = engine.run(graph, logs_root=logs_root)
        print(f"\nPipeline finished: {outcome.status.value}")
        if outcome.notes:
            print(f"  {outcome.notes}")
        return 0
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1


def _run_agent(args: argparse.Namespace) -> int:
    from attractor.agent.session import Session, SessionConfig
    from attractor.agent.loop import process_input
    from attractor.llm.client import Client

    config = SessionConfig()
    if args.max_turns:
        config.max_turns = args.max_turns

    try:
        llm_client = Client.from_env()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment.", file=sys.stderr)
        return 1

    session = Session(
        model=args.model or "claude-sonnet-4-5-20250929",
        provider=args.provider or "anthropic",
        llm_client=llm_client,
        config=config,
    )

    # Print events to console
    from attractor.agent.events import EventKind
    def _on_event(event):
        if event.kind == EventKind.ASSISTANT_TEXT_END:
            text = event.data.get("text", "")
            if text:
                print(f"\n{text}")
        elif event.kind == EventKind.TOOL_CALL_START:
            print(f"  -> {event.data.get('tool_name', '?')} ...", end="", flush=True)
        elif event.kind == EventKind.TOOL_CALL_END:
            print(" done.")

    session.events.on(_on_event)

    if args.prompt:
        asyncio.run(process_input(session, args.prompt))
    else:
        # Interactive REPL — use a single event loop to avoid
        # RuntimeError("Event loop is closed") from reused httpx clients.
        async def _repl() -> None:
            print("Attractor agent (type 'exit' to quit)")
            while True:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("\n> ").strip()
                    )
                except (EOFError, KeyboardInterrupt):
                    break
                if user_input.lower() in ("exit", "quit"):
                    break
                if not user_input:
                    continue
                await process_input(session, user_input)

        asyncio.run(_repl())

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="attractor", description="Deterministic execution harness for non-interactive agent runs")
    sub = parser.add_subparsers(dest="command")

    # Pipeline subcommand
    pipe = sub.add_parser("pipeline", aliases=["pipe", "run"], help="Run a DOT pipeline")
    pipe.add_argument("file", help="Path to .dot pipeline file")
    pipe.add_argument("--logs", default="./logs", help="Log output directory")
    pipe.add_argument("--interactive", "-i", action="store_true", help="Enable interactive human gates")
    pipe.add_argument("--simulate", "-s", action="store_true", help="Use simulated LLM (no API key needed)")
    pipe.add_argument("--model", "-m", help="Model ID for LLM backend")
    pipe.set_defaults(func=_run_pipeline)

    # Validate subcommand
    val = sub.add_parser("validate", aliases=["lint"], help="Validate a DOT pipeline without running it")
    val.add_argument("file", help="Path to .dot pipeline file")
    val.set_defaults(func=lambda a: _validate_only(a))

    # Agent subcommand
    agent = sub.add_parser("agent", help="Start a coding agent session")
    agent.add_argument("--prompt", "-p", help="Run a single prompt non-interactively")
    agent.add_argument("--model", "-m", help="Model ID (default: claude-sonnet-4-5-20250929)")
    agent.add_argument("--provider", help="Provider: anthropic or openai")
    agent.add_argument("--max-turns", type=int, default=0, help="Max turns (0 = unlimited)")
    agent.set_defaults(func=_run_agent)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    sys.exit(args.func(args))


def _validate_only(args: argparse.Namespace) -> int:
    dot_source = Path(args.file).read_text()
    graph = DotParser().parse(dot_source)
    diags = Validator().validate(graph)

    for d in diags:
        prefix = {"error": "ERROR", "warning": "WARN", "info": "INFO"}.get(d.severity.value, "INFO")
        loc = f" ({d.node_id})" if d.node_id else ""
        print(f"[{prefix}] {d.rule}{loc}: {d.message}")

    errors = [d for d in diags if d.severity == Severity.ERROR]
    if errors:
        print(f"\n{len(errors)} error(s) found.")
        return 1
    print(f"\nValid. {len(diags)} diagnostic(s).")
    return 0


if __name__ == "__main__":
    main()
