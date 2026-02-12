"""dot-agent CLI — conversational agent for building Attractor DOT pipelines."""

from __future__ import annotations

import argparse
import asyncio
import sys

from attractor.agent.events import EventKind
from attractor.agent.loop import process_input
from attractor.agent.session import Session, SessionConfig
from attractor.llm.client import Client

from dot_agent.prompt import build_system_prompt
from dot_agent.tools import create_dot_agent_registry


def _on_event(event):
    """Print agent events to the console."""
    if event.kind == EventKind.ASSISTANT_TEXT_END:
        text = event.data.get("text", "")
        if text:
            print(f"\n{text}")
    elif event.kind == EventKind.TOOL_CALL_START:
        name = event.data.get("tool_name", "?")
        print(f"  -> {name} ...", end="", flush=True)
    elif event.kind == EventKind.TOOL_CALL_END:
        if event.data.get("error"):
            print(f" error: {event.data['error']}")
        else:
            print(" done.")
    elif event.kind == EventKind.LOOP_DETECTION:
        print(f"\n  [!] {event.data.get('message', 'Loop detected')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dot-agent",
        description="Conversational agent for building Attractor DOT pipeline files",
    )
    parser.add_argument("--prompt", "-p", help="Run a single prompt non-interactively")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-5-20250929", help="Model ID")
    parser.add_argument("--provider", default="anthropic", help="Provider: anthropic or openai")
    parser.add_argument("--max-turns", type=int, default=0, help="Max tool rounds (0 = unlimited)")

    args = parser.parse_args()

    # Initialize LLM client
    try:
        llm_client = Client.from_env()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    # Build session with custom tools and system prompt
    config = SessionConfig()
    if args.max_turns:
        config.max_tool_rounds_per_input = args.max_turns

    session = Session(
        model=args.model,
        provider=args.provider,
        llm_client=llm_client,
        config=config,
        tool_registry=create_dot_agent_registry(),
        system_prompt_builder=build_system_prompt,
    )

    session.events.on(_on_event)

    if args.prompt:
        # Single-prompt mode
        asyncio.run(process_input(session, args.prompt))
    else:
        # Interactive REPL — use a single event loop to avoid
        # RuntimeError("Event loop is closed") from reused httpx clients.
        async def _repl() -> None:
            print("dot-agent — pipeline builder (type 'exit' to quit)")
            print("Describe the pipeline you want to build, and I'll help you create it.\n")

            while True:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("> ").strip()
                    )
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if user_input.lower() in ("exit", "quit"):
                    break
                if not user_input:
                    continue

                await process_input(session, user_input)

        asyncio.run(_repl())


if __name__ == "__main__":
    main()
