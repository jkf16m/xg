"""Standalone REPL module — runs inside the PTY child process."""

import os
from pathlib import Path

from rich.console import Console

from xg_project.config import Config
from xg_project.llm import (
    add_message,
    execute_tool,
    run_turn,
    tool_result,
)
from xg_project.session import (
    append,
    configure,
    create,
    file_context,
    load,
    messages,
)


def main() -> None:
    console = Console()

    continue_session = os.environ.get("XG_CONTINUE") == "1"
    session_dir = Path.cwd() / ".xg" / "sessions"

    # Session database lives at ~/.xg/sessions.db (the module default).
    configure()

    if continue_session:
        try:
            session_path = load(session_dir)
            msgs = list(messages(session_path))
            config = Config(session_path=session_path)
            console.print("[dim]Resumed the last conversation.[/dim]")
        except FileNotFoundError:
            console.print("[yellow]No previous conversation found. Starting new.[/yellow]")
            session_path = create(session_dir)
            config = Config(session_path=session_path)
            msgs = file_context(Path.cwd(), config)
            for msg in msgs:
                append(session_path, msg)
    else:
        session_path = create(session_dir)
        config = Config(session_path=session_path)
        msgs = file_context(Path.cwd(), config)
        for msg in msgs:
            append(session_path, msg)

    console.print("[bold green]xg[/bold green] ready")
    console.print("Type your message, or Ctrl-C / Ctrl-D to exit.\n")

    while True:
        try:
            prompt = input("human> ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return

        # Non-empty prompt = store it in context, no LLM call.
        if prompt.strip():
            human_msg = add_message([], prompt)[0]
            append(session_path, human_msg)
            msgs.append(human_msg)
            continue

        # Empty prompt = one LLM call.
        response, msgs = run_turn(msgs, config)

        if response.content:
            console.print(response.content)

        # Show tool calls for the human to decide on.
        if response.tool_calls:
            for tc in response.tool_calls:
                console.print(f"  [dim]tool_call: {tc['name']}({tc['args']})[/dim]")
            console.print(
                "[yellow]Type 'y' to execute, or type a new message to continue.[/yellow]"
            )

            try:
                decision = input("> ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                return

            if decision.strip().lower() == "y":
                for tc in response.tool_calls:
                    tm = execute_tool({"name": tc["name"], "args": tc["args"], "id": tc["id"]})
                    console.print(f"  [dim]{tm.name}({tc['args']})[/dim]")
                    tool_msg = tool_result([], tm)[0]
                    append(session_path, tool_msg)
                    msgs.append(tool_msg)


if __name__ == "__main__":
    main()
