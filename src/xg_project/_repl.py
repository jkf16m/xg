"""Standalone REPL module — runs inside the PTY child process."""

from pathlib import Path

from rich.console import Console

from xg_project.llm import (
    TOOLS,
    add_message,
    execute_tool,
    initial_file_messages,
    run_turn,
    tool_result,
)


def main() -> None:
    console = Console()
    messages = initial_file_messages(Path.cwd())
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
            messages = add_message(messages, prompt)
            continue

        # Empty prompt = one LLM call.
        response, messages = run_turn(messages, TOOLS)

        if response.content:
            console.print(response.content)

        # Show tool calls for the human to decide on.
        if response.tool_calls:
            for tc in response.tool_calls:
                console.print(f"  [dim]tool_call: {tc['name']}({tc['args']})[/dim]")
            console.print("[yellow]Type 'y' to execute, or type a new message to continue.[/yellow]")

            try:
                decision = input("> ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                return

            if decision.strip().lower() == "y":
                for tc in response.tool_calls:
                    tm = execute_tool(tc)
                    console.print(f"  [dim]{tm.name}({tc['args']})[/dim]")
                    messages = tool_result(messages, tm)


if __name__ == "__main__":
    main()
