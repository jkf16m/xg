"""Standalone REPL module — runs inside the PTY child process."""

from pathlib import Path

from rich.console import Console

from xg_project.graph import graph, initial_file_messages


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

        result = graph.invoke({"messages": messages, "prompt": prompt})
        messages = result.get("messages", messages)
        answer = messages[-1].content if messages else ""
        if answer:
            console.print(answer)


if __name__ == "__main__":
    main()
