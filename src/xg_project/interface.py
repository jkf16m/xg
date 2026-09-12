"""xg harness interface — real LLM calls via llm module."""

import json
import os
import sys
import termios
import tty
from pathlib import Path

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from xg_project.llm import (
    add_message,
    execute_tool,
    initial_file_messages,
    stream_turn,
    tool_result,
)


def _write(text: str) -> None:
    os.write(sys.stdout.fileno(), text.encode())


def _clear_line() -> None:
    _write("\r\033[2K")


def _read_key() -> str:
    return os.read(sys.stdin.fileno(), 1).decode("utf-8", errors="replace")


def _edit_request(initial: str = "") -> tuple[str, str]:
    """Read input. Non-empty Enter stores history; empty Enter submits."""
    value = list(initial)
    _write(">> " + initial)
    while True:
        key = _read_key()
        if key in ("\r", "\n"):
            _write("\r\n")
            return "".join(value), "submit" if not value else "store"
        if key in ("\x03", "\x04"):
            raise KeyboardInterrupt
        if key in ("\x7f", "\b"):
            if value:
                value.pop()
                _write("\b \b")
            continue
        if key.isprintable():
            value.append(key)
            _write(key)


def _request_loop(history: list[str]) -> str:
    """Show the idle hint and return whether this interaction submits."""
    _clear_line()
    _write("[Request LLM] (type to append message)")
    while True:
        key = _read_key()
        if key in ("\x03", "\x04"):
            raise KeyboardInterrupt
        if key in ("\r", "\n"):
            _write("\r\n")
            return "submit"
        if key.isprintable():
            _clear_line()
            value, action = _edit_request(key)
            if action == "store":
                history.append(value)
                return "stored"
            return "submit"


def _tool_decision() -> str:
    while True:
        key = _read_key()
        if key in ("\r", "\n"):
            _write("\r\n")
            return "accept"
        if key in ("\x7f", "\b"):
            _write("\r\n")
            return "reject"
        if key in ("\x03", "\x04"):
            raise KeyboardInterrupt
        if key.isprintable():
            _write("\r\n")
            return key


def _conversation_path() -> Path:
    return Path.cwd() / ".xg" / "conversation.json"


def _load_messages() -> list[BaseMessage]:
    with _conversation_path().open() as stream:
        return messages_from_dict(json.load(stream))


def _save_messages(messages: list[BaseMessage]) -> None:
    path = _conversation_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(messages_to_dict(messages), stream, indent=2)


def main() -> None:
    console = Console()
    console.print("[bold green]xg[/bold green] ready")
    console.print(
        "Single-turn mode: type a message to store it;"
        " press Enter on an empty prompt to send.\n"
    )

    resume = os.environ.get("XG_RESUME") == "1"
    conversation_path = _conversation_path()
    if resume and conversation_path.exists():
        messages = _load_messages()
        console.print("[dim]Resumed the last conversation.[/dim]")
    else:
        messages = initial_file_messages(Path.cwd())
    history: list[str] = []
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin)
        while True:
            action = _request_loop(history)
            if action == "stored":
                continue

            # Build messages from history
            for h in history:
                messages = add_message(messages, h)
            history.clear()

            # One LLM turn. Live re-renders the complete Markdown document as
            # each character arrives, so headings, lists, and code fences are
            # rendered dynamically rather than printed as plain text.
            rendered: list[str] = []

            def receive_character(
                character: str, _rendered: list[str] = rendered
            ) -> None:
                _rendered.append(character)
                live.update(Markdown("".join(_rendered)))

            console.print("\n[bold]xg:[/bold]")
            with Live(Markdown(""), console=console, refresh_per_second=30) as live:
                response, messages = stream_turn(messages, receive_character)

            # Handle tool calls
            if response.tool_calls:
                for tc in response.tool_calls:
                    console.print(f"  [dim]tool_call: {tc['name']}({tc['args']})[/dim]")
                console.print("[yellow]Press Enter to accept, Backspace to reject.[/yellow]")

                decision = _tool_decision()
                if decision == "accept":
                    for tc in response.tool_calls:
                        tm = execute_tool({"name": tc["name"], "args": tc["args"], "id": tc["id"]})
                        console.print(f"  [dim]{tm.name} -> {tm.content[:100]}[/dim]")
                        messages = tool_result(messages, tm)
                else:
                    console.print("[yellow]Tool call rejected.[/yellow]")

            _save_messages(messages)
            console.print("\n[dim]Turn saved. Run `xg -r` to continue.[/dim]")
            return

    except (KeyboardInterrupt, EOFError):
        _write("\r\n")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
