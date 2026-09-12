"""Manual xg harness interface simulation."""

import os
import sys
import termios
import tty

from rich.console import Console


RESPONSE = """\n## Simulated agent response

The turn was triggered manually by an empty request.

Automatic agent looping is disabled.
"""
TOOL_CALL = """\n[tool call pending]
Press Enter to accept the tool call.
Press Backspace to reject it.
Type anything to interrupt it and start a new request.
"""


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
            return "submit"  # Empty request: trigger a turn.
        if key.isprintable():
            _clear_line()
            value, action = _edit_request(key)
            if action == "store":
                history.append(value)
                return "stored"  # Stored only; do not trigger a turn.
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


def main() -> None:
    console = Console()
    console.print("[bold green]xg[/bold green] ready")
    console.print("Manual mode: type a message to store it; press Enter on an empty prompt to send.\n")

    history: list[str] = []
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin)
        while True:
            action = _request_loop(history)
            if action == "stored":
                # Appending history is deliberately inert. No simulated agent
                # response or tool-call state is entered here.
                continue

            console.print(RESPONSE)
            console.print(f"Stored requests: {len(history)}")
            console.print(TOOL_CALL)
            decision = _tool_decision()
            if decision == "accept":
                console.print("[green]Tool call accepted.[/green]")
            elif decision == "reject":
                console.print("[yellow]Tool call rejected.[/yellow]")
            else:
                _clear_line()
                _write("[Request LLM] (type to append message)\r\n")
                value, action = _edit_request(decision)
                if action == "store":
                    history.append(value)
    except (KeyboardInterrupt, EOFError):
        _write("\r\n")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
