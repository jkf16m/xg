"""xg harness interface — single-turn REPL with session persistence."""

import difflib
import os
import re
import select
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

from langchain_core.messages import ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from xg_project.config import Config, resolve
from xg_project.llm import (
    ProviderError,
    add_message,
    execute_tool,
    stream_turn,
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

ESC = "\x1b"

# At most one trailing-block redraw per frame. Text still arrives at whatever
# rate the provider emits it; this only caps how often the terminal is repainted
# so a fast provider cannot spend the reply re-rendering its own last block.
_MIN_DRAW_INTERVAL = 1 / 60

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*(\S*)[ \t]*$")
_LIST_RE = re.compile(r"^ {0,3}(?:[-*+]|\d{1,9}[.)])[ \t]")


def _write(text: str) -> None:
    os.write(sys.stdout.fileno(), text.encode())


def _crt(text: str) -> str:
    """Use CRLF line endings, correct whether or not output processing is on."""
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def _closes_fence(line: str, char: str, length: int) -> bool:
    """Whether ``line`` closes a fence of ``char`` at least ``length`` wide."""
    match = _FENCE_RE.match(line)
    return bool(
        match
        and match.group(1)[0] == char
        and len(match.group(1)) >= length
        and not match.group(2)
    )


class _MarkdownWriter:
    """Stream a reply to the terminal as rendered markdown.

    Markdown cannot be parsed incrementally, so the reply is split into
    complete blocks (paragraphs, lists, code fences, ...) at blank lines and
    closing fences. A finished block is rendered once and kept. Only the block
    still arriving is redrawn, which costs work proportional to the trailing
    block rather than to the whole reply and is what keeps the animation tied
    to the provider's token rate.

    The trailing block is repainted by moving the cursor back over it, so it
    must stay shorter than the screen: once it reaches that height it is
    written permanently and the next content is treated as a fresh block. A
    code fence that is split this way is reopened, so the remainder still
    renders as code.
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        # Every character received so far, and how much of it has been written
        # to the terminal for good.
        self._buffer = ""
        self._printed = 0
        # Boundaries are found by scanning whole lines from the last newline.
        self._line_start = 0
        self._scan = 0
        # Where the current block began, and where a blank line inside a list
        # would end it if the next line turns out not to continue the list.
        self._block_start = 0
        self._list_break: int | None = None
        # Fence state, so blank lines and fences inside code are not treated as
        # block boundaries.
        self._in_fence = False
        self._fence_char = ""
        self._fence_len = 0
        self._fence_line = ""
        # A fence reopened after a partial write, prepended to the next render.
        self._prefix = ""
        # Lines currently painted for the trailing block, and whether a blank
        # line already separates the previous block from this one.
        self._tail_lines = 0
        self._sep_written = False
        self._wrote_any = False
        self._last_draw = 0.0
        self.wrote = False

    # -- streaming ------------------------------------------------------

    def write(self, text: str) -> None:
        """Accept the next streamed fragment and paint what is complete."""
        if not text:
            return
        self._buffer += text
        self.wrote = True
        self._commit_ready()
        self._draw_tail(throttle=True)

    def finish(self) -> None:
        """Commit whatever remains, then leave the cursor after the reply."""
        self._commit_ready()
        self._draw_tail(throttle=False)
        self._flush_tail()

    # -- block boundaries -----------------------------------------------

    def _commit_ready(self) -> None:
        """Write every block completed by a newline seen so far."""
        while True:
            newline = self._buffer.find("\n", self._scan)
            if newline == -1:
                break
            line = self._buffer[self._line_start:newline]
            self._scan = newline + 1
            end: int | None = None
            if self._in_fence:
                if _closes_fence(line, self._fence_char, self._fence_len):
                    self._in_fence = False
                    end = newline + 1
            else:
                opener = _FENCE_RE.match(line)
                if opener:
                    self._in_fence = True
                    self._fence_char = opener.group(1)[0]
                    self._fence_len = len(opener.group(1))
                    self._fence_line = line
                elif not line.strip():
                    # A blank line ends most blocks, but a list may continue
                    # after one, so its end is only decided by the next line.
                    if self._is_list_block():
                        self._list_break = self._line_start
                    else:
                        end = self._line_start
                elif self._list_break is not None:
                    # The blank line ended a list unless this line carries on.
                    boundary = self._list_break
                    self._list_break = None
                    if not self._list_continues(line):
                        end = boundary
            self._line_start = newline + 1
            if end is not None and end > self._printed:
                self._commit(end)

    def _is_list_block(self) -> bool:
        """Whether the current block is a list rather than a paragraph."""
        head = self._buffer[self._block_start:].lstrip("\n")
        return bool(head) and _LIST_RE.match(head.split("\n", 1)[0]) is not None

    @staticmethod
    def _list_continues(line: str) -> bool:
        """Whether ``line`` keeps a list going after a blank line."""
        return bool(line[:1].isspace()) or _LIST_RE.match(line) is not None

    def _commit(self, end: int) -> None:
        """Write ``buffer[printed:end]`` as a finished block."""
        segment = self._prefix + self._buffer[self._printed:end]
        self._prefix = ""
        self._printed = end
        self._block_start = end
        if not segment.strip():
            return
        self._erase_tail()
        self._write_sep()
        _write(_crt(self._render(segment)))
        self._sep_written = False

    # -- trailing block -------------------------------------------------

    def _draw_tail(self, *, throttle: bool) -> None:
        """Repaint the block that is still arriving."""
        if throttle:
            now = time.monotonic()
            if now - self._last_draw < _MIN_DRAW_INTERVAL:
                return
            self._last_draw = now

        tail = self._prefix + self._buffer[self._printed:]
        if not tail.strip():
            self._erase_tail()
            return
        rendered = self._render(tail)
        if len(rendered) and not rendered.endswith("\n"):
            rendered += "\n"
        height = rendered.count("\n")
        if height > self._max_tail_lines():
            # A block may not be repainted taller than the screen. A fence is
            # safe to split because it can be reopened; the cut is made at the
            # last complete line so no word is broken across the two halves.
            # Any other block is left as drawn and written whole once it ends,
            # since splitting it would render the remainder in the wrong shape.
            if self._in_fence:
                cut = self._buffer.rfind("\n")
                if cut >= self._printed:
                    self._flush_tail(cut + 1)
                elif height >= self._screen_height() - 1:
                    # A single line too tall to repaint: split it to stay safe.
                    self._flush_tail()
            return
        self._erase_tail()
        self._write_sep()
        _write(_crt(rendered))
        self._tail_lines = height

    def _flush_tail(self, end: int | None = None) -> None:
        """Write the trailing block up to ``end`` permanently, reopening a fence."""
        end = len(self._buffer) if end is None else end
        tail = self._prefix + self._buffer[self._printed:end]
        self._prefix = ""
        self._printed = end
        self._block_start = end
        if tail.strip():
            self._erase_tail()
            self._write_sep()
            _write(_crt(self._render(tail)))
        else:
            self._erase_tail()
        self._tail_lines = 0
        if self._in_fence:
            self._prefix = self._fence_line + "\n"
            self._sep_written = True
        else:
            self._sep_written = False

    def _erase_tail(self) -> None:
        """Move back over the painted trailing block and erase it."""
        if not self._tail_lines:
            return
        _write(f"\r\x1b[{self._tail_lines}A\x1b[J")
        self._tail_lines = 0

    def _write_sep(self) -> None:
        """Leave a blank line before content that follows a finished block."""
        if not self._sep_written:
            if self._wrote_any:
                _write("\r\n")
            self._sep_written = True
        self._wrote_any = True

    def _screen_height(self) -> int:
        """Terminal height, with a usable default off a terminal."""
        return self._console.height or 25

    def _max_tail_lines(self) -> int:
        """Height a trailing block may reach before it stops being repainted."""
        return max(4, self._screen_height() - 4)

    def _render(self, text: str) -> str:
        """Render markdown to the console's width, with styles, as a string."""
        with self._console.capture() as capture:
            self._console.print(Markdown(text), end="")
        return capture.get()


def _read_byte() -> bytes:
    """Read one byte from stdin, raising EOFError at end of input."""
    data = os.read(sys.stdin.fileno(), 1)
    if not data:
        raise EOFError
    return data


def _read_key() -> str:
    """Read one logical key, consuming ANSI escape sequences as a unit.

    A bare ESC remains the exit key. CSI/SS3 sequences generated by arrows,
    Home/End, and similar keys are consumed and ignored by the line editor,
    so their bytes cannot leak into the submitted prompt. End of input raises
    EOFError so the caller can exit instead of spinning on an empty read.
    """
    key = _read_byte().decode("utf-8", errors="replace")
    if key != ESC:
        return key

    # A bare ESC is intentional; an escape sequence has more bytes shortly
    # after it. The short timeout keeps ESC responsive without blocking.
    if not select.select([sys.stdin], [], [], 0.03)[0]:
        return ESC

    prefix = _read_byte().decode("utf-8", errors="replace")
    if prefix not in ("[", "O"):
        return ""

    # CSI/SS3 sequences terminate with a byte in the final-byte range.
    while True:
        value = _read_byte()[0]
        if 0x40 <= value <= 0x7E:
            return ""


def _get_session_dir() -> Path:
    return Path.cwd() / ".xg" / "sessions"


def _readline(initial: str = "", prompt: str = "> ") -> str | None:
    """Read a line. Returns None on ESC, empty string on Enter, or the text."""
    value = list(initial)
    # Keep the interface minimal: a simple prompt, with user text styled.
    _write(prompt + "\033[36m" + initial)
    while True:
        key = _read_key()
        if key == ESC:
            _write("\r\n")
            return None
        if key in ("\r", "\n"):
            if value:
                _write("\033[0m\r\n")
            else:
                # Remove the empty prompt instead of leaving a marker on screen.
                _write("\033[0m\r\033[2K\n")
            return "".join(value)
        if key in ("\x03", "\x04"):
            raise KeyboardInterrupt
        if key in ("\x7f", "\b"):
            if value:
                value.pop()
                _write("\b \b")
            continue
        if key.isprintable():
            value.append(key)
            _write(key + "\033[36m")


def _unique_tool_calls(tool_calls: list[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse exact duplicate tool calls, preserving first-seen order.

    A provider can repeat a complete tool call across stream chunks, which the
    chunk merge appends rather than combines. Acting on every copy would run the
    same command many times, so duplicates are dropped before approval.
    """
    unique: list[dict[str, object]] = []
    seen: set[tuple[object, str]] = set()
    for call in tool_calls:
        args = call.get("args", {})
        signature = (call["name"], repr(sorted(args.items())))
        if signature not in seen:
            seen.add(signature)
            unique.append(call)
    return unique


def _tool_label(call: dict[str, object]) -> Text:
    """Render one tool call as a single line for the approval panel."""
    name = call["name"]
    args = call.get("args", {})
    if name == "cmd":
        return Text(f"$ {args.get('command', '')}", style="yellow")
    return Text(f"{name} {args.get('path', '')}".rstrip(), style="cyan")


def _preview_patch(name: str, args: dict[str, object]) -> str | None:
    """Build a unified diff for a write/edit call, for preview only.

    Display-only: the result is never sent to the model or stored. It mirrors
    the tool semantics so the preview matches what the call would change.
    """
    if name not in ("write", "edit"):
        return None
    path = Path(str(args.get("path", "")))
    if name == "write":
        if path.exists():
            return None
        old, new, fromfile = "", str(args.get("content", "")), "/dev/null"
    else:
        if not path.exists():
            return None
        old = path.read_text()
        old_text = str(args.get("old_text", ""))
        if old.count(old_text) != 1:
            return None
        new = old.replace(old_text, str(args.get("new_text", "")))
        fromfile = str(path)
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=str(path),
        )
    )


def _cmd_streams_live() -> bool:
    """Whether cmd streams its own output during interact.

    When true the command has already printed to the terminal, so the
    interface must not print its result again.
    """
    return os.environ.get("XG_PTY") == "1" and sys.stdin.isatty()


def _format_patch(patch: str, formatter: str | None) -> str | None:
    """Run a patch through an external formatter, or return None to fall back.

    ``formatter`` is a shell command read from settings (for example
    ``delta --paging=never``). A missing command or a non-zero exit falls back
    to the in-process renderer.
    """
    if not formatter:
        return None
    result = subprocess.run(  # noqa: S602
        formatter,
        shell=True,
        input=patch,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def _render_patch(patch: str, formatter: str | None, console: Console) -> None:
    """Show a preview patch, through the configured formatter when available."""
    formatted = _format_patch(patch, formatter)
    if formatted is None:
        console.print(Syntax(patch, "diff", background_color="default"))
        return
    _write(formatted if formatted.endswith("\n") else formatted + "\n")


def _tool_prompt(tool_calls: list[dict[str, object]], console: Console) -> str:
    """Approve, inspect, extract, or replace pending tool calls."""
    console.print(
        Text("Enter=allow  /v=view  /e=extract command  text=cancel", style="dim")
    )
    while True:
        choice = _readline(prompt="tool> ")
        if choice is None:
            raise SystemExit
        if not choice:
            return "accept"
        if choice == "/v":
            for call in tool_calls:
                console.print(Text(f"{call['name']} {call.get('args', {})}", style="dim"))
            continue
        if choice == "/e":
            commands = [
                str(call.get("args", {}).get("command", ""))
                for call in tool_calls
                if call["name"] == "cmd"
            ]
            if not commands:
                console.print("[red]/e is only available for command calls[/red]")
                continue
            os.execvp("/bin/sh", ["sh", "-lc", "\n".join(commands)])  # noqa: S606
        return f"cancel:{choice}"


def main() -> None:
    console = Console()
    continue_session = os.environ.get("XG_CONTINUE") == "1"
    session_dir = _get_session_dir()

    # Session database lives at ~/.xg/sessions.db (the module default).
    configure()
    project_config = resolve(Path.cwd())
    os.environ["XG_PROJECT_ROOT"] = str(Path.cwd().resolve())

    if continue_session:
        try:
            session_path = load(session_dir)
            msgs = list(messages(session_path))
            config = Config(
                session_path=session_path,
                use_gitignore=project_config.use_gitignore,
            )
            console.print("[dim]Resumed the last conversation.[/dim]\n")
        except FileNotFoundError:
            console.print("[yellow]No previous conversation found. Starting new.[/yellow]\n")
            session_path = create(session_dir)
            config = Config(
                session_path=session_path,
                use_gitignore=project_config.use_gitignore,
            )
            msgs = file_context(Path.cwd(), config)
            for msg in msgs:
                append(session_path, msg)
    else:
        session_path = create(session_dir)
        config = Config(
            session_path=session_path,
            use_gitignore=project_config.use_gitignore,
        )
        msgs = file_context(Path.cwd(), config)
        for msg in msgs:
            append(session_path, msg)

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin)
        while True:
            # Read input
            prompt = _readline()

            if prompt is None:
                # ESC pressed — exit
                return

            # Non-empty prompt = add to context
            if prompt.strip():
                human_msg = add_message([], prompt)[0]
                append(session_path, human_msg)
                msgs.append(human_msg)
                continue

            # Empty prompt = one turn. Reply content is rendered as markdown
            # and painted as it arrives so the answer animates in place.
            writer = _MarkdownWriter(console)
            try:
                response, msgs = stream_turn(msgs, config, writer.write)
            except ProviderError as exc:
                writer.finish()
                detail = (
                    f"{exc.message} (HTTP {exc.status_code})"
                    if exc.status_code
                    else exc.message
                )
                console.print(Text(f"provider error: {detail}", style="red"))
                console.print(Text("press Enter to retry", style="dim"))
                console.print()
                continue
            writer.finish()

            # A response with no streamable text (for example tool calls only,
            # or block content) still needs to reach the user.
            if not writer.wrote and response.content:
                text = response.content
                console.print(Markdown(text if isinstance(text, str) else str(text)))

            # Present tool calls for approval.
            if response.tool_calls:
                # Models can emit the same call more than once. Collapse exact
                # duplicates so the user approves and runs it only once.
                tool_calls = _unique_tool_calls(response.tool_calls)
                for call in tool_calls:
                    console.print(_tool_label(call))
                    patch = _preview_patch(str(call["name"]), call.get("args", {}))
                    if patch:
                        _render_patch(patch, project_config.patch_formatter, console)

                decision = _tool_prompt(tool_calls, console)
                if decision == "accept":
                    for tc in tool_calls:
                        tm = execute_tool(
                            {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
                        )
                        if tc["name"] == "cmd" and not _cmd_streams_live():
                            console.print(Text(tm.content))
                        tool_msg = tool_result([], tm)[0]
                        append(session_path, tool_msg)
                        msgs.append(tool_msg)
                    if any(tc["name"] in ("write", "edit") for tc in tool_calls):
                        console.print(Text("approved", style="green"))
                elif decision.startswith("cancel:"):
                    instruction_text = decision.removeprefix("cancel:")
                    for tc in tool_calls:
                        cancelled = ToolMessage(
                            content="tool call cancelled by user",
                            tool_call_id=str(tc["id"]),
                            name=str(tc["name"]),
                            status="error",
                        )
                        append(session_path, cancelled)
                        msgs.append(cancelled)
                    user_msg = add_message([], instruction_text)[0]
                    append(session_path, user_msg)
                    msgs.append(user_msg)
                    console.print(Text("Tool calls cancelled", style="red"))

            console.print()  # blank line before next prompt

    except (KeyboardInterrupt, EOFError, SystemExit):
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
