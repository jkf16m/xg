"""Private: tool definitions."""

import os
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool


@tool
def read(path: str) -> str:
    """Read a file; its result is appended at the bottom of the context."""
    return Path(path).read_text()


@tool
def write(path: str, content: str) -> str:
    """Create a new file; existing files must be changed with edit."""
    file_path = Path(path)
    if file_path.exists():
        return f"error: {path} already exists; use edit for existing files"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return f"created {path} ({len(content)} chars)"


@tool
def edit(path: str, old_text: str, new_text: str) -> str:
    """Replace one exact, unique occurrence in an existing file."""
    file_path = Path(path)
    if not file_path.exists():
        return f"error: {path} does not exist; use write for new files"
    text = file_path.read_text()
    if text.count(old_text) != 1:
        return f"error: old_text must occur exactly once in {path}"
    file_path.write_text(text.replace(old_text, new_text))
    return f"edited {path}"


@tool
def cmd(command: str) -> str:
    """Run a command in the current project environment."""
    if os.environ.get("XG_PTY") != "1":
        completed = subprocess.run(  # noqa: S602
            command, shell=True, capture_output=True, text=True, check=False
        )
        output = (completed.stdout + completed.stderr).strip()
        return output or f"(exit code {completed.returncode})"

    import pexpect

    child = pexpect.spawn(
        "/bin/sh", ["-c", command],
        encoding="utf-8",
        timeout=None,
        dimensions=(24, 80),
    )
    seen: list[str] = []

    def collect(data: bytes) -> bytes:
        """Record every chunk the child prints so the model can see it.

        ``interact`` reads the pty directly and never fills ``child.before``,
        so this filter is the only place the output can be captured from.
        """
        seen.append(data.decode("utf-8", errors="replace"))
        return data

    try:
        child.setecho(False)
        if sys.stdin.isatty():
            child.interact(escape_character=None, output_filter=collect)
            output = "".join(seen)
        else:
            child.expect(pexpect.EOF)
            output = child.before or ""
        status = child.exitstatus
    finally:
        child.close(force=True)

    return output.replace("\r\n", "\n").replace("\r", "\n").strip() or (
        f"(exit code {status or 0})"
    )


TOOLS = [read, write, edit, cmd]
