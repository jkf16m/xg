"""Private: tool definitions."""

import os
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Read a file; its result is appended at the bottom of the context."""
    return Path(path).read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Write a file."""
    Path(path).write_text(content)
    return f"wrote {path} ({len(content)} chars)"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one exact, unique occurrence in a file."""
    text = Path(path).read_text()
    if text.count(old_text) != 1:
        return f"error: old_text must occur exactly once in {path}"
    Path(path).write_text(text.replace(old_text, new_text))
    return f"edited {path}"


@tool
def run_cmd(command: str) -> str:
    """Run a command; interactive commands temporarily take over the PTY."""
    if os.environ.get("XG_PTY") != "1":
        completed = subprocess.run(  # noqa: S602
            command, shell=True, capture_output=True, text=True
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
    try:
        if sys.stdin.isatty():
            child.interact(escape_character=None)
        else:
            child.expect(pexpect.EOF)
        output = child.before or ""
        status = child.exitstatus
    finally:
        child.close(force=True)

    return output.strip() or f"(exit code {status or 0})"


TOOLS = [read_file, write_file, edit_file, run_cmd]
