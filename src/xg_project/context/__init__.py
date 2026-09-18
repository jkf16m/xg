"""Deterministic context windows.

A window is an ordered, fully-read view of the files Jev gathered. Nothing here
calls a model: the same files with the same mtimes always produce the same text,
which is what makes the generative step reproducible.

Files are read whole. There is no prefilter and no truncation, because the
research step already chose them and a silent trim would hide the evidence the
choice was made on. A file that cannot be read is reported in place rather than
dropped, so a window never quietly loses an entry.

The window is written to a temporary file outside the repository, so it can be
inspected after a run without polluting the project or its git listing.
"""

import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ContextEntry:
    """One file in a window: its path, its mtime, and its full text."""

    path: str
    mtime: float
    text: str
    missing: bool = False
    """True when the file could not be read; ``text`` then explains why."""


def read_entries(
    files: list[str], *, root: Path, newest_first: bool = True
) -> list[ContextEntry]:
    """Read every file whole, ordered by mtime.

    Ordering is by mtime, with the path as a tie-break, so a window does not
    depend on the order the research step happened to return. ``newest_first``
    puts the most recently changed files nearest the top, which is where a
    reader looks first.
    """
    base = Path(root).resolve()
    entries: list[ContextEntry] = []
    for relative in files:
        target = base / relative
        try:
            text = target.read_text(encoding="utf-8")
            mtime = target.stat().st_mtime
        except (OSError, UnicodeDecodeError) as exc:
            entries.append(
                ContextEntry(relative, 0.0, f"[unreadable: {exc}]", missing=True)
            )
            continue
        entries.append(ContextEntry(relative, mtime, text))
    entries.sort(key=lambda entry: (-entry.mtime if newest_first else entry.mtime, entry.path))
    return entries


def render(entries: list[ContextEntry], *, request: str = "") -> str:
    """Render a window as text: the request, then every file whole."""
    lines: list[str] = []
    if request:
        lines += ["# Request", "", request, ""]
    lines += [f"# Context ({len(entries)} files, newest mtime first)", ""]
    for entry in entries:
        if entry.missing:
            stamp = "missing"
        else:
            stamp = datetime.fromtimestamp(entry.mtime, tz=UTC).isoformat()
        lines += [f"## {entry.path}  (mtime {stamp})", "", entry.text, ""]
    return "\n".join(lines)


def write_window(text: str, *, directory: Path | None = None) -> Path:
    """Write a rendered window to a temporary file and return its path."""
    base = Path(directory) if directory is not None else Path(
        tempfile.mkdtemp(prefix="xg-context-")
    )
    base.mkdir(parents=True, exist_ok=True)
    path = base / "window.md"
    path.write_text(text, encoding="utf-8")
    return path
