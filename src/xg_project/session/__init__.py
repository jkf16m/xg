"""xg_project.session — line-by-line conversation persistence.

Public API
----------
    create(directory) -> Path
    load(directory) -> Path
    append(path, message) -> None
    remove(path, message_id) -> None
    messages(path) -> Iterator[BaseMessage]
    stream(path) -> Iterator[bytes]
"""

import time
from collections.abc import Iterator
from pathlib import Path

from langchain_core.messages import BaseMessage

LAST_FILE = ".last"


def _check_directory(directory: Path) -> None:
    """Validate that a directory path was provided."""
    if not isinstance(directory, Path):
        raise TypeError(f"expected Path, got {type(directory).__name__}")


def create(directory: Path) -> Path:
    """Create a new session file in the given directory and return its path.

    The file is named with the current UNIX timestamp. A `.last` pointer
    file is created or updated to reference the new session. The directory
    must be provided — this is not optional.
    """
    _check_directory(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    path = directory / f"{timestamp}.jsonl"
    path.write_text("")
    (directory / LAST_FILE).write_text(str(path.name))
    return path


def load(directory: Path) -> Path:
    """Read the `.last` pointer and return the path to the session file it points to."""
    _check_directory(directory)
    last = directory / LAST_FILE
    if not last.exists():
        raise FileNotFoundError(f"no session found in {directory}")
    return directory / last.read_text()


def append(path: Path, message: BaseMessage) -> None:
    """Append a single message to the session file."""
    ...


def remove(path: Path, message_id: str) -> None:
    """Remove a message by its id from the session file.

    Rewrites the file without the matching line.
    """
    ...


def messages(path: Path) -> Iterator[BaseMessage]:
    """Yield each message, one line at a time.

    Reads lazily — does not load the full file into memory.
    """
    if False:
        yield


def stream(path: Path) -> Iterator[bytes]:
    """Yield each serialized message as a newline-delimited line.

    Use this to stream the session into an HTTP body or buffer.
    """
    if False:
        yield
