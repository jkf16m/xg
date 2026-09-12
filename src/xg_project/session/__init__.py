"""xg_project.session — conversation persistence via SQLite.

Public API
----------
    create(directory) -> Path
    load(directory) -> Path
    append(path, message) -> None
    remove(path, message_id) -> None
    messages(path) -> Iterator[BaseMessage]
    stream(path) -> Iterator[bytes]
    migrate() -> None
"""

import json
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

DB_DIR = Path.home() / ".xg"
DB_PATH = DB_DIR / "sessions.db"
LAST_FILE = ".last"

TYPE_MAP = {
    "ai": AIMessage,
    "human": HumanMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
}

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        _init_db()
        _conn = sqlite3.connect(str(DB_PATH))
    return _conn


def _init_db() -> None:
    from xg_project.session._migrations import migrate

    migrate(DB_PATH)


def _check_directory(directory: Path) -> None:
    """Validate that a directory path was provided."""
    if not isinstance(directory, Path):
        raise TypeError(f"expected Path, got {type(directory).__name__}")


def migrate() -> None:
    """Run any pending schema migrations."""
    _init_db()


def _get_or_create_session_id(conn: sqlite3.Connection, directory: Path, filename: str) -> int:
    row = conn.execute(
        "SELECT id FROM sessions WHERE directory = ? AND filename = ?",
        (str(directory), filename),
    ).fetchone()
    if row:
        return row[0]
    conn.execute(
        "INSERT INTO sessions (directory, filename) VALUES (?, ?)",
        (str(directory), filename),
    )
    conn.commit()
    return conn.execute(
        "SELECT id FROM sessions WHERE directory = ? AND filename = ?",
        (str(directory), filename),
    ).fetchone()[0]


def create(directory: Path) -> Path:
    """Create a new session file in the given directory and return its path.

    The file is named with the current UNIX timestamp. A `.last` pointer
    file is created or updated to reference the new session. The directory
    must be provided — this is not optional.
    """
    _check_directory(directory)
    conn = _get_conn()
    timestamp = int(time.time())
    filename = f"{timestamp}.jsonl"

    _get_or_create_session_id(conn, directory, filename)

    directory.mkdir(parents=True, exist_ok=True)
    (directory / LAST_FILE).write_text(filename)
    return directory / filename


def load(directory: Path) -> Path:
    """Read the `.last` pointer and return the path to the session file it points to."""
    _check_directory(directory)
    conn = _get_conn()

    row = conn.execute(
        "SELECT filename FROM sessions WHERE directory = ? ORDER BY id DESC LIMIT 1",
        (str(directory),),
    ).fetchone()
    if row:
        return directory / row[0]

    last = directory / LAST_FILE
    if not last.exists():
        raise FileNotFoundError(f"no session found in {directory}")
    return directory / last.read_text()


def append(path: Path, message: BaseMessage) -> None:
    """Append a single message to the session file."""
    conn = _get_conn()
    data = message.model_dump()
    msg_id = data.get("id")
    msg_type = data.get("type", "unknown")
    content = data.get("content", "")

    session_id = _get_or_create_session_id(conn, path.parent, path.name)
    conn.execute(
        "INSERT INTO messages (session_id, message_id, message_type, content, data)"
        " VALUES (?, ?, ?, ?, ?)",
        (session_id, msg_id, msg_type, content, json.dumps(data)),
    )
    conn.commit()


def remove(path: Path, message_id: str) -> None:
    """Remove a message by its id from the session file.

    Rewrites the file without the matching line.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT id FROM sessions WHERE directory = ? AND filename = ?",
        (str(path.parent), path.name),
    ).fetchone()
    if row:
        conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND message_id = ?",
            (row[0], message_id),
        )
        conn.commit()


def messages(path: Path) -> Iterator[BaseMessage]:
    """Yield each message, one line at a time.

    Reads lazily — does not load the full file into memory.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT id FROM sessions WHERE directory = ? AND filename = ?",
        (str(path.parent), path.name),
    ).fetchone()
    if not row:
        return
    session_id = row[0]
    rows = conn.execute(
        "SELECT data FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    for (data_str,) in rows:
        data = json.loads(data_str)
        msg_type = data.pop("type", None)
        cls = TYPE_MAP.get(msg_type)
        if cls is None:
            continue
        yield cls(**data)


def stream(path: Path) -> Iterator[bytes]:
    """Yield each serialized message as a newline-delimited line.

    Use this to stream the session into an HTTP body or buffer.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT id FROM sessions WHERE directory = ? AND filename = ?",
        (str(path.parent), path.name),
    ).fetchone()
    if not row:
        return
    session_id = row[0]
    rows = conn.execute(
        "SELECT data FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    for (data_str,) in rows:
        yield (data_str + "\n").encode()
