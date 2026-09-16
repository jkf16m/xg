"""xg_project.session — conversation persistence via SQLite.

All sessions and messages are stored in a single SQLite database. A session
is identified by a logical path key — a composite of (directory, filename)
that is resolved to a canonical absolute path before storage or lookup.

Public API
----------
    configure(db_path) -> None
    create(directory) -> Path          # logical session key
    load(directory) -> Path            # logical session key
    append(path, message) -> None      # path = session key
    remove(path, message_id) -> None
    messages(path) -> Iterator[BaseMessage]
    stream(path) -> Iterator[bytes]
    migrate() -> None
    file_context(directory, config) -> list[BaseMessage]
"""

import json
import time
from collections.abc import Iterator
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from sqlite_utils import Database

from xg_project.config import Config

DEFAULT_DB_DIR = Path.home() / ".xg"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "sessions.db"

TYPE_MAP = {
    "ai": AIMessage,
    "human": HumanMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
}

_db_path: Path = DEFAULT_DB_PATH
_db: Database | None = None


def configure(db_path: Path | None = None) -> None:
    """Configure the database path. Call before using other functions.

    Pass a Path to set the database location, or None to reset to the default.
    Any open connection is closed so the next call reconnects with the new path.
    """
    global _db_path, _db
    if db_path is not None:
        _db_path = db_path
    else:
        _db_path = DEFAULT_DB_PATH
    _db = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db_path.parent.mkdir(parents=True, exist_ok=True)
        _db = Database(str(_db_path))
        _db.execute("PRAGMA foreign_keys = ON")
        migrate()
    return _db


def _check_directory(directory: Path) -> None:
    """Validate that a directory path was provided."""
    if not isinstance(directory, Path):
        raise TypeError(f"expected Path, got {type(directory).__name__}")


def _resolve(directory: Path) -> Path:
    """Resolve to canonical absolute path for consistent DB lookups."""
    return directory.resolve()


def migrate() -> None:
    """Run any pending schema migrations."""
    from xg_project.session._migrations import migrations

    migrations.apply(_get_db())


def _get_or_create_session_id(db: Database, directory: Path, filename: str) -> int:
    row = db.execute(
        "SELECT id FROM sessions WHERE directory = ? AND filename = ?",
        (str(directory), filename),
    ).fetchone()
    if row:
        return row[0]
    db.execute(
        "INSERT INTO sessions (directory, filename) VALUES (?, ?)",
        (str(directory), filename),
    )
    return db.execute(
        "SELECT id FROM sessions WHERE directory = ? AND filename = ?",
        (str(directory), filename),
    ).fetchone()[0]


def _session_key(path: Path) -> tuple[str, str]:
    """Extract the (directory, filename) key from a session path."""
    resolved = path.resolve()
    return str(resolved.parent), resolved.name


def create(directory: Path) -> Path:
    """Create a new session in the given directory and return its path key.

    The path is a logical identifier stored in the database — no file is
    created on disk. The directory is resolved to a canonical absolute path
    so that symlinks and relative paths map to the same session.
    """
    _check_directory(directory)
    directory = _resolve(directory)
    db = _get_db()
    filename = f"{int(time.time())}.jsonl"
    _get_or_create_session_id(db, directory, filename)
    return directory / filename


def load(directory: Path) -> Path:
    """Return the path key of the most recent session in the given directory."""
    _check_directory(directory)
    directory = _resolve(directory)
    db = _get_db()

    row = db.execute(
        "SELECT filename FROM sessions WHERE directory = ? ORDER BY id DESC LIMIT 1",
        (str(directory),),
    ).fetchone()
    if row:
        return directory / row[0]

    raise FileNotFoundError(f"no session found in {directory}")


def append(path: Path, message: BaseMessage) -> None:
    """Append a single message to the session identified by path."""
    db = _get_db()
    data = message.model_dump()
    msg_id = data.get("id")
    msg_type = data.get("type", "unknown")
    content = data.get("content", "")

    directory, filename = _session_key(path)
    session_id = _get_or_create_session_id(db, Path(directory), filename)
    db.execute(
        "INSERT INTO messages (session_id, message_id, message_type, content, data)"
        " VALUES (?, ?, ?, ?, ?)",
        (session_id, msg_id, msg_type, content, json.dumps(data)),
    )


def remove(path: Path, message_id: str) -> None:
    """Remove a message by its id from the session identified by path."""
    db = _get_db()
    directory, filename = _session_key(path)
    row = db.execute(
        "SELECT id FROM sessions WHERE directory = ? AND filename = ?",
        (directory, filename),
    ).fetchone()
    if row:
        db.execute(
            "DELETE FROM messages WHERE session_id = ? AND message_id = ?",
            (row[0], message_id),
        )


def messages(path: Path) -> Iterator[BaseMessage]:
    """Yield each message in the session identified by path, in order.

    Reads lazily — does not load the full result set into memory.
    """
    db = _get_db()
    directory, filename = _session_key(path)
    row = db.execute(
        "SELECT id FROM sessions WHERE directory = ? AND filename = ?",
        (directory, filename),
    ).fetchone()
    if not row:
        return
    session_id = row[0]
    rows = db.execute(
        "SELECT data FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    for (data_str,) in rows:
        data = json.loads(data_str)
        msg_type = data.get("type")
        cls = TYPE_MAP.get(msg_type)
        if cls is None:
            continue
        kwargs = {k: v for k, v in data.items() if k != "type"}
        yield cls(**kwargs)


def stream(path: Path) -> Iterator[bytes]:
    """Yield each serialized message as a newline-delimited line.

    Use this to stream the session into an HTTP body or buffer.
    """
    db = _get_db()
    directory, filename = _session_key(path)
    row = db.execute(
        "SELECT id FROM sessions WHERE directory = ? AND filename = ?",
        (directory, filename),
    ).fetchone()
    if not row:
        return
    session_id = row[0]
    rows = db.execute(
        "SELECT data FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    for (data_str,) in rows:
        yield (data_str + "\n").encode()


def file_context(directory: Path, config: Config | None = None) -> list[BaseMessage]:
    """Create deterministic read tool calls ordered by mtime."""
    from xg_project.llm._context import project_files

    root = directory.resolve()
    session_path = config.session_path if config else None

    if session_path:
        existing = list(messages(session_path))
        if existing:
            return existing

    result: list[BaseMessage] = []
    for number, path in enumerate(
        project_files(root, use_gitignore=config.use_gitignore if config else True), 1
    ):
        try:
            content = path.read_text()
        except (UnicodeDecodeError, OSError) as exc:
            content = f"[unreadable file: {exc}]"
        tool_id = f"launch-read-{number}"
        result.extend([
            AIMessage(content="", tool_calls=[{
                "name": "read", "args": {"path": str(path)}, "id": tool_id, "type": "tool_call"
            }]),
            ToolMessage(content=content, tool_call_id=tool_id, name="read"),
        ])
    return result
