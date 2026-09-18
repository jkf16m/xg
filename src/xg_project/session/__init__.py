"""xg_project.session — conversation persistence.

Rebuild status
--------------
The SQLite/sqlite-utils implementation and its migrations were removed on the
``refactor/jev-langgraph`` branch. The names below are kept so the new
persistence layer can be wired in without re-deciding the module's public
surface. Every callable raises ``NotImplementedError``.
"""

from collections.abc import Iterator
from pathlib import Path

from langchain_core.messages import BaseMessage

from xg_project.config import Config

DEFAULT_DB_DIR = Path.home() / ".xg"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "sessions.db"


def configure(db_path: Path | None = None) -> None:
    """Configure the storage location. Call before using other functions."""
    raise NotImplementedError("session.configure: rebuild pending")


def create(directory: Path) -> Path:
    """Create a new session in ``directory`` and return its logical path key."""
    raise NotImplementedError("session.create: rebuild pending")


def load(directory: Path) -> Path:
    """Return the logical path key of the most recent session in ``directory``."""
    raise NotImplementedError("session.load: rebuild pending")


def append(path: Path, message: BaseMessage) -> None:
    """Append a single message to the session identified by ``path``."""
    raise NotImplementedError("session.append: rebuild pending")


def remove(path: Path, message_id: str) -> None:
    """Remove a message by its id from the session identified by ``path``."""
    raise NotImplementedError("session.remove: rebuild pending")


def messages(path: Path) -> Iterator[BaseMessage]:
    """Yield each message in the session identified by ``path``, in order."""
    raise NotImplementedError("session.messages: rebuild pending")


def stream(path: Path) -> Iterator[bytes]:
    """Yield each serialized message as a newline-delimited line."""
    raise NotImplementedError("session.stream: rebuild pending")


def migrate() -> None:
    """Run any pending schema migrations."""
    raise NotImplementedError("session.migrate: rebuild pending")


def file_context(directory: Path, config: Config | None = None) -> list[BaseMessage]:
    """Build the initial file context for a session."""
    raise NotImplementedError("session.file_context: rebuild pending")
