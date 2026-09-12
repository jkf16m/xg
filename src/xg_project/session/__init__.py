"""xg_project.session — conversation persistence.

Public API
----------
    session_path() -> Path
    save(messages) -> None
    load() -> list
"""

import json
from pathlib import Path

from langchain_core.messages import messages_from_dict, messages_to_dict


def session_path() -> Path:
    """Return the path to the current session's JSONL file."""
    path = Path.cwd() / ".xg" / "session.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save(messages: list) -> None:
    """Append all messages to the session file as one JSONL line."""
    with session_path().open("a") as f:
        f.write(json.dumps(messages_to_dict(messages)) + "\n")


def load() -> list:
    """Load the full conversation from the last line of the session file.

    Returns an empty list if no session exists.
    """
    path = session_path()
    if not path.exists():
        return []
    last_line = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return []
    return messages_from_dict(json.loads(last_line))
