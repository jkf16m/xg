"""Private: OpenRouter LLM configuration."""

import subprocess
from pathlib import Path

from langchain_openrouter import ChatOpenRouter

from xg_project.config import resolve

DEFAULT_MODEL = "@preset/mimo"
SYSTEM = """You are xg, a coding agent in a 1:1 human-guided loop.
There is exactly one agent turn for each human message. Do not invent a task,
do not stop because of a stop reason, and do not ask for confirmation before
using tools. The launch context contains the current project files in mtime
order. A read call appends that file to the bottom of context. Respond
with a concise summary when your turn is complete."""
SYSTEM_FILE = Path(".xg") / "SYSTEM.md"


def model(root: Path | None = None) -> str:
    """Return the configured model for a project root.

    The ``model`` key is read from the composed ``.xg/config.json`` settings,
    with ``$HOME/.xg/config.json`` as the base. Falls back to the built-in
    default when no layer sets one.
    """
    return resolve(root or Path.cwd()).model or DEFAULT_MODEL


def system_prompt(root: Path | None = None) -> str:
    """Return the system prompt for a project root.

    ``<root>/.xg/SYSTEM.md`` replaces the built-in default when it exists.
    """
    path = (root or Path.cwd()).resolve() / SYSTEM_FILE
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return SYSTEM


def get_api_key() -> str:
    return subprocess.run(
        ["pass", "show", "pi/openrouter"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def get_llm(root: Path | None = None, **kwargs) -> ChatOpenRouter:
    return ChatOpenRouter(model=model(root), openrouter_api_key=get_api_key(), **kwargs)
