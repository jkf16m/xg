"""xg_project.llm — the generative layer.

The graph reaches a language model only at the last moment, to fill in
generative information: the path of a new file, and that file's contents. No
tools are bound, and the model never decides where the turn goes. Everything
before these calls is a Jev decision or deterministic code.

OpenRouter is reached through ``langchain_openrouter.ChatOpenRouter``. The key
is read from ``pass show pi/openrouter``, and the model from the ``model`` key
of the composed ``.xg`` settings, falling back to :data:`DEFAULT_MODEL`.

Public API
----------
    build_llm(model=None, *, root=None) -> ChatOpenRouter
    complete(llm, system, prompt) -> str
    propose_path(request, *, existing, llm) -> str
    generate_file(request, path, window, *, llm) -> str
    parse_path(text) -> str
    safe_relative(path, root) -> str | None
    strip_fence(text) -> str
"""

import re
import subprocess
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from openrouter.errors import OpenRouterError

from xg_project.config import resolve

_LABEL = re.compile(r"^(?:path|file|filename|new file)\s*[:=]\s*", re.IGNORECASE)
"""Strips a leading ``path:`` style label from a model-proposed path."""

_PATH_CHARS = re.compile(r"^[^\s<>:\"|?*]+$")
"""A conservative path shape: no whitespace, no prose punctuation."""

PATH_MAX = 200
"""Longer than any sane repository path, and long enough to be prose."""

DEFAULT_MODEL = "@preset/mimo"
"""Used when no ``model`` key is set in the composed ``.xg`` settings."""

KEY_ENTRY = "pi/openrouter"
"""The ``pass`` entry holding the OpenRouter API key."""

PATH_SYSTEM = """You plan file paths for a coding agent.

Given a request and the files that already exist, reply with ONLY the
repository-relative path of the single new file to create. Use forward slashes.
The path must not already exist. No prose, no code fences, no explanation."""

FILE_SYSTEM = """You write files for a coding agent.

You are given a request and a context window of existing files, read whole and
ordered by modification time. Reply with the complete contents of the new file
and nothing else. No prose before or after, and no code fences."""


class LlmError(RuntimeError):
    """A provider-side failure (auth, rate limit, outage) the caller can retry."""


def get_api_key() -> str:
    """Read the OpenRouter key from the password store."""
    return subprocess.run(  # noqa: S603
        ["pass", "show", KEY_ENTRY],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def default_model(root: Path | None = None) -> str:
    """The configured generative model, or :data:`DEFAULT_MODEL`."""
    return resolve(root or Path.cwd()).model or DEFAULT_MODEL


def build_llm(model: str | None = None, *, root: Path | None = None) -> ChatOpenRouter:
    """Build the OpenRouter chat client."""
    return ChatOpenRouter(
        model=model or default_model(root),
        openrouter_api_key=get_api_key(),
    )


def complete(llm: ChatOpenRouter, system: str, prompt: str) -> str:
    """Run one non-streaming turn and return its text."""
    try:
        response = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
    except OpenRouterError as exc:
        raise LlmError(str(exc)) from exc
    content = response.content
    return content if isinstance(content, str) else str(content)


def parse_path(text: str) -> str:
    """Pull a repository-relative path out of a model reply.

    Keeps the first line that looks like a path and strips code fences,
    surrounding quotes, and a ``path:`` label. Returns ``""`` when nothing
    usable is there, so the caller reroutes instead of writing to a guessed
    path.

    The shape test is deliberately strict. A model that refuses, or explains
    instead of answering, must not have its prose mistaken for a filename: a
    reply of "I cannot" would otherwise create a file called ``I cannot``.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            line = line[3:].strip()
        if line.endswith("```"):
            line = line[:-3].strip()
        line = _LABEL.sub("", line).strip().strip("\"'`").strip()
        if not line or len(line) > PATH_MAX:
            continue
        if not _PATH_CHARS.match(line):
            continue
        if "/" not in line and "." not in line:
            continue
        return line
    return ""


def safe_relative(path: str, root: Path) -> str | None:
    """Return ``path`` if it stays inside ``root``, otherwise ``None``.

    Rejects absolute paths and any path that escapes through ``..``, so a model
    cannot write outside the repository.
    """
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    base = Path(root).resolve()
    try:
        (base / candidate).resolve().relative_to(base)
    except ValueError:
        return None
    return candidate.as_posix()


def strip_fence(text: str) -> str:
    """Drop one wrapping markdown code fence, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines) + "\n"


def propose_path(request: str, *, existing: list[str], llm: ChatOpenRouter) -> str:
    """Ask for the path of the new file to create."""
    listing = "\n".join(existing) if existing else "(the repository is empty)"
    prompt = (
        f"Request:\n{request}\n\n"
        f"Files that already exist:\n{listing}\n\n"
        "New file path:"
    )
    return complete(llm, PATH_SYSTEM, prompt)


def generate_file(
    request: str, path: str, window: str, *, llm: ChatOpenRouter
) -> str:
    """Ask for the full contents of the new file."""
    prompt = (
        f"Request:\n{request}\n\n"
        f"Create this new file: {path}\n\n"
        f"Context window:\n{window}"
    )
    return complete(llm, FILE_SYSTEM, prompt)

