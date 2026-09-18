"""local_research — the forced graph step that finds relevant files.

Given a request, it lists the repository, loads the full text of each candidate
file, and puts those contents in the Jev state as a JSON object keyed by path::

    state = {"request": ..., "files": {path: text, ...}}

One ``Noul`` question is then asked per candidate ("is this file needed?"), all
in one ``system_one`` call, and the files whose probability clears a threshold
are selected. Selection is decided from file *contents*, not their names.

``Choice`` is single-select — its answer is one label — so multi-file selection
is expressed as independent ``Noul`` questions, one per file.

Scope is an outcome of this step, not an input to classification: the number of
files found is how far-reaching the request turned out to be.
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from typesafe_sdk import Noul, TypeSafeClient, TypeSafeError

from xg_project.jev import JevError

MAX_CANDIDATES = 120
"""Cap on files asked about in one call; above this, the prefilter trims."""

MAX_FILE_BYTES = 200_000
"""Files larger than this are skipped. Lockfiles and generated artifacts are
large and carry little signal, and one of them can cost more tokens than the
rest of the repository combined (this repo's ``uv.lock`` is 262 KB)."""

SELECT_THRESHOLD = 0.5
"""Minimum ``Noul`` probability for a file to be selected."""

_TEXT = re.compile(r"[a-z0-9]+")
_SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)


@dataclass(frozen=True)
class Research:
    """The outcome of one research pass."""

    request: str
    files: list[str]
    """Selected files, most relevant first."""
    contents: dict[str, str] = field(default_factory=dict)
    """Full text of the selected files, keyed by relative path."""
    relevance: dict[str, float] = field(default_factory=dict)
    """Probability per candidate path."""
    candidates: int = 0
    """How many files were sent to Jev."""
    listed: int = 0
    """How many files the repository listing returned."""
    model: str = ""


def repo_files(root: Path) -> list[str]:
    """List repository files relative to ``root``.

    Prefers ``git ls-files`` (which respects ``.gitignore`` and skips untracked
    files); falls back to a directory walk with the usual caches skipped.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "ls-files"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        listed = [line for line in result.stdout.splitlines() if line.strip()]
        if listed:
            return sorted(listed)
    except (OSError, subprocess.CalledProcessError):
        pass
    return _walk(root)


def _walk(root: Path) -> list[str]:
    files: list[str] = []
    for path in root.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            files.append(str(path.relative_to(root)))
    return sorted(files)


def _tokens(text: str) -> set[str]:
    return set(_TEXT.findall(text.lower()))


def _prefilter(files: list[str], request: str, limit: int) -> list[str]:
    """Trim a large listing to ``limit`` paths by lexical overlap.

    This only runs when the repository is bigger than one call should carry.
    With the listing under the limit, every file is sent, so selection is never
    name-based.
    """
    if len(files) <= limit:
        return files
    wanted = _tokens(request)
    ranked = sorted(files, key=lambda path: (-len(wanted & _tokens(path)), len(path)))
    return sorted(ranked[:limit])


def _read_text(path: Path) -> str | None:
    """Read a file as UTF-8 text, or ``None`` when it is binary or unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _load(
    root: Path,
    files: list[str],
    request: str,
    limit: int,
    max_file_bytes: int,
) -> list[tuple[str, str]]:
    """Read the full text of each candidate, skipping unreadable and huge files."""
    loaded: list[tuple[str, str]] = []
    for relative in _prefilter(files, request, limit):
        text = _read_text(root / relative)
        if text is None or len(text.encode("utf-8")) > max_file_bytes:
            continue
        loaded.append((relative, text))
    return loaded


def _questions(files: list[str]) -> dict[str, Noul]:
    """One ``Noul`` per candidate file, all answered in one call.

    The file's full text is in the shared state under its path, so the answer
    is grounded in contents rather than the filename.
    """
    return {
        f"file_{index}": Noul(
            instructions=(
                f"Is the repository file {path!r} needed to answer or carry out "
                "the request? Its full text is in the request state under that path."
            ),
            criteria={
                "true": "The file is likely to be read, changed, or used as evidence.",
                "false": "The file is unlikely to matter for this request.",
            },
        )
        for index, path in enumerate(files)
    }


def _collect(
    request: str,
    root: Path,
    max_candidates: int,
    max_file_bytes: int,
) -> tuple[int, list[tuple[str, str]]]:
    """List the repository and load the full text of each candidate."""
    listed = repo_files(root)
    return len(listed), _load(root, listed, request, max_candidates, max_file_bytes)


def build_research_state(
    request: str,
    *,
    root: Path | None = None,
    max_candidates: int = MAX_CANDIDATES,
    max_file_bytes: int = MAX_FILE_BYTES,
) -> dict[str, object]:
    """The exact Jev state sent by :func:`local_research`.

    ``{"request": ..., "files": {path: full text, ...}}`` — the file map is a
    JSON object keyed by path, and the SDK escapes the text when serializing.
    Exposed so the state can be inspected without sending it.
    """
    if not request.strip():
        raise ValueError("request must not be empty")
    root = (root or Path.cwd()).resolve()
    _, candidates = _collect(request, root, max_candidates, max_file_bytes)
    return {"request": request, "files": dict(candidates)}


def local_research(
    request: str,
    *,
    client: TypeSafeClient,
    model: str | None = None,
    root: Path | None = None,
    threshold: float = SELECT_THRESHOLD,
    max_candidates: int = MAX_CANDIDATES,
    max_file_bytes: int = MAX_FILE_BYTES,
) -> Research:
    """Find the files relevant to ``request`` with one Jev call."""
    if not request.strip():
        raise ValueError("request must not be empty")

    root = (root or Path.cwd()).resolve()
    listed, candidates = _collect(request, root, max_candidates, max_file_bytes)
    if not candidates:
        return Research(request=request, listed=listed)

    paths = [path for path, _ in candidates]
    contents = dict(candidates)
    state: dict[str, object] = {"request": request, "files": contents}
    try:
        response = client.system_one(
            state=state, questions=_questions(paths), model=model
        )
    except TypeSafeError as exc:
        raise JevError(str(exc)) from exc

    relevance = {
        path: response.nouls[f"file_{index}"].noul
        for index, path in enumerate(paths)
    }
    selected = sorted(
        (path for path, probability in relevance.items() if probability >= threshold),
        key=lambda path: relevance[path],
        reverse=True,
    )
    return Research(
        request=request,
        files=selected,
        contents={path: contents[path] for path in selected},
        relevance=relevance,
        candidates=len(paths),
        listed=listed,
        model=response.model,
    )
