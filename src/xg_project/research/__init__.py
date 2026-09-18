"""local_research — the forced graph step that finds relevant files.

Given a request, it lists the repository, asks Jev one ``Noul`` per candidate
file ("is this file needed for the request?"), and returns the files whose
probability clears a threshold. All questions are evaluated in parallel in one
``system_one`` call.

Scope is an outcome of this step, not an input to classification: the number of
files found is how far-reaching the request turned out to be.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from typesafe_sdk import Noul, TypeSafeClient, TypeSafeError

from xg_project.jev import JevError

MAX_CANDIDATES = 120
"""Cap on files asked about in one call; the rest are dropped by the prefilter."""

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
    relevance: dict[str, float]
    """Probability per candidate path."""
    candidates: int
    """How many files were asked about."""
    model: str


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
    """Keep the ``limit`` paths most lexically related to the request.

    A cheap guard for large repositories: an exact token match on the path is
    weak evidence, but it is enough to choose which files are worth asking Jev
    about when the repository is bigger than one call should carry.
    """
    if len(files) <= limit:
        return files
    wanted = _tokens(request)
    ranked = sorted(files, key=lambda path: (-len(wanted & _tokens(path)), len(path)))
    return sorted(ranked[:limit])


def _questions(files: list[str]) -> dict[str, Noul]:
    """One ``Noul`` per candidate file, all answered in one call."""
    return {
        f"file_{index}": Noul(
            instructions=(
                f"Is the repository file {path!r} needed to answer or carry out "
                "the request?"
            ),
            criteria={
                "true": "The file is likely to be read, changed, or used as evidence.",
                "false": "The file is unlikely to matter for this request.",
            },
        )
        for index, path in enumerate(files)
    }


def local_research(
    request: str,
    *,
    client: TypeSafeClient,
    model: str | None = None,
    root: Path | None = None,
    threshold: float = SELECT_THRESHOLD,
    max_candidates: int = MAX_CANDIDATES,
) -> Research:
    """Find the files relevant to ``request`` with one Jev call."""
    if not request.strip():
        raise ValueError("request must not be empty")

    root = (root or Path.cwd()).resolve()
    files = _prefilter(repo_files(root), request, max_candidates)
    if not files:
        return Research(request=request, files=[], relevance={}, candidates=0, model="")

    state: dict[str, object] = {"request": request, "repository": files}
    try:
        response = client.system_one(
            state=state, questions=_questions(files), model=model
        )
    except TypeSafeError as exc:
        raise JevError(str(exc)) from exc

    relevance = {
        path: response.nouls[f"file_{index}"].noul
        for index, path in enumerate(files)
    }
    selected = sorted(
        (path for path, probability in relevance.items() if probability >= threshold),
        key=lambda path: relevance[path],
        reverse=True,
    )
    return Research(
        request=request,
        files=selected,
        relevance=relevance,
        candidates=len(files),
        model=response.model,
    )
