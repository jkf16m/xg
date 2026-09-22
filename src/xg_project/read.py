"""A deterministic read of a directory tree.

What xg_collect does: walk a root, read every file it is allowed to read, and
return a mapping of path to content. Deterministic means the same tree produces
the same mapping in the same order, so a run can be reproduced from the path
alone.

What is read is decided by two ignore files, and by nothing else:

- ``.gitignore`` — the repository's own rules, honoured as git honours them.
- ``.xgignore`` — xg's own additions, for files git tracks but an agent should
  not read: local secrets, vendored dumps, anything large and irrelevant.

The matching is done by ``pathspec.GitIgnoreSpec`` rather than by hand. Git
ignore syntax has negation, anchoring, directory-only patterns and ``**``, and a
hand-rolled subset would quietly disagree with git about the repository it is
pointing at. ``.xgignore`` lines are appended after ``.gitignore`` lines so, as
in git, the last matching pattern wins and the more specific file has the final
say.

A nested ignore file applies to its own directory and below. Its patterns are
rewritten into root-relative form once, when the file is read, so every pattern
is matched against the same root-relative path and the walk stays a single
comparison.

Two things are always skipped and are not configurable: the ``.git`` directory,
which is git's own storage rather than project content, and the ignore files
themselves, which are instructions about the read rather than part of it.
Files that are not UTF-8 text, and files above ``max_bytes``, are recorded in
``skipped`` rather than silently vanishing: a caller can tell "ignored" from
"could not be read".

``read_tree`` can also be bounded by an ``include`` list — gitignore-syntax
patterns, rooted at ``root``. That is how a context module's declaration becomes
a hard bound on the read: a file the module does not expose is not merely
discouraged, it is never opened. The bound composes with the ignores rather than
replacing them, so ``.xgignore`` still removes a file a module declared.

``.xg`` is xg's own directory. A module manifest says what a folder exposes,
which is an instruction about the read in the same sense an ignore file is, so
``read_tree`` never returns one as content; :mod:`xg_project.module` is what
reads them, and it does so through the same walk.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pathspec

GITIGNORE = ".gitignore"
XGIGNORE = ".xgignore"
GIT_DIR = ".git"
XG_DIR = ".xg"
"""Where a folder declares what it exposes as context; see :mod:`xg_project.module`."""

DEFAULT_MAX_BYTES = 200_000
"""Files above this are skipped rather than read.

Not a correctness bound: a minified bundle or a lockfile is legitimate project
content that is useless as context and expensive as a Jev question. The value is
a starting point, not a law.
"""


@dataclass(frozen=True)
class Tree:
    """The result of one read.

    ``files`` is path -> content, ordered by path. ``skipped`` names what was
    found but not read, in the form ``"path (reason)"``, so a caller can report
    why the mapping is smaller than the directory.
    """

    root: Path
    files: dict[str, str]
    skipped: tuple[str, ...] = ()


def _relpath(path: Path, root: Path) -> str:
    """The POSIX-style path of ``path`` below ``root``.

    POSIX separators because the same string is used for a state key, a
    question name, and an ignore match, and only one of those is filesystem
    specific.
    """
    return path.relative_to(root).as_posix()


def _prefix_patterns(directory: str, lines: Iterable[str]) -> Iterator[str]:
    """Rewrite one ignore file's patterns into root-relative form.

    ``directory`` is the POSIX path of the file's directory, relative to the
    root, or ``""`` for the root itself. Git's rules about what a pattern without
    a slash means are then reproduced:

    - no slash and not anchored: matches at any depth below ``directory``, so it
      gains a ``directory/**/`` prefix.
    - a slash, or a leading ``/``: anchored to ``directory``, so it gains a
      ``directory/`` prefix and the leading slash is dropped.
    """
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        body = line[1:].strip() if negated else line
        if not body:
            continue

        anchored = body.startswith("/")
        body = body.lstrip("/")
        if directory:
            if anchored or "/" in body.rstrip("/"):
                body = f"{directory}/{body}"
            else:
                body = f"{directory}/**/{body}"
        yield ("!" if negated else "") + body


class _Ignores:
    """The accumulated root-relative patterns, rematched as the walk descends.

    The spec is rebuilt only when a directory contributes new patterns, so the
    common repository — one ignore file at the root — costs one build.
    """

    def __init__(self) -> None:
        self._patterns: list[str] = []
        self._spec = pathspec.GitIgnoreSpec.from_lines([])

    def add(self, directory: str, lines: Iterable[str]) -> None:
        patterns = list(_prefix_patterns(directory, lines))
        if patterns:
            self._patterns.extend(patterns)
            self._spec = pathspec.GitIgnoreSpec.from_lines(self._patterns)

    def match(self, relpath: str) -> bool:
        return self._spec.match_file(relpath)


def _read_ignore_file(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def is_bookkeeping(relpath: str) -> bool:
    """Whether a path is xg's own bookkeeping rather than project content.

    Public because two callers need the same answer: the reader, which must not
    return a manifest as content, and module resolution, which must not expose
    one.
    """
    return XG_DIR in relpath.split("/")


def candidates(
    root: str | Path = ".", *, include: Sequence[str] | None = None
) -> Iterator[tuple[str, Path]]:
    """Every file below ``root`` that the ignore rules allow, in sorted order.

    Yields ``(relpath, path)``. The walk is shared by the reader and by module
discovery, so there is one definition of what "the project's files" means
    rather than two that agree until they do not.

    ``include``, when given, holds gitignore-syntax patterns rooted at ``root``.
    A file that matches none of them is not a candidate, which is how an exposed
    context is enforced: the files outside it are never offered to anyone.

    The ignore files are never candidates. The ``.xg`` directory is walked, so
    that manifests can be found, but the files in it are left for the caller to
    identify by path.
    """
    root = Path(root).resolve()
    ignores = _Ignores()
    bounded = pathspec.GitIgnoreSpec.from_lines(include) if include is not None else None

    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        directory = "" if here == root else _relpath(here, root)

        # Ignore files in this directory apply from here down, so they are read
        # before anything below them is considered. .xgignore comes second, so
        # its patterns are the last word.
        ignores.add(directory, _read_ignore_file(here / GITIGNORE))
        ignores.add(directory, _read_ignore_file(here / XGIGNORE))

        dirnames[:] = sorted(
            name
            for name in dirnames
            if name != GIT_DIR and not ignores.match(_relpath(here / name, root))
        )

        for name in sorted(filenames):
            if name in (GITIGNORE, XGIGNORE) or name == GIT_DIR:
                continue
            path = here / name
            relpath = _relpath(path, root)
            if ignores.match(relpath):
                continue
            if bounded is not None and not bounded.match_file(relpath):
                continue
            yield relpath, path


def read_tree(
    root: str | Path = ".",
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    include: Sequence[str] | None = None,
) -> Tree:
    """Read every readable file below ``root``, honouring the ignore files.

    ``include`` bounds what is read to the files a context module declared; with
    no include, the whole non-ignored tree is read. Files are visited in sorted
    order at every level, so the mapping and the skipped list are stable between
    runs.
    """
    resolved = Path(root).resolve()
    files: dict[str, str] = {}
    skipped: list[str] = []

    for relpath, path in candidates(resolved, include=include):
        if is_bookkeeping(relpath):
            continue
        try:
            size = path.stat().st_size
        except OSError as error:
            skipped.append(f"{relpath} (stat failed: {error})")
            continue
        if size > max_bytes:
            skipped.append(f"{relpath} ({size} bytes, over {max_bytes})")
            continue

        try:
            files[relpath] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped.append(f"{relpath} (not UTF-8 text)")
        except OSError as error:
            skipped.append(f"{relpath} (read failed: {error})")

    return Tree(root=resolved, files=dict(sorted(files.items())), skipped=tuple(skipped))
