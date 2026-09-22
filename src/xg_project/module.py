"""Context modules: a folder's declaration of what it offers as context.

A **context module** is any folder containing ``.xg/module.json``. The folder is
the anchor — "xg will act on this folder" means the module's own directory — and
the manifest says what the module offers and what it needs.

.. code-block:: json

    {
      "name": "graph",
      "description": "The node registry, the routing policy, and the workflows.",
      "expose": ["*.py", "read_tree.py"],
      "import": ["../read"]
    }

- ``name`` — what to call the module in the state and to a model. Optional; it
  defaults to the folder's name.
- ``description`` — required, and required to be a sentence. It is the routing
  signal: this is what a model reads when it decides whether the module's context
  is the context the request is about.
- ``expose`` — required. What the module contributes as context: paths and globs,
  relative to the module's own folder. ``"."`` means the whole folder. A leading
  ``..`` is allowed and reaches upward, because a module that needs the project's
  ``pyproject.toml`` should be able to say so.
- ``import`` — optional. Other modules whose exposed context this module also
  needs, named by their folder, likewise relative to this module's folder. The
  imported module's ``expose`` is added to the scope; the import is transitive,
  and a cycle is tolerated rather than chased.

Exposure is a **bound, not a hint**. Once a module is selected, the files it and
its imports expose are the only files the rest of the workflow may read: a file
outside that set is never opened, never shown to a model, and never a candidate
for an edit. The declaration is enforced by the read itself — see
``read_tree(include=...)`` — so there is no path around it and no second place
that has to agree about what is in scope.

Unknown keys are refused rather than ignored. A misspelled ``expose`` would
otherwise read as "exposes nothing", and a silently empty context is much harder
to notice than a manifest that says so.

Discovery is deterministic: the ignore rules decide which folders are walked at
all, manifests are visited in sorted order, and an unreadable or malformed
manifest becomes a line in ``Modules.problems`` instead of failing the run. One
bad manifest costs its own module, not the graph.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from xg_project.read import XG_DIR as MODULE_DIR
from xg_project.read import candidates, is_bookkeeping

MANIFEST = "module.json"

KEYS = frozenset({"name", "description", "expose", "import"})
"""Exactly the keys a manifest may carry. Anything else is a refusal."""


@dataclass(frozen=True)
class Module:
    """One folder's declaration of what it exposes as context."""

    path: Path
    """The module's folder, resolved: the folder the manifest sits in ``.xg/`` of."""

    name: str
    description: str
    expose: tuple[str, ...]
    imports: tuple[str, ...] = ()

    def id(self, root: Path) -> str:
        """The module's name for a run rooted at ``root``: its path from there.

        A path rather than the manifest's ``name``, because two modules may
        legitimately call themselves the same thing and a path cannot. The root
        module's id is ``"."``.
        """
        return self.path.relative_to(root).as_posix()

    def brief(self, root: Path) -> str:
        """What this module is, as one text for a model to judge.

        The description comes first because it is the part that answers the
        question being asked; the exposures follow as evidence for it.
        """
        lines = [f"{self.id(root)}: {self.description}"]
        if self.expose:
            lines.append("exposes " + ", ".join(self.expose))
        if self.imports:
            lines.append("imports " + ", ".join(self.imports))
        return "\n".join(lines)


@dataclass(frozen=True)
class Modules:
    """Every module found below one root, and every manifest that could not be used."""

    root: Path
    found: tuple[Module, ...] = ()
    problems: tuple[str, ...] = ()
    """``"path (reason)"`` for each manifest that was found and could not be used."""

    def by_path(self) -> dict[Path, Module]:
        """The modules keyed by folder, for resolving an ``import``."""
        return {module.path: module for module in self.found}

    def briefs(self) -> dict[str, str]:
        """Module id -> its brief, shaped for the Jev question set."""
        return {module.id(self.root): module.brief(self.root) for module in self.found}


@dataclass(frozen=True)
class Scope:
    """The context a selection admits: which modules, and exactly which files.

    ``files`` is root-relative paths, and it is the authoritative answer to
    "what may be read". ``missing`` keeps what the manifests asked for and did not
    get — an expose glob that matched nothing, an import that named no module —
    so a declaration that has quietly stopped working is visible in the state
    rather than merely having no effect.
    """

    modules: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        """Whether the scope admits nothing, which is different from no scope at all."""
        return not self.files


def _strings(value: object, key: str, *, required: bool) -> tuple[str, ...]:
    """Read a manifest's list-of-strings entry, refusing anything else."""
    if value is None:
        if required:
            raise ValueError(f"{key!r} is required")
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{key!r} must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key!r} entries must be non-empty strings")
        items.append(item.strip())
    if required and not items:
        raise ValueError(f"{key!r} must name at least one thing")
    return tuple(items)


def parse(data: object, *, directory: Path) -> Module:
    """Validate one manifest object into a :class:`Module`.

    Raises :class:`TypeError` when a value is the wrong kind of thing and
    :class:`ValueError` when it is the right kind and unusable, which is the
    split a reader of the refusal will care about. Both mean the same thing to
    the caller: this manifest cannot be trusted to describe a context.
    """
    if not isinstance(data, Mapping):
        raise TypeError("the manifest is not a JSON object")
    unknown = sorted(set(data) - KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) {', '.join(unknown)}; known keys are {', '.join(sorted(KEYS))}")

    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("'description' must be a non-empty string")
    name = data.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("'name' must be a non-empty string when it is given")

    return Module(
        path=directory,
        name=name.strip() if isinstance(name, str) else directory.name,
        description=description.strip(),
        expose=_strings(data.get("expose"), "expose", required=True),
        imports=_strings(data.get("import"), "import", required=False),
    )


def discover(root: str | Path = ".") -> Modules:
    """Find every context module below ``root``, in sorted order.

    The walk is the reader's own, so a folder ``.xgignore`` excludes is not
    searched for modules either: a module inside a folder nothing may read could
    never contribute a file, and reporting it would only offer a context that is
    entirely filtered out.
    """
    resolved = Path(root).resolve()
    found: list[Module] = []
    problems: list[str] = []

    for relpath, path in candidates(resolved):
        if path.name != MANIFEST or path.parent.name != MODULE_DIR:
            continue
        # The module is the folder the manifest sits beside, one level above the
        # .xg directory that holds it.
        directory = path.parent.parent
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            problems.append(f"{relpath} (not UTF-8 text)")
            continue
        except json.JSONDecodeError as error:
            problems.append(f"{relpath} (not valid JSON: {error.msg})")
            continue
        except OSError as error:
            problems.append(f"{relpath} (read failed: {error})")
            continue
        try:
            found.append(parse(data, directory=directory))
        except (TypeError, ValueError) as error:
            problems.append(f"{relpath} ({error})")

    return Modules(root=resolved, found=tuple(found), problems=tuple(problems))


def _matches(directory: Path, pattern: str) -> list[Path]:
    """The files one ``expose`` entry names.

    A pattern that names a directory means the directory and everything below it,
    because that is what a reader writing ``"expose": ["src/"]`` means by it. A
    pattern that is not a directory is globbed, so ``"*.py"`` and
    ``"graph/**/*.py"`` are the caller's choice rather than a special case here.

    ``.xg`` is left out whatever the pattern says. A manifest is an instruction
    about the read, so ``"expose": ["."]`` means the folder's content and not the
    bookkeeping that describes it.
    """
    target = directory / pattern
    if target.is_dir():
        found = [path for path in target.rglob("*") if path.is_file()]
    else:
        found = [path for path in directory.glob(pattern) if path.is_file()]
    return [
        path
        for path in found
        if not is_bookkeeping(path.relative_to(directory).as_posix())
    ]


def scope(modules: Modules, selected: Module) -> Scope:
    """Resolve a selected module, and everything it imports, into a file set.

    The files are filtered through the reader's own candidate walk, so a declared
    file that ``.gitignore`` or ``.xgignore`` excludes is not in scope: an
    exposure says what the module offers, and the ignores still decide what may
    be read. The two compose, with the ignores winning, because a rule that
    removes a file is an instruction about reading and exposure is a statement
    about the module.
    """
    allowed = {path: relpath for relpath, path in candidates(modules.root)}
    by_path = modules.by_path()

    seen: set[Path] = set()
    order: list[str] = []
    files: set[str] = set()
    missing: list[str] = []

    queue: list[Module] = [selected]
    while queue:
        module = queue.pop(0)
        if module.path in seen:
            continue
        seen.add(module.path)
        module_id = module.id(modules.root)
        order.append(module_id)

        for pattern in module.expose:
            matched = [path for path in _matches(module.path, pattern) if path in allowed]
            if not matched:
                missing.append(f"{module_id}: exposes {pattern!r}, which matched nothing")
                continue
            files.update(allowed[path] for path in matched)

        for target in module.imports:
            imported = by_path.get((module.path / target).resolve())
            if imported is None:
                missing.append(f"{module_id}: imports {target!r}, which is not a module")
                continue
            queue.append(imported)

    return Scope(modules=tuple(order), files=tuple(sorted(files)), missing=tuple(missing))
