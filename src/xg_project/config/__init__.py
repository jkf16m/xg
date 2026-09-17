"""xg_project.config — shared configuration.

Two surfaces live under a ``.xg/`` directory:

``.xg/config.json``
    Settings that compose down the directory tree (global base → ancestors →
    leaf). Read with :func:`resolve`.

``.xg/module.json``
    A context-module declaration. Module declarations are boundaries: they
    never inherit from an enclosing module. Read with :func:`load_module`.

A ``.xg/`` directory describes the directory that contains it. Every relative
path in these files resolves against that parent (the module root), never
against ``.xg/`` itself.
"""

import json
from dataclasses import dataclass
from pathlib import Path

GLOBAL_CONFIG_PATH = Path.home() / ".xg" / "config.json"
XG_DIR_NAME = ".xg"
CONFIG_FILE_NAME = "config.json"
MODULE_FILE_NAME = "module.json"


@dataclass
class Config:
    """Composed settings for a directory.

    Settings inherit down the tree; see :func:`resolve`. Module-scoped values
    such as a file allowlist belong to :class:`Module`, not here.
    """

    session_path: Path | None = None
    use_gitignore: bool = True
    model: str | None = None
    patch_formatter: str | None = None


@dataclass(frozen=True)
class Module:
    """A context module rooted at ``root``.

    ``files`` is an allowlist of paths relative to ``root``. ``None`` means the
    module declares no allowlist and does not restrict its subtree. A module is
    a boundary, so it never inherits from an enclosing module.
    """

    root: Path
    files: tuple[str, ...] | None


def _read_json_object(path: Path) -> dict[str, object] | None:
    """Return the JSON object at ``path``, or ``None`` when it is absent."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: configuration must be a JSON object")
    return data


def _settings_from_data(data: dict[str, object], source: str) -> Config:
    """Build a Config from merged settings data, validating each field."""
    if "files" in data:
        raise ValueError(
            f"{source}: files is a module key; declare it in "
            f"{XG_DIR_NAME}/{MODULE_FILE_NAME}"
        )
    use_gitignore = data.get("use_gitignore", True)
    if not isinstance(use_gitignore, bool):
        raise ValueError(f"{source}: use_gitignore must be a boolean")
    model = data.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError(f"{source}: model must be a string")
    patch_formatter = data.get("patch_formatter")
    if patch_formatter is not None and not isinstance(patch_formatter, str):
        raise ValueError(f"{source}: patch_formatter must be a string")
    return Config(
        use_gitignore=use_gitignore, model=model, patch_formatter=patch_formatter
    )


def _module_from_data(root: Path, data: dict[str, object], source: str) -> Module:
    """Build a Module from a single module declaration's data."""
    files = data.get("files")
    if files is None:
        module_files = None
    elif isinstance(files, list) and all(isinstance(path, str) for path in files):
        module_files = tuple(files)
    else:
        raise ValueError(f"{source}: files must be an array of strings")
    return Module(root=root, files=module_files)


def resolve(path: Path) -> Config:
    """Return the composed settings that apply to ``path``.

    Settings are layered from least to most specific:

    1. ``$HOME/.xg/config.json`` — the global base.
    2. Ancestor ``.xg/config.json`` files, from the farthest ancestor down to
       the directory of ``path``.

    Each layer overrides the previous one per key; a key a nearer layer omits
    stays inherited. Walking upward stops after the first ``.xg/config.json``
    that sets ``"stopWalking": true`` — that layer is included, nothing above
    it is.

    ``path`` may be a file, in which case its parent directory is the starting
    point.
    """
    start = path.resolve()
    if start.is_file():
        start = start.parent

    layers: list[tuple[str, dict[str, object]]] = []
    global_data = _read_json_object(GLOBAL_CONFIG_PATH)
    if global_data is not None:
        layers.append((str(GLOBAL_CONFIG_PATH), global_data))

    ancestors: list[tuple[str, dict[str, object]]] = []
    for directory in [start, *start.parents]:
        config_path = directory / XG_DIR_NAME / CONFIG_FILE_NAME
        data = _read_json_object(config_path)
        if data is None:
            continue
        stop = data.get("stopWalking", False)
        if not isinstance(stop, bool):
            raise ValueError(f"{config_path}: stopWalking must be a boolean")
        ancestors.append((str(config_path), data))
        if stop:
            break
    layers.extend(reversed(ancestors))

    merged: dict[str, object] = {}
    for _, data in layers:
        merged.update(data)
    return _settings_from_data(merged, f"{XG_DIR_NAME}/{CONFIG_FILE_NAME}")


def load_module(root: Path) -> Module | None:
    """Read ``root/.xg/module.json`` as a module declaration.

    Returns ``None`` when the directory is not an xg module. The declaration is
    read alone: it does not inherit from ancestors or from the global base.
    """
    root = root.resolve()
    module_path = root / XG_DIR_NAME / MODULE_FILE_NAME
    data = _read_json_object(module_path)
    if data is None:
        return None
    return _module_from_data(root, data, str(module_path))
