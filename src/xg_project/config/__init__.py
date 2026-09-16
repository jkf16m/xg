"""xg_project.config — shared configuration."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Global configuration.

    ``files`` is a module file allowlist. ``None`` means the directory declares
    no allowlist, so every discovered file in its subtree is kept.
    """

    session_path: Path | None = None
    use_gitignore: bool = True
    files: tuple[str, ...] | None = None


def load_project_config(root: Path) -> Config:
    """Load optional project settings from ``root/.xg/config.json``.

    A directory is an xg module when ``.xg/config.json`` exists; this loader is
    also the module manifest reader.
    """
    config_path = root / ".xg" / "config.json"
    if not config_path.is_file():
        return Config()

    data = json.loads(config_path.read_text(encoding="utf-8"))
    use_gitignore = data.get("use_gitignore", True)
    if not isinstance(use_gitignore, bool):
        raise ValueError(".xg/config.json: use_gitignore must be a boolean")

    files = data.get("files")
    if files is None:
        module_files = None
    elif isinstance(files, list) and all(isinstance(path, str) for path in files):
        module_files = tuple(files)
    else:
        raise ValueError(".xg/config.json: files must be an array of strings")
    return Config(use_gitignore=use_gitignore, files=module_files)
