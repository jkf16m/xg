"""Private: project file discovery."""

import subprocess
from pathlib import Path

from pathspec.gitignore import GitIgnoreSpec

from xg_project.config import MODULE_FILE_NAME, XG_DIR_NAME, Module, load_module


def load_context_module(root: Path) -> Module | None:
    """Load ``root/.xg/module.json`` as a module declaration.

    Returns ``None`` when the directory is not an xg module.
    """
    return load_module(root)


def project_files(
    root: Path | None = None,
    *,
    use_gitignore: bool = True,
) -> list[Path]:
    """Return project files in deterministic (mtime, path) order."""
    root = (root or Path.cwd()).resolve()
    candidates = [
        path for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(root).parts
        and "__pycache__" not in path.relative_to(root).parts
    ]

    # A module declaration replaces recursive discovery for that module root.
    # The nearest module declaration wins for nested modules.
    modules = []
    for module_path in root.rglob(f"{XG_DIR_NAME}/{MODULE_FILE_NAME}"):
        module_root = module_path.parent.parent.resolve()
        module = load_context_module(module_root)
        if module is not None:
            modules.append(module)
    if modules:
        filtered = []
        for path in candidates:
            matching = [module for module in modules if module.root in path.parents]
            if not matching:
                filtered.append(path)
                continue
            module = max(matching, key=lambda item: len(item.root.parts))
            if module.files is None:
                # No allowlist: the module does not restrict its subtree.
                filtered.append(path)
                continue
            relative = path.relative_to(module.root).as_posix()
            if relative in module.files:
                filtered.append(path)
        candidates = filtered

    relative_paths = [path.relative_to(root).as_posix() for path in candidates]

    ignored: set[str] = set()
    if relative_paths and use_gitignore:
        ignored.update(_git_ignored(root, relative_paths))

    if relative_paths:
        # .xgignore is an xg-specific ignore file. Use pathspec's GitIgnoreSpec
        # so this works outside Git repositories and does not invoke Git again.
        xgignore = root / ".xgignore"
        if xgignore.is_file():
            patterns = xgignore.read_text(encoding="utf-8").splitlines()
            spec = GitIgnoreSpec.from_lines(patterns)
            ignored.add(".xgignore")
            ignored.update(path for path in relative_paths if spec.match_file(path))

    files = [
        path for path in candidates
        if path.relative_to(root).as_posix() not in ignored
    ]
    # Pre-compute mtime to avoid repeated stat() calls during sort.
    mtime: dict[Path, int] = {p: p.stat().st_mtime_ns for p in files}
    return sorted(
        files,
        key=lambda p: (mtime[p], p.relative_to(root).as_posix()),
    )


def _git_ignored(root: Path, relative_paths: list[str]) -> set[str]:
    """Return paths ignored by the repository's Git rules."""
    command = ["git", "-C", str(root), "check-ignore", "--stdin", "-z", "--no-index"]
    try:
        result = subprocess.run(  # noqa: S603
            command,  # noqa: S607
            input="\0".join(relative_paths) + "\0",
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    return set(filter(None, result.stdout.split("\0")))
