"""Private: project file discovery."""

import subprocess
from pathlib import Path


def project_files(root: Path | None = None) -> list[Path]:
    """Return non-ignored project files in deterministic (mtime, path) order."""
    root = (root or Path.cwd()).resolve()
    candidates = [
        path for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(root).parts
        and "__pycache__" not in path.relative_to(root).parts
    ]
    relative_paths = [path.relative_to(root).as_posix() for path in candidates]

    ignored: set[str] = set()
    if relative_paths:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "check-ignore", "--stdin", "-z"],
                input="\0".join(relative_paths) + "\0",
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode in (0, 1):
                ignored = set(filter(None, result.stdout.split("\0")))
        except OSError:
            pass

    files = [
        path for path in candidates
        if path.relative_to(root).as_posix() not in ignored
    ]
    return sorted(files, key=lambda p: (p.stat().st_mtime_ns, p.relative_to(root).as_posix()))
