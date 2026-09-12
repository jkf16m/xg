#!/usr/bin/env python3
"""Project build script — runs linters and type checkers.

Usage:
    uv run python build.py
"""

import subprocess
import sys


def run(name: str, cmd: list[str]) -> bool:
    """Run a command and return True if it succeeds."""
    print(f"\n=== {name} ===")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main() -> int:
    checks = [
        ("ruff", ["uv", "run", "ruff", "check", "src/"]),
        ("mypy", ["uv", "run", "mypy", "src/", "--ignore-missing-imports"]),
    ]

    results = []
    for name, cmd in checks:
        results.append(run(name, cmd))

    print("\n" + "=" * 40)
    if all(results):
        print("All checks passed.")
        return 0
    else:
        failed = [name for (name, _), ok in zip(checks, results) if not ok]
        print(f"Failed: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
