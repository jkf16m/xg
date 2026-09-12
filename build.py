#!/usr/bin/env python3
"""Project build script — runs linters.

Usage:
    uv run python build.py
"""

import subprocess
import sys


def main() -> int:
    print("=== ruff ===")
    result = subprocess.run(["uv", "run", "ruff", "check", "src/"])
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
