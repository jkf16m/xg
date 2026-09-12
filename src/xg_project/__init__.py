"""Command-line entry point for xg."""

import argparse
import os
import shutil
import sys

from rich.console import Console


def _show_help() -> None:
    Console().print(
        "[bold]xg[/bold] - AI coding agent\n\n"
        "Usage:\n"
        "  [cyan]xg --start[/cyan]   Start xg inside a PTY\n"
        "  [cyan]xg --pty[/cyan]     Start xg inside a PTY"
    )


def _run_pty() -> None:
    """Run the user's normal interactive shell through a PTY."""
    import pexpect

    shell = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
    env = os.environ.copy()
    env["XG_PTY"] = "1"

    # pexpect.spawn creates the PTY, starts the shell on its slave side, and
    # returns a handle for the PTY master. interact() then directly bridges
    # that master with this terminal: no prompts or input are intercepted.
    child = pexpect.spawn(
        shell,
        ["-i"],
        env=env,
        cwd=os.getcwd(),
        timeout=None,
    )
    try:
        child.interact(escape_character=None)
    finally:
        child.close(force=True)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True, prog="xg")
    parser.add_argument("--start", "-s", action="store_true")
    parser.add_argument("--pty", action="store_true")
    args = parser.parse_args()

    if args.start or args.pty:
        os.environ["XG_PTY"] = "1"

    if os.environ.get("XG_PTY") == "1":
        _run_pty()
    else:
        _show_help()


if __name__ == "__main__":
    main()
