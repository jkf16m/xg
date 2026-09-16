"""Command-line entry point for xg."""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


def _show_info() -> None:
    """Show how the current working directory becomes LLM context."""
    from xg_project.config import load_project_config
    from xg_project.llm._api import MODEL, SYSTEM_FILE, system_prompt
    from xg_project.llm._context import project_files
    from xg_project.llm._tools import TOOLS
    from xg_project.session import DEFAULT_DB_PATH

    root = Path.cwd().resolve()
    project_config = load_project_config(root)
    files = project_files(root, use_gitignore=project_config.use_gitignore)
    console = Console()

    console.print(f"[bold]root[/bold] {root}")
    console.print(f"[bold]model[/bold] {MODEL}")
    console.print("[bold]provider[/bold] OpenRouter")
    console.print(f"[bold]use_gitignore[/bold] {project_config.use_gitignore}")
    console.print(f"[bold]session database[/bold] {DEFAULT_DB_PATH}")
    console.print()

    console.print("[bold]context construction[/bold]")
    console.print("1. A system message is prepended to every LLM request.")
    console.print("2. Session messages are loaded in database insertion order.")
    console.print("3. New user messages are appended to the session.")
    console.print("4. Each request is sent as: system message + conversation messages.")
    console.print("5. Tool results are appended after approved tool execution.")
    console.print()
    console.print("[bold]file discovery[/bold]")
    console.print("Files are recursively discovered, excluding .git and __pycache__,")
    console.print(
        "then module file allowlists, Git-ignored files, and .xgignore patterns "
        "are applied; the remainder is ordered by mtime."
    )

    table = Table("#", "path", "bytes", "mtime")
    for number, path in enumerate(files, 1):
        stat = path.stat()
        table.add_row(
            str(number),
            str(path.relative_to(root)),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        )
    console.print(table)
    console.print(f"[bold]files in initial file context:[/bold] {len(files)}")
    console.print()

    console.print("[bold]tools bound to the model[/bold]")
    for tool in TOOLS:
        console.print(f"- {tool.name}: {tool.description.splitlines()[0]}")
    console.print()
    console.print("[bold]system prompt[/bold]")
    system_file = root / SYSTEM_FILE
    source = system_file if system_file.is_file() else "built-in default"
    console.print(f"source: {source}")
    console.print(system_prompt(root))


def _show_help() -> None:
    Console().print(
        "[bold]xg[/bold] - AI coding agent harness\n\n"
        "Usage:\n"
        "  [cyan]xg[/cyan]           Start a new conversation\n"
        "  [cyan]xg -c[/cyan]        Continue the last conversation\n"
        "\n"
        "In the REPL:\n"
        "  [dim]Empty prompt = send turn[/dim]\n"
        "  [dim]ESC = exit[/dim]\n"
    )


def _run_pty(continue_session: bool) -> None:
    """Run the xg interface in a PTY."""
    import pexpect

    env = os.environ.copy()
    env["XG_PTY"] = "1"
    if continue_session:
        env["XG_CONTINUE"] = "1"

    child = pexpect.spawn(
        sys.executable,
        ["-m", "xg_project.interface"],
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
    parser.add_argument("-c", "--continue-session", action="store_true",
                        help="continue the last conversation")
    parser.add_argument("--info", action="store_true",
                        help="show context construction and model information")
    args = parser.parse_args()

    if args.info:
        _show_info()
        return

    os.environ["XG_PTY"] = "1"
    if args.continue_session:
        os.environ["XG_CONTINUE"] = "1"

    _run_pty(args.continue_session)


if __name__ == "__main__":
    main()
