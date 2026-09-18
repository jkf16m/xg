"""Command-line entry point for xg.

The agent is being rebuilt on LangGraph with TypeSafe Jev for decisions. The
first stage is wired up: ``xg`` takes a prompt and returns the request
classification. Omit the prompt to classify prompts from stdin in a loop.

    xg "add retry to the http client"
    xg
"""


def main() -> int:
    """Run the xg entry point. Currently the first filter."""
    from xg_project.jev.__main__ import main as first_filter

    return first_filter()


if __name__ == "__main__":
    raise SystemExit(main())
