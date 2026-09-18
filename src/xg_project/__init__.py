"""Command-line entry point for xg.

Rebuild status
--------------
The argparse surface, ``--info`` reporting, and PTY launcher were removed on
the ``refactor/jev-langgraph`` branch. ``main`` stays as the ``xg`` console
script entry point and is re-implemented with the LangGraph agent.
"""


def main() -> None:
    """Run the xg CLI. Re-implement on LangGraph."""
    raise NotImplementedError("xg_project.main: rebuild pending")


if __name__ == "__main__":
    main()
