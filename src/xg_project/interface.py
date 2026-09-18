"""xg harness interface — REPL entry point.

Rebuild status
--------------
The streaming markdown renderer, key handling, and tool-approval flow were
removed on the ``refactor/jev-langgraph`` branch. ``main`` is the PTY child
entry point and is re-implemented once the LangGraph agent exists.
"""


def main() -> None:
    """Run the interactive harness. Re-implement on LangGraph."""
    raise NotImplementedError("interface.main: rebuild pending")


if __name__ == "__main__":
    main()
