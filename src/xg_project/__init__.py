"""Command-line entry point for xg.

The agent is being rebuilt on LangGraph with TypeSafe Jev for decisions.
``xg`` runs the agent graph: the Jev first filter classifies the prompt, then
the router sends it to a downstream node.

    xg "add retry to the http client"
    xg
"""


def main() -> int:
    """Run the xg agent graph."""
    from xg_project.agent.__main__ import main as agent_main

    return agent_main()


if __name__ == "__main__":
    raise SystemExit(main())
