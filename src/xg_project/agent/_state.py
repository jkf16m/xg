"""State for the xg agent graph."""

from typing import TypedDict

from xg_project.jev import Classification


class AgentState(TypedDict, total=False):
    """State threaded through the graph.

    ``prompt`` is the only input. ``classification`` is written by the classify
    node. ``route`` is written by whichever terminal node the router picked, so
    a caller can see where the run went.
    """

    prompt: str
    classification: Classification
    route: str
