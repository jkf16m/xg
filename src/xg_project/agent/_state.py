"""State for the xg agent graph."""

from typing import TypedDict

from xg_project.jev import Classification
from xg_project.research import Research


class AgentState(TypedDict, total=False):
    """State threaded through the graph.

    ``prompt`` is the only input. ``classification`` is written by the classify
    node. ``research`` and ``relevant_files`` are written by the local_research
    node. ``route`` is written by whichever node the run ended in, so a caller
    can see where it went.
    """

    prompt: str
    classification: Classification
    research: Research
    relevant_files: list[str]
    route: str
