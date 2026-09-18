"""State for the xg agent graph."""

from typing import TypedDict

from xg_project.jev import Classification
from xg_project.research import Research


class AgentState(TypedDict, total=False):
    """State threaded through the graph.

    ``prompt`` is the only input. ``classification`` is written by the classify
    node. ``research``, ``relevant_files``, and ``file_contents`` are written by
    the local_research node, which copies the selected files' text into
    ``file_contents`` so the downstream generative step has the content, not
    just the paths. ``route`` is written by whichever node the run ended in.
    """

    prompt: str
    classification: Classification
    research: Research
    relevant_files: list[str]
    file_contents: dict[str, str]
    route: str
