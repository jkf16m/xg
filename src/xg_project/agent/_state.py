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
    just the paths. ``decision`` is written by the confirm node: ``continue``
    to move on, or ``reprompt`` to re-classify the prompt the human supplied.
    ``route`` is written by whichever node the run ended in.

    ``__interrupt__`` is added by LangGraph rather than by a node, when a run
    stops at the confirm node. Read it with ``xg_project.agent.pending``.
    """

    prompt: str
    classification: Classification
    research: Research
    relevant_files: list[str]
    file_contents: dict[str, str]
    decision: str
    route: str
