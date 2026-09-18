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
    just the paths.

    The write path adds ``proposed_path`` (the model's raw reply), ``target_path``
    (the same reply once validated as a safe relative path), ``context_path``
    (the temporary window), and ``written_path`` / ``written_bytes``.

    ``decision`` is written by whichever node asked the user: ``continue`` to
    move on, or ``reprompt`` to re-classify the prompt the user supplied.
    ``reason`` explains why a run was rerouted, and ``route`` records where the
    run ended.

    ``__interrupt__`` is added by LangGraph rather than by a node, when a run
    stops at the confirm or reroute node. Read it with
    ``xg_project.agent.pending``.
    """

    prompt: str
    classification: Classification
    research: Research
    relevant_files: list[str]
    file_contents: dict[str, str]
    decision: str
    reason: str
    route: str
    proposed_path: str
    target_path: str
    context_path: str
    context_files: int
    context_bytes: int
    written_path: str
    written_bytes: int
