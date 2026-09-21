"""The built-in graph: the nodes xg ships with.

Only two exist so far, and they are one of each kind.

``none``
    The origin, and a decision node. The most generic place a session can be.
    The user states a goal here; the node contributes that goal to the context
    window, and Jev is asked which child should handle it. A prompt arriving
    here is what Jev chooses from.

``command``
    A generative node. An LLM is driven here and produces the result. It
    contributes nothing to the context window: the prompt goes straight in and
    no files are read. That is its defining property, and it follows from the
    kind rather than being configured beside it.

Both summaries are written as the text Jev chooses between. That is not
overstatement — ``summary`` is the only thing the model is shown about a node,
so a summary that describes a node without saying when to pick it is a routing
bug waiting to happen.
"""

from __future__ import annotations

from xg_project.graph._registry import NodeDeclaration, NodeKind, Registry

ORIGIN = "none"
COMMAND = "command"


def _record_goal(prompt: str) -> str:
    """Take the user's goal at the origin and add it to the context window.

    Stating the goal is all that happens here. Which node handles it is the
    decision Jev is asked for immediately afterwards, so this returns the
    contribution rather than a result.
    """
    return f"the user wants: {prompt}"


def _run_command(prompt: str) -> str:
    """Handle a prompt at ``command``, which needs no context window.

    The prompt is returned unchanged. Nothing executes it: the executor is the
    language model, the last step of a run, and it is not wired yet. This
    function is the seam it will plug into.
    """
    return prompt


def default_graph() -> Registry:
    """Build the graph xg always has, before discovery adds to it."""
    registry = Registry()
    registry.add(
        NodeDeclaration(
            name=ORIGIN,
            level=0,
            kind=NodeKind.DECISION,
            summary=(
                "The origin: the most generic node, where the user states the "
                "goal. Nothing is read and nothing runs here. Every session "
                "starts here and every move back up ends here."
            ),
            children=(COMMAND,),
            handle=_record_goal,
        )
    )
    registry.add(
        NodeDeclaration(
            name=COMMAND,
            level=1,
            kind=NodeKind.GENERATIVE,
            summary=(
                "Run a self-contained instruction and answer it, with no "
                "repository context and no files read. Pick this when the "
                "request can be satisfied from the prompt alone."
            ),
            children=(),
            handle=_run_command,
        )
    )
    return registry

