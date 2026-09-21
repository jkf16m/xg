"""One turn: a prompt arrives, and either Jev routes it or a node answers it.

The turn is where the session (pure, synchronous, testable) meets the driver
(asynchronous, networked). Keeping it out of both means the app is only
responsible for drawing, and this is testable by handing it a Jev that returns a
fixed answer.

The shape of a turn depends entirely on the kind of node the prompt landed on:

- **generative** — the node produces the result. The turn ends there. This is
  the end of a run, and the point of the whole design: an LLM is used once, at
  the leaf, on a context window that was assembled deterministically.
- **decision** — the node contributes to the context window, and Jev is asked
  which child is next. The turn ends with a move, not a result.

A routing that fails is not an error. It leaves the session exactly where it was,
with the context contribution already made, and the caller reports why.
"""

from __future__ import annotations

from dataclasses import dataclass

from xg_project.graph import Actor, NodeKind
from xg_project.jev import Jev, Routing
from xg_project.session import Move, Run, Session


@dataclass(frozen=True)
class Turn:
    """Everything that happened between a prompt and the next prompt."""

    run: Run
    """What the node the prompt landed on produced."""

    routing: Routing | None = None
    """Jev's answer, when the node was a decision node. ``None`` at a generative
    node, where no routing question was asked."""

    moved: Move | None = None
    """The move Jev's answer produced, when it produced one. A routing that failed
    leaves this ``None`` and the session untouched."""

    @property
    def descended(self) -> bool:
        """Whether the session is now at a node Jev chose."""
        return self.moved is not None and self.moved.ok


async def take_turn(session: Session, jev: Jev, prompt: str) -> Turn:
    """Submit ``prompt`` at the current node and follow through on its kind."""
    run = session.submit(prompt)

    if run.kind is NodeKind.GENERATIVE:
        return Turn(run=run)

    # The contribution is already in the context window, so Jev is shown the
    # window as it now stands. It contains the goal and every note along the way,
    # which is the whole of what Jev decides from.
    routing = await jev.decide(
        current=run.node,
        context=session.context,
        prompt=prompt,
        options=session.graph.options(run.node),
    )

    if not routing.ok:
        return Turn(run=run, routing=routing)

    assert routing.node is not None  # guaranteed by Routing.ok
    return Turn(run=run, routing=routing, moved=session.move(routing.node, Actor.JEV))
