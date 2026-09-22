"""One turn: a prompt arrives, and the node it lands on introduces its state.

The turn is where the session (pure, bookkeeping) meets the drivers
(asynchronous, networked). Keeping it out of both means the app is only
responsible for drawing, and this is testable by handing it a Jev and an
executor that return fixed answers.

What a turn does is the same at every node: run the node's handler, record what
it introduced under that node's key, and then move on if there is somewhere to
go. The kind of the node decides how the move is chosen:

- **one child** — the move is the edge, taken without asking anyone.
- **several children** — Jev is asked which child, from the state as it stands.
- **no children** — the run stops here. A *proposing* leaf hands back a
  :class:`Gate`: the executor's action is a proposal, and the turn ends awaiting
  the user rather than acting.

A routing that fails is not an error. It leaves the session exactly where it was,
with the node's state already recorded, and the caller reports why.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

from xg_project.graph import Actor, NodeInput
from xg_project.jev import Jev, Routing
from xg_project.llm import Executor, Proposal
from xg_project.session import Move, Session

PENDING = "pending"
ACCEPTED = "accepted"
REJECTED = "rejected"


@dataclass
class Gate:
    """A proposal awaiting the user's decision, and what came of it.

    The gate is the session state a proposing node introduces, so the pending
    action, its acceptance or rejection, and the outcome all live in one place
    the state panel can render.
    """

    proposal: Proposal
    status: str = PENDING
    outcome: object | None = None

    @property
    def settled(self) -> bool:
        """Whether the user has decided."""
        return self.status is not PENDING

    def preview(self) -> str:
        """A one-line rendering of the gate, for the state panel."""
        if self.status == ACCEPTED:
            return f"accepted: {self.proposal.preview()} — {_outcome_text(self.outcome)}"
        if self.status == REJECTED:
            return f"rejected: {self.proposal.preview()}"
        return f"pending: {self.proposal.preview()}"

    def detail(self) -> str:
        """What the proposal would change, in full, for the state panel.

        The pending preview is the point of it: a patch is what makes an edit
        decidable, and it belongs in the state because that is the pane built to
        be scrolled when it is longer than the screen.
        """
        return self.proposal.detail()


def _outcome_text(outcome: object) -> str:
    """A sentence for an accepted proposal's outcome, if it can produce one."""
    if outcome is None:
        return "done"
    explain = getattr(outcome, "explain", None)
    return str(explain()) if callable(explain) else str(outcome)


@dataclass(frozen=True)
class Turn:
    """Everything that happened between a prompt and the next prompt."""

    node: str
    """The node the prompt landed on."""

    prompt: str
    """What the user submitted, unchanged."""

    produced: object
    """What the node introduced into the state."""

    routing: Routing | None = None
    """Jev's answer, when the node had several children. ``None`` when the move
    was a plain edge, or when there was nowhere to go."""

    moved: Move | None = None
    """The move that was made, or ``None`` at a leaf or after a failed routing."""

    gate: Gate | None = None
    """The proposal awaiting the user, when the node proposes one."""

    @property
    def descended(self) -> bool:
        """Whether the session is now at a node the graph chose."""
        return self.moved is not None and self.moved.ok

    @property
    def awaiting(self) -> bool:
        """Whether the turn ended on a proposal the user has to accept or reject."""
        return self.gate is not None

    @property
    def proposal(self) -> Proposal | None:
        """The proposal, when there is one."""
        return self.gate.proposal if self.gate is not None else None


async def take_turn(
    session: Session,
    prompt: str,
    *,
    jev: Jev | None = None,
    executor: Executor | None = None,
    root: str | Path | None = None,
) -> Turn:
    """Run ``prompt`` at the current node, record its state, and follow the edge.

    ``jev`` routes when a node has several children; ``executor`` proposes at a
    leaf; ``root`` is the tree a filter node reads. Each is only consulted by the
    nodes that need it, and a missing one becomes a problem in the state rather
    than an exception.
    """
    node = session.node
    produced = await _run(node, session, prompt, jev=jev, executor=executor, root=root)

    if node.proposes and isinstance(produced, Proposal):
        gate = Gate(proposal=produced)
        session.record(node.name, gate)
        return Turn(node=node.name, prompt=prompt, produced=produced, gate=gate)

    session.record(node.name, produced)
    return await _advance(session, node.name, prompt, produced, jev=jev)


async def _run(
    node,
    session: Session,
    prompt: str,
    *,
    jev: Jev | None,
    executor: Executor | None,
    root: str | Path | None,
) -> object:
    """Run one node's handler and return the state it introduced."""
    if node.handle is None:
        return {"problem": f"{node.name} has nothing to do yet"}

    inp = NodeInput(
        node=node.name,
        prompt=prompt,
        state=session.state,
        jev=jev,
        executor=executor,
        root=root,
    )
    produced = node.handle(inp)
    if inspect.isawaitable(produced):
        produced = await produced
    return produced


async def _advance(
    session: Session, node: str, prompt: str, produced: object, *, jev: Jev | None
) -> Turn:
    """Choose the next node and move, or stop when there is no next node."""
    options = session.graph.options(node)

    if not options:
        return Turn(node=node, prompt=prompt, produced=produced)

    if len(options) == 1:
        # One child is the edge itself; there is nothing to decide, so nothing is
        # asked and no round trip is spent proving it.
        target = next(iter(options))
        return Turn(
            node=node, prompt=prompt, produced=produced, moved=session.move(target, Actor.JEV)
        )

    if jev is None:
        return Turn(
            node=node,
            prompt=prompt,
            produced=produced,
            routing=Routing(problem="no Jev is attached, so nothing can be routed"),
        )

    routing = await jev.decide(
        current=node,
        context=session.context,
        prompt=prompt,
        options=options,
    )
    if not routing.ok:
        return Turn(node=node, prompt=prompt, produced=produced, routing=routing)

    assert routing.node is not None  # guaranteed by Routing.ok
    return Turn(
        node=node,
        prompt=prompt,
        produced=produced,
        routing=routing,
        moved=session.move(routing.node, Actor.JEV),
    )
