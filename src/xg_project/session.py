"""The session: where the user is, what they have asked for, and where they have been.

The session owns exactly one piece of durable state — ``position``, a node name.
Everything else it holds is derived from the graph or is a record of what
happened. That is deliberate: it is the whole reason a run needs no executor to
be resumable.

Two things accumulate as a run proceeds, and they are different in kind.
``trail`` is where the user has *been*: a path, kept for display. ``context`` is
what has been *learned*: the entries that decision nodes contributed, in the
order they were contributed, which is what a later node is handed. A trail entry
is a node name; a context entry is a sentence.

Moves and prompts return small result records instead of raising. Being refused
a move is a normal thing for a user to do — ``:none`` from ``none``, or typing a
node name that does not exist — and the TUI shows the refusal as a sentence
rather than a traceback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xg_project.graph import Actor, NodeKind, Registry


@dataclass(frozen=True)
class Move:
    """The outcome of an attempt to change position."""

    frm: str
    to: str
    ok: bool
    reason: str | None = None


@dataclass(frozen=True)
class Run:
    """The outcome of submitting a prompt at whichever node is current.

    ``result`` means different things by kind, and that is the point of keeping
    ``kind`` on the record: at a generative node it is the answer, and at a
    decision node it is what the node added to the context window instead. A
    caller that logs both the same way would be reporting a routing note as an
    answer.
    """

    node: str
    prompt: str
    result: str
    kind: NodeKind = NodeKind.GENERATIVE
    is_goal: bool = False


@dataclass
class Session:
    """One conversation: a position, a trail, a context window, and the goal."""

    graph: Registry
    position: str = ""
    trail: tuple[str, ...] = field(default_factory=tuple)
    goal: str | None = None
    context: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.position:
            self.position = self.graph.origin.name
        if not self.trail:
            # The trail starts where the session did. A path that lists only the
            # moves made reads as if the session began at the first move.
            self.trail = (self.position,)

    @property
    def node(self):
        """The declaration for the current position."""
        return self.graph.get(self.position)

    def move(self, to: str, actor: Actor = Actor.USER) -> Move:
        """Attempt to move to ``to``, applying the registry's movement policy.

        The policy lives in the registry, not here, so that a move Jev requests
        and a move the user types go through the identical check.
        """
        to = to.strip()
        if not to:
            return Move(frm=self.position, to=to, ok=False, reason="no node named")

        refusal = self.graph.check_move(self.position, to, actor)
        if refusal is not None:
            return Move(frm=self.position, to=to, ok=False, reason=refusal)

        moved = Move(frm=self.position, to=to, ok=True)
        self.position = to
        self.trail = (*self.trail, to)
        return moved

    def submit(self, prompt: str) -> Run:
        """Run ``prompt`` at the current node and return what it produced.

        The node's kind decides what "produced" means. A prompt arriving at the
        origin is a *goal*: it is kept so a later move can act on it, and it is
        added to the context window because the origin is a decision node. At a
        generative node the handler's return value is the result, and nothing is
        added to the context window.
        """
        node = self.node
        is_goal = node.name == self.graph.origin.name
        if is_goal:
            self.goal = prompt

        if node.handle is None:
            result = "this node has nothing to do yet"
        else:
            result = node.handle(prompt)

        if node.builds_context:
            # The contribution is what later nodes are handed, so it is appended
            # here rather than at the call site: every route into a decision
            # node has to contribute, and a caller that forgot would silently
            # produce a context window with a hole in it.
            self.context.append(result)

        return Run(
            node=node.name,
            prompt=prompt,
            result=result,
            kind=node.kind,
            is_goal=is_goal,
        )
