"""The session: where the user is, and what the nodes have introduced so far.

The session owns two pieces of durable state. ``position`` is a node name — where
the run is. ``state`` is a mapping keyed by node name, holding the value each
node introduced when it ran. That is deliberate: a run needs no executor to be
resumable because everything it learned is a plain value under a node key.

The state is keyed by node rather than accumulated in a list, and that is the
point of the model. A later node reads one earlier node's entry by that node's
key (SORT reads FILTER's files, EDIT reads SORT's ranking), so the shape a node
depends on is written down and checked rather than inferred from ordering.

Two things are not stored. ``trail`` is the path from the origin down to where
the run is, derived from the graph rather than remembered from the moves, so it
shrinks when the run goes back up instead of growing forever. ``context`` is
derived from the state for the one place prose is wanted — the text Jev is shown
when it routes — so it cannot drift from the state it summarizes.

**The state is the path, not a history.** A node's entry is dropped as soon as
the run moves somewhere that leaves it behind, so the state holds exactly what
the nodes on the current path contributed and nothing else. This is what makes
walking back up safe. A deeper node's state was derived from a shallower one's,
so going up to re-run the shallower node has to invalidate the derivation: after
moving from SORT back to FILTER, SORT's ranking of the files that were just
discarded must not be left where EDIT can find it. The same applies sideways —
a node the run stepped away from is a node it is no longer standing on.

Moves return small result records instead of raising. Being refused a move is a
normal thing for a user to do — ``/go ..`` from the origin, or typing a node name
that does not exist — and the TUI shows the refusal as a sentence rather than a
traceback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from xg_project.graph import Actor, Registry


@dataclass(frozen=True)
class Move:
    """The outcome of an attempt to change position."""

    frm: str
    to: str
    ok: bool
    reason: str | None = None


def brief(value: object) -> str:
    """A short, one-line reading of one node's state, for routing text.

    Deliberately conservative: it recognizes the few shapes the built-in graph
    produces and falls back to ``str`` for anything a user defined, so a new node
    is still legible to Jev without teaching this function about it.
    """
    if isinstance(value, str):
        return value
    preview = getattr(value, "preview", None)
    if callable(preview):
        # A gate or a proposal knows how to render itself; asking it keeps this
        # function from having to know every node's value type.
        return str(preview())
    if isinstance(value, Mapping):
        goal = value.get("goal")
        if isinstance(goal, str):
            return goal
        files = value.get("files")
        if isinstance(files, Mapping):
            return f"{len(files)} files"
        problem = value.get("problem")
        if problem:
            return f"problem: {problem}"
    return str(value)


@dataclass
class Session:
    """One conversation: a position, and the state the nodes on the way here left.

    There is no trail field: the trail is the path to ``position``, and the state
    is pruned to the same path on every move. Both are therefore consequences of
    the position rather than a second thing that could disagree with it.
    """

    graph: Registry
    position: str = ""
    state: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.position:
            self.position = self.graph.origin.name

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
        self._keep_path()
        return moved

    def _path(self) -> tuple[str, ...] | None:
        """The path from the origin down to the position, or ``None`` if there is none.

        A node with no route back to the origin is a defect in a graph somebody
        added, not a user error, and it is not worth raising in the middle of a
        move that was otherwise allowed. ``None`` says "cannot tell", and each
        caller does the safe thing with that rather than the two of them crashing.
        """
        try:
            return tuple(self.graph.ancestors(self.position))
        except ValueError:
            return None

    def _keep_path(self) -> None:
        """Drop the state of every node that is not on the path to the position."""
        path = self._path()
        if path is None:
            # An unreadable path means we cannot tell what is still current, and
            # clearing the wrong half is worse than leaving both halves standing.
            return
        on_path = set(path)
        for name in [name for name in self.state if name not in on_path]:
            del self.state[name]

    @property
    def trail(self) -> tuple[str, ...]:
        """The path from the origin down to here.

        A path, not a history: going back up makes it shorter, and descending and
        returning any number of times leaves it the length it is. It is derived
        from the graph, so it also names the nodes the run passed through without
        landing on — which is exactly the set of nodes that may have state.
        """
        path = self._path()
        return path if path is not None else (self.position,)

    def record(self, node: str, produced: object) -> None:
        """Store what a node introduced under that node's own key.

        One entry per node, replaced on re-entry: running a node again produces
        the state for where the run is now, not a second version to compare
        against.
        """
        self.state[node] = produced

    @property
    def goal(self) -> str | None:
        """The origin's reading of the request, once the origin has run."""
        value = self.state.get(self.graph.origin.name)
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) and isinstance(value.get("goal"), str):
            return str(value["goal"])
        return None

    @property
    def context(self) -> list[str]:
        """The state as labelled lines, for the one consumer that wants prose.

        Jev routes from this. Deriving it means the state remains the single
        source: there is no parallel list that a node could forget to update.
        """
        return [f"{name}: {brief(value)}" for name, value in self.state.items()]
