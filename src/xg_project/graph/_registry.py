"""The node registry: what nodes exist, and who may move between them.

Two ideas live here and nothing else does.

**A node declaration is data, not behaviour.** It says what a node is called, how
generic it is, its kind, and where it was defined. The behaviour hangs off
`handle` as a plain callable so a declaration stays comparable, printable, and
testable without running anything.

**Every node introduces state, and the state is keyed by node.** A handler is
handed the prompt, the state so far, and the drivers it needs, and returns the
value it adds under its own name. Nothing is appended to a shared context
window: a later node reads one earlier node's entry by that node's key, which is
what makes a workflow a sequence of explicit contributions rather than an
accumulating transcript.

**Kind is one of two things, and it says where the run goes next.** A decision
node has children and moves to one of them; a generative node is a leaf where a
model produces the result. Both introduce state; the kind only decides whether
there is a move afterwards.

**Movement is a policy over levels, not a property of the graph.** The graph is
bidirectional; what is asymmetric is who may drive a move. ``Actor.JEV`` may
only descend (strictly higher level); ``Actor.USER`` may move anywhere. That rule
is one predicate, held in one place, so there is exactly one definition of it.

**The graph is a graph, not a tree.** A node may be the child of two others, and
a node may reach one of its own ancestors. Both are things a user may configure,
so neither is treated as a defect: the drawing carries every edge, and "the path
back to the origin" is the shortest one rather than the only one. Anything that
walked parent by parent would need a cycle check before it could terminate, and
would then have to decide what a cycle means — a shortest path needs no answer,
because a node already reached is never expanded twice.

Levels are genericity. Level 0 is the origin, the most generic node; deeper
nodes are more specific. "Up" always means toward the origin, and "down" always
means away from it.

**Built-in nodes are namespaced.** :func:`xg` prefixes them with ``_XG_``, so a
user's node never collides with one xg ships, and the rendering makes the
boundary visible.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from graphtty import RenderOptions, render

if TYPE_CHECKING:  # annotations only, so the registry stays free of the drivers
    from xg_project.jev import Jev
    from xg_project.llm import Executor

HERE = "◀ you are here"
"""How the drawing marks the node the run is standing on.

A line inside the node's own box rather than a colour or a symbol beside it: the
drawing is a string, and a marker that still says what it meant after being
logged or pasted somewhere is worth more than one that does not.
"""

XG_PREFIX = "_XG_"
"""Every node the built-in graph ships is named with this prefix.

A node without it came from a user or an extension, which is how a reader tells
the two apart. The prefix is written once, here, so it cannot drift between the
default graph and anything that checks for it.
"""


def xg(name: str) -> str:
    """Name a built-in node. The prefix lives here and nowhere else."""
    return f"{XG_PREFIX}{name}"


@dataclass(frozen=True)
class NodeInput:
    """What a node is handed when a prompt arrives.

    A node reads the prompt, the state earlier nodes introduced, and the drivers
    it needs. ``state`` is the whole accumulated state keyed by node name, so a
    node reads exactly what a previous node wrote and nothing else.
    """

    node: str
    prompt: str
    state: Mapping[str, object]
    jev: Jev | None = None
    executor: Executor | None = None
    root: str | Path | None = None


# A node handler receives a `NodeInput` and returns the value it introduces into
# the state under its own name. It may be async, because a node is where a model
# is asked; the turn awaits it.
Handler = Callable[[NodeInput], "object | Awaitable[object]"]


class Actor(StrEnum):
    """Who is asking for a move.

    The distinction exists solely so the up-only rule has something to check.
    Anything xg drives on its own is ``JEV``; anything the user asked for at the
    prompt is ``USER``.
    """

    USER = "user"
    JEV = "jev"


class UnknownNode(KeyError):
    """A move or lookup named a node the registry does not have."""


class NodeKind(StrEnum):
    """What happens when a prompt arrives at a node.

    The distinction is about *where the run goes next*: a decision node has
    children and moves to one of them, a generative node is a leaf where a model
    produces the final state. Every node introduces state; the kind says what
    comes after it.
    """

    DECISION = "decision"
    """The run moves on from here, to a child Jev chooses (or the only child).
    The user may also traverse it by hand."""

    GENERATIVE = "generative"
    """A leaf: a model is driven here and produces the result, a response or a
    tool call for xg to make. There is nowhere below it."""


@dataclass(frozen=True)
class NodeDeclaration:
    """One node: its place in the graph and what it does.

    ``summary`` is a generic description of what the node *does*: the effect it
    has on a request, not advice to whichever node links to it. A node never
    tells its parent when to route to it — the router matches a request against
    the described effects — so the description is the whole contract, and it has
    to be specific about the node's own behaviour rather than about a workflow it
    happens to sit in.

    ``handle`` is what the node introduces when a prompt arrives: the value stored
    in the state under this node's name. ``None`` means the node is graph structure
    only. It may be async, because a node is where a model is asked; the turn
    awaits it.

    ``children`` names the nodes reachable downward from here. It is the
    successor list, so the graph the TUI draws, the path walk that decides "up",
    and the option set Jev chooses from are all derived from it rather than
    stored twice. Successors rather than edges: a node declares what it leads to,
    and every edge in the graph is one declaration's successor entry.
    """

    name: str
    level: int
    summary: str
    kind: NodeKind = NodeKind.DECISION
    children: tuple[str, ...] = ()
    handle: Handler | None = None
    origin: str = "builtin"
    proposes: bool = False
    """Whether the node's result is a proposal the user must accept or reject
    before anything happens. Only a generative node may set it, because only a
    generative node is a leaf a model proposes from."""

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError(f"{self.name!r}: level must be >= 0, got {self.level}")
        if not self.summary.strip():
            raise ValueError(f"{self.name!r}: summary must not be empty")
        if self.proposes and self.kind is not NodeKind.GENERATIVE:
            raise ValueError(
                f"{self.name!r}: only a generative node can propose; "
                f"{self.kind} has no executor"
            )


@dataclass
class Registry:
    """The set of known nodes, and the movement rules over them.

    Insertion order is never load-bearing. Everything that produces a list
    sorts, so the graph the TUI draws and the `Choice` criteria handed to Jev do
    not shift between runs.
    """

    _nodes: dict[str, NodeDeclaration] = field(default_factory=dict)

    def add(self, node: NodeDeclaration, *, override: bool = False) -> None:
        """Register a node, refusing to clobber a name unless told to.

        A silent override in a system built on determinism is the sort of thing
        that costs an afternoon later, so the default is to refuse and the
        caller has to say ``override=True`` out loud.
        """
        if node.name in self._nodes and not override:
            raise ValueError(
                f"node {node.name!r} is already defined by "
                f"{self._nodes[node.name].origin!r}"
            )
        self._nodes[node.name] = node

    def __contains__(self, name: object) -> bool:
        return name in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[NodeDeclaration]:
        return iter(self.sorted())

    def get(self, name: str) -> NodeDeclaration:
        """Look up a node, or raise :class:`UnknownNode`."""
        try:
            return self._nodes[name]
        except KeyError:
            raise UnknownNode(name) from None

    def sorted(self) -> list[NodeDeclaration]:
        """Every node, ordered by genericity then name."""
        return sorted(self._nodes.values(), key=lambda n: (n.level, n.name))

    @property
    def origin(self) -> NodeDeclaration:
        """The most generic node. Ties are an error, not a coin flip."""
        lowest = min(n.level for n in self._nodes.values())
        at_lowest = [n for n in self._nodes.values() if n.level == lowest]
        if len(at_lowest) != 1:
            names = ", ".join(sorted(n.name for n in at_lowest))
            raise ValueError(f"ambiguous origin at level {lowest}: {names}")
        return at_lowest[0]

    def criteria(self) -> dict[str, str]:
        """Name -> summary, shaped for a Jev `Choice`.

        This is the bridge between the graph and the driver: the same mapping
        that gets handed to Jev as the option set is the one the TUI draws, so
        the two cannot drift apart.
        """
        return {n.name: n.summary for n in self.sorted()}

    def options(self, frm: str) -> dict[str, str]:
        """The option set Jev chooses from when standing at ``frm``.

        Only the children of the current node, because Jev may only descend and
        a sibling is not a descent. Offering a node Jev cannot legally move to
        would produce a decision that the movement policy then refuses.

        Nodes Jev cannot reach are excluded rather than merely discouraged: the
        criteria text is the model's only input, so an option that is always
        wrong is a wasted option and a chance to get the answer wrong.
        """
        return {
            child: self._nodes[child].summary
            for child in self._nodes[frm].children
            if self.check_move(frm, child, Actor.JEV) is None
        }

    def edges(self) -> list[tuple[str, str]]:
        """Every edge as ``(parent, child)``, sorted so a drawing is stable.

        The declarations hold successors, so this is the same structure read as
        edges rather than a second thing to keep in step. A child that is not a
        node is a defect in a declaration, not a node that is missing, and is
        raised here rather than drawn as a box nobody declared.
        """
        out: list[tuple[str, str]] = []
        for node in self._nodes.values():
            for child in node.children:
                if child not in self._nodes:
                    raise UnknownNode(f"{node.name!r} lists unknown child {child!r}")
                out.append((node.name, child))
        return sorted(
            out,
            key=lambda edge: (
                self._nodes[edge[0]].level,
                edge[0],
                self._nodes[edge[1]].level,
                edge[1],
            ),
        )

    def parents(self) -> dict[str, tuple[str, ...]]:
        """Child name -> every node that leads to it, derived from ``children``.

        Built fresh rather than stored, so a declaration cannot contradict
        itself about what its neighbours are.

        A tuple rather than one name because the graph is not a tree: two nodes
        may lead to the same child, and a node that leads back to one of its own
        ancestors is a cycle. Both are configurations a user may write, so
        neither is collapsed into a single parent or refused.
        """
        out: dict[str, list[str]] = {}
        for parent, child in self.edges():
            out.setdefault(child, []).append(parent)
        return {child: tuple(parents) for child, parents in out.items()}

    def parent(self, name: str) -> str | None:
        """What ``/go ..`` resolves to: the node above ``name`` on the path here.

        With a single route back to the origin this is the only parent. With
        several, or with a cycle making some routes longer, it is the parent on
        the *shortest* path — the one :meth:`ancestors` returns — so "up" agrees
        with the trail the TUI draws instead of being a second opinion about it.

        ``None`` at the origin, and for a node no route reaches. It looks the
        node up first, so a typo raises :class:`UnknownNode` rather than reading
        as "already at the origin".
        """
        self.get(name)
        try:
            path = self.ancestors(name)
        except ValueError:
            return None
        return path[-2] if len(path) > 1 else None

    def ancestors(self, name: str) -> list[str]:
        """The shortest path from the origin down to ``name``, inclusive.

        Breadth-first from the origin. Two properties follow from that and both
        matter here: a graph offering two routes picks the shorter one
        deterministically rather than whichever the insertion order happened to
        visit first, and a cycle terminates instead of walking it forever. A
        node already reached is never expanded twice.

        A node no route reaches is a defect in a graph somebody added, and is
        raised rather than papered over: the trail would otherwise claim a path
        that does not exist.
        """
        self.get(name)
        origin = self.origin.name

        successors: dict[str, list[str]] = {node.name: [] for node in self._nodes.values()}
        for parent, child in self.edges():
            successors[parent].append(child)

        came_from: dict[str, str] = {origin: origin}
        queue: deque[str] = deque([origin])
        while queue:
            current = queue.popleft()
            if current == name:
                break
            for child in successors[current]:
                if child not in came_from:
                    came_from[child] = current
                    queue.append(child)

        if name not in came_from:
            raise ValueError(f"{name!r} is not reachable from the origin")

        path = [name]
        while path[-1] != origin:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def check_move(self, frm: str, to: str, actor: Actor) -> str | None:
        """Return ``None`` when the move is allowed, else why it was refused.

        The reason is a string rather than an exception because the TUI shows it
        to the user, and a refusal is an ordinary outcome there, not a fault.
        """
        if frm not in self._nodes:
            return f"unknown node {frm!r}"
        if to not in self._nodes:
            return f"unknown node {to!r}"
        if frm == to:
            return f"already at {to!r}"

        level_from = self._nodes[frm].level
        level_to = self._nodes[to].level
        if actor is Actor.JEV and level_to <= level_from:
            direction = "sideways" if level_to == level_from else "toward the origin"
            return f"only the user moves {direction}"
        return None

    def render_graph(self, current: str, *, max_width: int | None = None) -> str:
        """The whole graph drawn as a graph, with ``current`` marked.

        Every edge is handed to the layout engine, because a traversal is a
        statement about trees: walking ``children`` from the origin would draw a
        node with two parents under only one of them, and would have to decide
        what to do about a cycle before it could finish at all. The registry's
        edges are what the drawing is made of, so what is shown cannot disagree
        with what the movement rules allow.

        ``max_width`` bounds the drawing horizontally. The engine re-renders with
        shortened text to fit, rather than truncating the right-hand edge, which
        would take the boxes there with it.
        """
        self.get(current)
        origin = self.origin.name
        drawing = {
            "nodes": [self._drawn_node(node, current=current, origin=origin) for node in self.sorted()],
            "edges": [
                {"source": parent, "target": child} for parent, child in self.edges()
            ],
        }
        return render(drawing, RenderOptions(max_width=max_width))

    def _drawn_node(
        self, node: NodeDeclaration, *, current: str, origin: str
    ) -> dict[str, str]:
        """One node in the shape the layout engine takes.

        The type label is the node's kind, except at the origin, which gets a
        label of its own: "where a run starts" is the one thing about the origin
        worth reading that its kind does not say.
        """
        drawn = {
            "id": node.name,
            "name": node.name,
            "type": "origin" if node.name == origin else node.kind.value,
        }
        if node.name == current:
            drawn["description"] = HERE
        return drawn
