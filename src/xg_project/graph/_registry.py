"""The node registry: what nodes exist, and who may move between them.

Two ideas live here and nothing else does.

**A node declaration is data, not behaviour.** It says what a node is called, how
generic it is, its kind, and where it was defined. The behaviour hangs off
`handle` as a plain callable so a declaration stays comparable, printable, and
testable without running anything.

**Kind is one of two things, and it decides everything else.** A decision node is
where Jev reads the request and picks the next node, and its contribution is an
addition to the context window later nodes are handed. A generative node is where
an LLM is driven and produces the result itself, so it contributes no context and
receives the prompt directly. `builds_context` is a property of the kind rather
than a field beside it, so a node cannot claim to be one kind and behave like the
other.

The one exception is the origin, which is a decision node whose contribution is
the *goal* rather than a piece of context: it is where the user says what they
want before any node has been chosen.

**Movement is a policy over levels, not a property of the graph.** The graph is
bidirectional; what is asymmetric is who may drive a move. ``Actor.JEV`` may
only descend (strictly higher level); ``Actor.USER`` may move anywhere. That rule
is one predicate, held in one place, so there is exactly one definition of it.

Levels are genericity. Level 0 is the origin, the most generic node; deeper
nodes are more specific. "Up" always means toward the origin, and "down" always
means away from it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum

# A node handler receives the submitted prompt and returns the text to show.
# `command` ignores the context window entirely; later nodes will be handed one.
Handler = Callable[[str], str]


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

    The distinction is about *who produces the next thing*: Jev picks a
    destination at a decision node, an LLM produces an answer at a generative
    one. It is the only thing that varies between nodes; the rest of a node's
    behaviour follows from it.
    """

    DECISION = "decision"
    """Jev routes here. The node contributes to the context window built for the
    nodes below it, and the user may also traverse it by hand."""

    GENERATIVE = "generative"
    """An LLM is driven here and produces the result: a response, or a tool call
    for xg to make."""


@dataclass(frozen=True)
class NodeDeclaration:
    """One node: its place in the graph and what it does.

    ``summary`` is not documentation. It is the *only* text Jev sees when
    choosing between nodes, so it has to say what the node is for in terms a
    request can be matched against. A vague summary is a routing bug.

    ``handle`` is what the node contributes when a prompt arrives. At a decision
    node that is an addition to the context window; at a generative node it is
    the result. ``None`` means the node is graph structure only.

    ``children`` names the nodes reachable downward from here. It is the
    successor list, so the tree the TUI draws, the ancestor walk that decides
    "up", and the option set Jev chooses from are all derived from it rather
    than stored twice.
    """

    name: str
    level: int
    summary: str
    kind: NodeKind = NodeKind.DECISION
    children: tuple[str, ...] = ()
    handle: Handler | None = None
    origin: str = "builtin"

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError(f"{self.name!r}: level must be >= 0, got {self.level}")
        if not self.summary.strip():
            raise ValueError(f"{self.name!r}: summary must not be empty")

    @property
    def builds_context(self) -> bool:
        """Whether nodes below this one are handed what this node contributed."""
        return self.kind is NodeKind.DECISION


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

    def parents(self) -> dict[str, str]:
        """Child name -> parent name, derived from ``children``.

        Built fresh rather than stored, so a declaration cannot contradict
        itself about what its neighbours are.
        """
        out: dict[str, str] = {}
        for node in self._nodes.values():
            for child in node.children:
                if child not in self._nodes:
                    raise UnknownNode(f"{node.name!r} lists unknown child {child!r}")
                out[child] = node.name
        return out

    def ancestors(self, name: str) -> list[str]:
        """The path from the origin down to ``name``, inclusive.

        Walks parents until the origin. A node with no parent that is not the
        origin is unreachable, which is a graph defect rather than a user error.
        """
        self.get(name)
        parents = self.parents()
        path = [name]
        while (parent := parents.get(path[-1])) is not None:
            if parent in path:
                raise ValueError(f"cycle in graph above {name!r}")
            path.append(parent)
        path.reverse()
        if path[0] != self.origin.name:
            raise ValueError(f"{name!r} is not reachable from the origin")
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

    def render_tree(self, current: str) -> str:
        """A Rich-markup drawing of the whole graph with ``current`` marked.

        Rendered from the origin by depth-first walk over ``children``, so what
        is shown is the graph's own structure and not a second description of it.
        """
        marked: set[str] = set()
        lines: list[str] = []

        def walk(name: str, prefix: str, is_last: bool, root: bool) -> None:
            node = self._nodes[name]
            if node.name in marked:
                lines.append(f"{prefix}[dim]{node.name} (already shown)[/dim]")
                return
            marked.add(node.name)

            here = node.name == current
            connector = "" if root else ("└── " if is_last else "├── ")
            marker = "[b cyan]◆[/b cyan]" if here else "[dim]○[/dim]"
            label = f"[b]{node.name}[/b]" if here else node.name
            kind = "decision" if node.kind is NodeKind.DECISION else "generative"
            note = f"  [dim]{kind}"
            if root:
                note += " · origin"
            if here:
                note += "[/dim]  [cyan]← you are here[/cyan]"
            elif not node.children:
                note += " · leaf[/dim]"
            else:
                note += "[/dim]"
            lines.append(f"{prefix}{connector}{marker} {label}{note}")

            child_prefix = prefix + ("" if root else ("    " if is_last else "│   "))
            for index, child in enumerate(node.children):
                walk(child, child_prefix, index == len(node.children) - 1, False)

        walk(self.origin.name, "", True, True)
        return "\n".join(lines)
