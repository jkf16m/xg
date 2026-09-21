"""A decision tree in LangGraph, and how LangGraph works.

This module is a worked example, separate from the registry graph in
``_registry.py``. It exists to show the LangGraph primitives in one small,
readable place, and to make one design point about xg legible: **the graph is
assembled first and compiled once, at the end.**

LangGraph in five ideas
-----------------------

**State is a schema, not an object you mutate.** ``TreeState`` is a
``TypedDict``. Each node is a function that receives the current state and
returns *only the keys it changed*; LangGraph merges that partial dict into the
state and passes the result to the next node.

**A reducer says how a key merges.** ``trail`` is declared
``Annotated[list[str], operator.add]``, so a node returning ``{"trail": ["x"]}``
appends to the trail rather than replacing it. Without the reducer, the same
return value would overwrite. This is the seam where accumulation is configured.

**Nodes are named functions; edges connect names.** ``add_edge("a", "b")`` is an
unconditional hop. ``add_conditional_edges(source, decide, routes)`` calls
``decide(state)`` and follows the edge registered for the returned key. That
conditional edge is the branch of a decision tree.

**START and END are sentinels, not nodes.** ``START`` is where a run enters;
``END`` is where it stops. A run ends when it reaches ``END`` or when no node
has an outgoing edge.

**compile() is a build step.** Nothing can be invoked until ``StateGraph`` is
compiled, and compiling validates that every edge names a real node.

Why compile is deferred
-----------------------

Extensions add nodes and branches. An extension may want to attach a branch to a
node the default tree created, so the structure has to stay open while
extensions register. ``TreeBuilder`` therefore accumulates *declarations* — node
functions, edges, routers — and touches nothing in LangGraph until ``compile()``.

``compile()`` refuses while the tree is unsealed. Registration is open by
default; ``seal()`` says "no more extensions", and it is the single point where
discovery will finish. This is what makes the graph extensible: adding a
workflow adds a declaration, and every branch that already exists can be
extended by name right up until the seal.

The naming rule: every node the default tree ships is prefixed ``xg_``, so a
node without that prefix is known to have come from an extension. The prefix is
written once, in ``_default_name``.
"""

from __future__ import annotations

import argparse
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

# Every node the default tree ships carries this prefix. A node without it came
# from an extension, which is how a reader tells the two apart in the trail.
DEFAULT_PREFIX = "xg"


def _default_name(core: str) -> str:
    """Name a default node. The prefix lives here and nowhere else."""
    return f"{DEFAULT_PREFIX}_{core}"


class GraphNotSealed(RuntimeError):
    """``compile()`` was called while extensions could still arrive.

    Raised instead of compiling a graph that would silently omit whatever was
    registered a moment later. Call ``seal()`` once discovery is done.
    """


class UnknownTarget(ValueError):
    """A declaration points at a node that does not exist."""


class TreeState(TypedDict, total=False):
    """The single value threaded through the whole run.

    ``total=False`` because each node supplies only what it changed. The
    reducer on ``trail`` makes every node's return an append, so the trail is
    the ordered record of where the run has been and is the cheapest way to
    watch routing happen.
    """

    prompt: str
    """What the user asked, unchanged, from entry to exit."""

    goal: str
    """The origin's reading of the prompt: what the run is trying to do."""

    decision: str
    """The key the last decision node returned, for the trail to explain."""

    trail: Annotated[list[str], operator.add]
    """Node names visited, in order. Appended by every node, merged by reducer."""

    answer: str
    """The leaf's contribution. In the real system this is where an LLM runs."""


# A node receives the whole state and returns the keys it changed.
NodeFn = Callable[[TreeState], dict[str, object]]
# A router receives the whole state and returns the key of one registered route.
DecideFn = Callable[[TreeState], str]


class Extension(Protocol):
    """Anything that adds to the tree before it is sealed.

    An extension is a Python object, because Python is xg's extension
    mechanism rather than a plugin DSL. ``register`` is handed the open builder
    and may add nodes, edges, and routes; it may also read what is already
    there, which is why registration has to happen before ``compile()``.
    """

    name: str

    def register(self, tree: TreeBuilder) -> None: ...


@dataclass(frozen=True)
class Node:
    """One declared node: the function to run and where the declaration came from."""

    name: str
    fn: NodeFn
    summary: str
    origin: str = "default"


@dataclass(frozen=True)
class Router:
    """A declared branch: how the next node is chosen at one source.

    ``routes`` maps a key the decide function may return to the node that key
    means. Keep the key space separate from node names so one decision function
    can serve several targets and an extension can add a target without
    touching the existing ones.
    """

    source: str
    decide: DecideFn
    routes: dict[str, str] = field(default_factory=dict)


class TreeBuilder:
    """Declarations for one tree, open to extensions until ``seal()``.

    Nothing here talks to LangGraph. ``compile()`` is the only method that does,
    and it runs once, after the tree is closed. That split is the whole point:
    there is no compiled structure to invalidate when an extension registers.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[tuple[str, str]] = []
        self._routers: dict[str, Router] = {}
        self._entry: str | None = None
        self._sealed = False

    # -- registration (open until seal) -----------------------------------

    def add_node(
        self,
        name: str,
        fn: NodeFn,
        *,
        summary: str,
        origin: str = "default",
    ) -> None:
        """Declare a node. Names are unique; a clash is a defect, not a merge."""
        if self._sealed:
            raise GraphNotSealed(f"cannot add node {name!r}: the tree is sealed")
        if name in self._nodes:
            raise ValueError(f"node {name!r} is already declared")
        self._nodes[name] = Node(name=name, fn=fn, summary=summary, origin=origin)

    def set_entry(self, name: str) -> None:
        """Say which node a run starts at. Called once, by the default tree."""
        if self._sealed:
            raise GraphNotSealed("cannot set the entry: the tree is sealed")
        if self._entry is not None and self._entry != name:
            raise ValueError(f"entry is already {self._entry!r}")
        self._entry = name

    def add_edge(self, frm: str, to: str) -> None:
        """Declare an unconditional hop. ``START`` and ``END`` are allowed."""
        if self._sealed:
            raise GraphNotSealed(f"cannot add edge {frm!r} -> {to!r}: sealed")
        self._edges.append((frm, to))

    def add_router(self, source: str, decide: DecideFn, routes: Mapping[str, str]) -> None:
        """Declare a branch at ``source``, merging routes into any existing one.

        Merging rather than replacing is what lets an extension attach a branch
        to a decision node the default tree already declared. The decide
        function is replaced, because a new route is only reachable if the
        function can return its key; an extension that adds a route supplies a
        decide that handles its key and delegates the rest.
        """
        if self._sealed:
            raise GraphNotSealed(f"cannot route from {source!r}: the tree is sealed")
        existing = self._routers.get(source)
        if existing:
            clashes = existing.routes.keys() & routes.keys()
            if clashes:
                raise ValueError(
                    f"route keys already declared at {source!r}: {sorted(clashes)}"
                )
            merged = {**existing.routes, **routes}
        else:
            merged = dict(routes)
        self._routers[source] = Router(source=source, decide=decide, routes=merged)

    # -- closing and building ---------------------------------------------

    def seal(self) -> None:
        """Close registration. After this, nothing may be added.

        This is the one call discovery will make when it has finished walking
        the builtin, user, and project directories.
        """
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def nodes(self) -> list[Node]:
        """Declared nodes, sorted by name, so iteration never depends on order."""
        return sorted(self._nodes.values(), key=lambda n: n.name)

    def node(self, name: str) -> Node:
        """The declaration for one node, or ``UnknownTarget``."""
        try:
            return self._nodes[name]
        except KeyError:
            raise UnknownTarget(f"no node named {name!r}") from None

    def router(self, name: str) -> Router | None:
        """The branch declared at ``name``, or ``None`` for a leaf.

        Having a router is what makes a node a decision node in this tree. It is
        derived rather than stored, so the two cannot disagree.
        """
        return self._routers.get(name)

    @property
    def entry(self) -> str:
        """The node a run enters at. Only meaningful once one is declared."""
        if self._entry is None:
            raise ValueError("no entry node declared; call set_entry()")
        return self._entry

    def edges(self) -> list[tuple[str, str]]:
        """Declared unconditional hops, as ``(from, to)`` pairs."""
        return list(self._edges)

    def next_node(self, name: str) -> str | None:
        """The node a plain, unconditional edge leads to, or ``None``.

        Edges to ``END`` do not count: going to END is stopping, not walking on.
        A branching node uses a router instead, so at most one such edge exists
        per node and ``_validate`` enforces that.
        """
        for frm, to in self._edges:
            if frm == name and to != END:
                return to
        return None

    def parents(self) -> dict[str, str]:
        """Child name -> parent name, derived from edges and routes.

        Built fresh so a declaration cannot contradict itself about its
        neighbours. A node reached from two places is refused here, because the
        walk back up would otherwise have to guess; this is what keeps the
        structure a tree rather than a general graph.
        """
        out: dict[str, str] = {}

        def record(child: str, parent: str) -> None:
            if child in out and out[child] != parent:
                raise ValueError(
                    f"node {child!r} has two parents: {out[child]!r} and {parent!r}; "
                    "back-navigation needs exactly one"
                )
            out[child] = parent

        for frm, to in self._edges:
            if frm == START or to == END:
                continue
            record(to, frm)
        for router in self._routers.values():
            for target in router.routes.values():
                record(target, router.source)
        return out

    def parent(self, name: str) -> str | None:
        """The parent of ``name``, or ``None`` when it is the entry."""
        return self.parents().get(name)

    def compile(self) -> CompiledStateGraph:
        """Build the ``StateGraph`` and compile it. Refuses while unsealed."""
        if not self._sealed:
            raise GraphNotSealed(
                "the tree has unregistered extensions; call seal() before compile()"
            )
        self._validate()

        graph: StateGraph = StateGraph(TreeState)
        for node in self.nodes():
            graph.add_node(node.name, node.fn)

        assert self._entry is not None  # _validate guarantees this
        graph.add_edge(START, self._entry)
        for frm, to in self._edges:
            graph.add_edge(frm, to)
        for router in self._routers.values():
            graph.add_conditional_edges(router.source, router.decide, dict(router.routes))
        return graph.compile()

    def _validate(self) -> None:
        """Check every declaration names a real node before LangGraph sees it."""
        if self._entry is None:
            raise ValueError("no entry node declared; call set_entry()")
        if self._entry not in self._nodes:
            raise UnknownTarget(f"entry {self._entry!r} is not a declared node")

        for frm, to in self._edges:
            if frm != START and frm not in self._nodes:
                raise UnknownTarget(f"edge source {frm!r} is not a declared node")
            if to != END and to not in self._nodes:
                raise UnknownTarget(f"edge target {to!r} is not a declared node")

        for router in self._routers.values():
            if router.source not in self._nodes:
                raise UnknownTarget(f"router source {router.source!r} is not a declared node")
            if not router.routes:
                raise ValueError(f"router at {router.source!r} has no routes")
            for key, target in router.routes.items():
                if target not in self._nodes:
                    raise UnknownTarget(
                        f"route {key!r} at {router.source!r} points at unknown node {target!r}"
                    )

        # A tree, not a general graph: every non-entry node has exactly one parent.
        parents = self.parents()
        for name in self._nodes:
            if name != self._entry and name not in parents:
                raise UnknownTarget(f"node {name!r} is not reachable from {self._entry!r}")

        # A node walks on by one mechanism or the other, never both: a router if
        # it branches, a single plain edge if it does not. Two plain edges would
        # make "the next node" ambiguous for the walk and for /back alike.
        for name in self._nodes:
            plain = [to for frm, to in self._edges if frm == name and to != END]
            if len(plain) > 1:
                raise ValueError(
                    f"node {name!r} has {len(plain)} plain edges; branch with a router instead"
                )
            if plain and name in self._routers:
                raise ValueError(
                    f"node {name!r} has both a router and a plain edge; pick one"
                )


# ---------------------------------------------------------------------------
# The default tree
# ---------------------------------------------------------------------------

# Deterministic keyword sets. The real system puts Jev here; the example keeps
# the classification readable so the routing is the thing on show, not the
# classifier.
_CODE_WORDS = ("write", "code", "implement", "function", "script", "parse", "test")
_REVIEW_WORDS = ("review", "refactor", "fix", "bug", "improve")
_EXPLAIN_WORDS = ("explain", "what", "why", "how", "describe", "summar")


def _xg_origin(state: TreeState) -> dict[str, object]:
    """Entry node: record the goal, then hand off to classification.

    A run always starts here. It contributes the goal and nothing else; which
    branch handles the request is the very next decision.
    """
    return {
        "goal": f"the user wants: {state['prompt']}",
        "trail": [_default_name("origin")],
    }


def _xg_intent(state: TreeState) -> dict[str, object]:
    """First decision node: is this code work or an explanation?

    It writes the chosen key into ``decision`` so the trail shows what the
    router picked, then returns. The edge that is actually followed is the
    conditional edge declared from this node.
    """
    return {"trail": [_default_name("intent")]}


def _decide_intent(state: TreeState) -> str:
    """Route on the prompt. Returns a *route key*, not a node name."""
    text = state["prompt"].lower()
    # Review words are code requests too; the second decision splits them apart.
    if any(word in text for word in (*_CODE_WORDS, *_REVIEW_WORDS)):
        return "code"
    if any(word in text for word in _EXPLAIN_WORDS):
        return "explain"
    return "other"


def _xg_code_intent(state: TreeState) -> dict[str, object]:
    """Second decision node, reached only for code requests: write or review?"""
    return {"trail": [_default_name("code_intent")]}


def _decide_code_intent(state: TreeState) -> str:
    """Split code requests. Any code request that is not a review is a write."""
    text = state["prompt"].lower()
    if any(word in text for word in _REVIEW_WORDS):
        return "review"
    return "write"


def _xg_write_code(state: TreeState) -> dict[str, object]:
    """Leaf: produce new code. The LLM call belongs at this seam."""
    return {
        "trail": [_default_name("write_code")],
        "answer": f"[write code] would implement: {state['prompt']}",
    }


def _xg_review_code(state: TreeState) -> dict[str, object]:
    """Leaf: read existing code and report on it."""
    return {
        "trail": [_default_name("review_code")],
        "answer": f"[review code] would review: {state['prompt']}",
    }


def _xg_explain(state: TreeState) -> dict[str, object]:
    """Leaf: answer a question in prose."""
    return {
        "trail": [_default_name("explain")],
        "answer": f"[explain] would answer: {state['prompt']}",
    }


def _xg_unknown(state: TreeState) -> dict[str, object]:
    """Leaf: nothing matched, so ask for a clearer goal rather than guess."""
    return {
        "trail": [_default_name("unknown")],
        "answer": f"[unclassified] no branch matched: {state['prompt']}",
    }


def default_tree() -> TreeBuilder:
    """The tree xg ships with, before any extension registers.

    Node names are built through ``_default_name`` so the ``xg_`` prefix is
    applied in exactly one place.
    """
    tree = TreeBuilder()
    origin = _default_name("origin")
    intent = _default_name("intent")
    code_intent = _default_name("code_intent")

    tree.set_entry(origin)
    tree.add_node(origin, _xg_origin, summary="Where a run starts; records the goal.")
    tree.add_node(intent, _xg_intent, summary="Decides code work versus explanation.")
    tree.add_node(
        code_intent,
        _xg_code_intent,
        summary="Decides whether a code request writes or reviews.",
    )
    tree.add_node(_default_name("write_code"), _xg_write_code, summary="Writes new code.")
    tree.add_node(_default_name("review_code"), _xg_review_code, summary="Reviews code.")
    tree.add_node(_default_name("explain"), _xg_explain, summary="Explains something.")
    tree.add_node(_default_name("unknown"), _xg_unknown, summary="Fallback for no match.")

    tree.add_edge(origin, intent)
    tree.add_router(
        intent,
        _decide_intent,
        {
            "code": code_intent,
            "explain": _default_name("explain"),
            "other": _default_name("unknown"),
        },
    )
    tree.add_router(
        code_intent,
        _decide_code_intent,
        {
            "write": _default_name("write_code"),
            "review": _default_name("review_code"),
        },
    )
    for leaf in ("write_code", "review_code", "explain", "unknown"):
        tree.add_edge(_default_name(leaf), END)
    return tree


def tree_with(*extensions: Extension) -> TreeBuilder:
    """The default tree with every extension registered, then sealed.

    Registration is open for the whole loop and closed exactly once, at the end.
    Both the one-shot compile below and the walking session use this, so there is
    one definition of "all extensions are done".
    """
    tree = default_tree()
    for extension in extensions:
        extension.register(tree)
    tree.seal()
    return tree


def build_tree(*extensions: Extension) -> CompiledStateGraph:
    """Compose the default tree with extensions, seal it, and compile once.

    This function *is* the ordering rule: every extension registers against an
    open tree, and only the seal lets the build happen. Discovery will be
    another caller of this same sequence.
    """
    return tree_with(*extensions).compile()


# ---------------------------------------------------------------------------
# A worked extension
# ---------------------------------------------------------------------------


class TranslateExtension:
    """Adds a ``translate`` branch to the first decision node.

    The node name has no ``xg_`` prefix, so a trail containing it says plainly
    that an extension put it there.

    Adding a route is not enough on its own: ``_decide_intent`` cannot return
    ``"translate"``, so the route would be dead. The extension supplies a decide
    that answers its own key and delegates every other prompt to the default
    router. That composition is the extension's job, and it is the reason
    ``add_router`` replaces the decide while merging the routes.
    """

    name = "translate"

    def register(self, tree: TreeBuilder) -> None:
        tree.add_node(
            "translate",
            _translate,
            summary="Translates text between languages.",
            origin=self.name,
        )
        tree.add_edge("translate", END)
        tree.add_router(
            _default_name("intent"),
            _decide_intent_with_translate,
            {"translate": "translate"},
        )


def _translate(state: TreeState) -> dict[str, object]:
    return {"trail": ["translate"], "answer": f"[translate] would translate: {state['prompt']}"}


def _decide_intent_with_translate(state: TreeState) -> str:
    """The default intent decision, plus the extension's key."""
    if "translate" in state["prompt"].lower():
        return "translate"
    return _decide_intent(state)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

_DEMO_PROMPTS = (
    "write a parser for the config file",
    "review the session module for bugs",
    "explain how the registry finds parents",
    "translate this into French",
    "banana",
)


def main(argv: list[str] | None = None) -> int:
    """Run prompts through the compiled tree and print the walk.

    ``python -m xg_project.graph.langgraph_tree [--extensions] [prompt ...]``
    """
    parser = argparse.ArgumentParser(description="Run the LangGraph decision tree.")
    parser.add_argument("prompt", nargs="*", help="prompts to route; defaults to a demo set")
    parser.add_argument(
        "--extensions",
        action="store_true",
        help="register TranslateExtension before compiling",
    )
    args = parser.parse_args(argv)

    graph = build_tree(TranslateExtension()) if args.extensions else build_tree()

    for prompt in args.prompt or _DEMO_PROMPTS:
        state = graph.invoke({"prompt": prompt, "trail": []})
        print(f"prompt   : {prompt}")
        print(f"goal     : {state['goal']}")
        print(f"trail    : {' -> '.join(state['trail'])}")
        print(f"answer   : {state['answer']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
