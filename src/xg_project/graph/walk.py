"""Walking the tree one node per prompt, and back up with ``/back``.

This is a different execution model from ``build_tree()``'s one-shot run, and the
difference is the whole point of xg's design.

A one-shot LangGraph run enters at ``START``, flows forward through whatever
routing chooses, and stops at ``END``. Nothing persists; the run is over.

Here the session *sits at a node* and stays there. A prompt is the downward
move: submitting it runs the current node and follows its router to exactly one
child. The prompt never moves up, so a prompt is unidirectional. Coming back is
the user's alone, through the ``/back`` (or ``/b``) command, which returns to the
parent node. That mirrors the movement rule in the registry graph — descent is
driven, ascent is manual — but here it is position-based, so the session can sit
at a node indefinitely and the same prompt can be reused at each step.

Position is an explicit value, and the tree is walked by name rather than by an
executor. That is deliberate: LangGraph's run model runs forward to ``END`` and
has no notion of a parent to return to, so it is the wrong engine for this walk.
The compiled graph is still the topology — it is what validates that every route
names a real node and that every node has exactly one parent — and the walk is
the executor over it.

Node functions are reused unchanged. They return partial states with the same
reducer semantics LangGraph applies, so the walk merges ``trail`` by appending
and other keys by replacement, exactly as a run would.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field

from xg_project.graph.langgraph_tree import (
    GraphNotSealed,
    Router,
    TreeBuilder,
    TreeState,
)
from xg_project.jev import Jev

#: The command that moves back to the parent node. Both spellings are accepted.
BACK_COMMANDS = ("/back", "/b")


@dataclass(frozen=True)
class Step:
    """What one prompt did at one node.

    ``to`` is where the walk moved, or ``None`` at a leaf. A decision node's
    ``result`` is the route it chose; a generative node's is its answer.
    """

    frm: str
    """The node the prompt landed on."""

    prompt: str
    result: str
    kind: str
    """``"decision"`` when the node routed onward, ``"generative"`` at a leaf."""

    to: str | None = None
    """Where the walk moved, or ``None`` when there was nowhere to go."""

    reason: str | None = None
    """Why no move happened, as a sentence for the user."""


@dataclass(frozen=True)
class Back:
    """The outcome of ``/back`` or ``/b``."""

    frm: str
    to: str | None
    ok: bool
    reason: str | None = None


@dataclass
class Walk:
    """A session that walks a sealed tree, one node per prompt.

    The trail is the path from the entry to the current position: the nodes that
    have actually run. ``/back`` pops the node it leaves, so the invariant
    ``trail + [position]`` always describes the path.

    The tree must be sealed, because a walk that could have nodes appear
    underneath it would not be reproducible, and reproducing a path is the point.
    """

    tree: TreeBuilder
    jev: Jev | None = None
    """When set, a decision node asks Jev which child to take. When ``None`` the
    tree's own keyword router decides, so the walk works with no key and no
    network — which is also what the offline tests exercise."""

    position: str = ""
    trail: list[str] = field(default_factory=list)
    data: dict[str, object] = field(default_factory=dict)
    """Everything a node contributed that is not the trail: the goal, the files a
    read collected, the subset a sort kept, the answer. A node's keys are merged
    in here and handed forward on the next visit, so a linear pipeline like
    origin -> collect -> sort -> response carries its work between nodes without
    each node knowing the shape of the whole state."""

    @property
    def goal(self) -> str | None:
        """The origin's reading of the request, once the origin has run."""
        value = self.data.get("goal")
        return value if isinstance(value, str) else None

    @property
    def answer(self) -> str | None:
        """The last generative node's result, if one has run in this position."""
        value = self.data.get("answer")
        return value if isinstance(value, str) else None

    @property
    def files(self) -> dict[str, str]:
        """Every file the last read collected, path -> content."""
        value = self.data.get("files")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def selected(self) -> dict[str, str]:
        """The subset the last sort kept."""
        value = self.data.get("selected")
        return dict(value) if isinstance(value, dict) else {}

    def __post_init__(self) -> None:
        if not self.tree.sealed:
            raise GraphNotSealed(
                "walking needs a sealed tree: register every extension, then seal"
            )
        if not self.position:
            self.position = self.tree.entry

    async def astep(self, prompt: str) -> Step:
        """One prompt, one node, awaiting the node and any Jev routing it needs.

        This is the general entry point: it awaits an async node function, and at
        a decision node it asks Jev when one is attached and falls back to the
        tree's own router when not. :meth:`step` is the synchronous subset, for
        trees whose nodes are all synchronous.

        A routing that fails is a value, not an exception: the session does not
        move, and ``reason`` carries the sentence. The node's contribution was
        already merged, so a failed route leaves the context as it stood.
        """
        node = self.tree.node(self.position)
        produced = await self._run(node, prompt)

        router = self.tree.router(node.name)
        if router is not None and self.jev is not None:
            options = self.options(router)
            routing = await self.jev.decide(
                current=node.name,
                context=self.context,
                prompt=prompt,
                options=options,
            )
            if not routing.ok:
                return self._stopped(node, prompt, routing.problem)
            target = routing.node
            assert target is not None  # guaranteed by routing.ok
            if target not in options:
                # Jev.decide already refuses an un-offered label, so this guards
                # the seam rather than the model: the walk builds the option set,
                # and it will not move to anything outside it.
                return self._stopped(node, prompt, f"jev chose {target!r}, which was not offered")
            return self._moved(node, prompt, target, f"jev chose {routing.explain()}")

        local = self._local_route(node, produced, prompt)
        if local is None:
            return self._leaf(node, prompt)
        target, result = local
        return self._moved(node, prompt, target, result)

    def step(self, prompt: str) -> Step:
        """Run ``prompt`` at the current node and walk at most one node down.

        The synchronous subset of :meth:`astep`: the node function must return a
        mapping, not an awaitable, and routing uses the tree's own router rather
        than Jev. A node that is async raises rather than being silently skipped.
        """
        node = self.tree.node(self.position)
        produced = node.fn(self._state(prompt))
        if inspect.isawaitable(produced):
            # Close it so the coroutine is not left un-awaited; the caller wants
            # astep, and the error says so.
            if inspect.iscoroutine(produced):
                produced.close()
            raise RuntimeError(
                f"node {node.name!r} is async; use Walk.astep(), not Walk.step()"
            )
        self._merge(produced)

        local = self._local_route(node, produced, prompt)
        if local is None:
            return self._leaf(node, prompt)
        target, result = local
        return self._moved(node, prompt, target, result)

    async def _run(self, node, prompt: str) -> dict[str, object]:
        """Visit a node: run it, await it if needed, merge, and return the delta."""
        # A result belongs to the visit that produced it, so nothing carries over
        # from a previous node; a leaf reads what its own function wrote.
        self.data.pop("answer", None)
        produced = node.fn(self._state(prompt))
        if inspect.isawaitable(produced):
            produced = await produced
        self._merge(produced)
        return produced

    def _moved(self, node, prompt: str, target: str, result: str) -> Step:
        """Record a move and return the step that describes it."""
        self.position = target
        return Step(frm=node.name, prompt=prompt, result=result, kind="decision", to=target)

    def _stopped(self, node, prompt: str, reason: str | None) -> Step:
        """A step that did not move, with the sentence explaining why."""
        return Step(
            frm=node.name, prompt=prompt, result="", kind="decision", to=None, reason=reason
        )

    def _local_route(
        self, node, produced: dict[str, object], prompt: str
    ) -> tuple[str, str] | None:
        """Route without Jev: a router decides, or a plain edge is followed.

        Returns ``(target, result)``, or ``None`` when the node has nowhere to go
        and is therefore a leaf.
        """
        router = self.tree.router(node.name)
        if router is not None:
            key = router.decide(self._state(prompt))
            target = router.routes[key]
            return target, str(produced.get("goal") or f"decided {key!r} -> {target}")
        plain = self.tree.next_node(node.name)
        if plain is not None:
            return plain, str(produced.get("goal") or f"moved to {plain}")
        return None

    def back(self) -> Back:
        """Move to the parent node. The only way up, and it is the user's."""
        parent = self.tree.parent(self.position)
        if parent is None:
            return Back(
                frm=self.position,
                to=None,
                ok=False,
                reason=f"already at the origin {self.position!r}; nothing above it",
            )
        frm = self.position
        # The node we are leaving ran, so it is the last entry in the trail. The
        # parent we return to has not run in this position, so it must not be.
        if self.trail:
            self.trail.pop()
        self.data.pop("answer", None)
        self.position = parent
        return Back(frm=frm, to=parent, ok=True)

    def command(self, line: str) -> Step | Back | None:
        """Interpret a line as a command, or ``None`` when it is a prompt.

        Only ``/back`` and ``/b`` are commands. Anything else is a prompt, which
        is what keeps the prompt free to contain slashes.
        """
        if line.strip() in BACK_COMMANDS:
            return self.back()
        return None

    @property
    def path(self) -> list[str]:
        """The full path from the entry to the current position, inclusive."""
        return [*self.trail, self.position]

    @property
    def context(self) -> list[str]:
        """What the nodes on the current path have contributed, in order.

        Derived from the path rather than accumulated, so re-entering a node
        produces the same context every time. Only the origin contributes so far
        — the goal — but the shape is the one `jev.build_state` already takes, so
        a node that starts contributing later needs no change here.
        """
        return [self.goal] if self.goal else []

    def options(self, router: Router) -> dict[str, str]:
        """The option set Jev is asked about at a decision node.

        Its children, labelled by their summaries — exactly the shape a Jev
        ``Choice`` takes. Taking it from the router's targets is what keeps the
        option set and the graph the same object, and it means Jev cannot be
        offered a node the walk could not move to: a router's targets are the
        children, and a sibling is never among them.
        """
        return {target: self.tree.node(target).summary for target in router.routes.values()}

    def _leaf(self, node, prompt: str) -> Step:
        """The step a generative node produces: an answer, and nowhere to walk."""
        return Step(
            frm=node.name,
            prompt=prompt,
            result=self.answer or "",
            kind="generative",
            to=None,
            reason="leaf: nothing below it, use /back to return",
        )

    def _state(self, prompt: str) -> TreeState:
        """Build the state a node or router is shown, from the session.

        Everything a node contributed earlier is carried forward verbatim, so a
        node reads ``files`` or ``selected`` the way it would read any other key.
        """
        state: TreeState = dict(self.data)  # type: ignore[assignment]
        state["prompt"] = prompt
        state["trail"] = list(self.trail)
        return state

    def _merge(self, produced: dict[str, object]) -> None:
        """Merge a node's partial state, applying the reducer ``trail`` declares.

        LangGraph would do this between nodes. Without an executor the walk does
        it, and it has to match ``TreeState``: ``trail`` appends, everything else
        replaces.
        """
        for key, value in produced.items():
            if key == "trail":
                self.trail.extend(value)  # type: ignore[arg-type]
            else:
                self.data[key] = value


def main(argv: list[str] | None = None) -> int:
    """Walk a recorded session to show the model.

    ``python -m xg_project.graph.walk [--extensions] [--jev] [prompt ...]``
    """
    import argparse

    from xg_project.graph.langgraph_tree import TranslateExtension, tree_with

    parser = argparse.ArgumentParser(description="Walk the decision tree one node per prompt.")
    parser.add_argument("prompts", nargs="*", help="prompts submitted in order")
    parser.add_argument("--extensions", action="store_true", help="register TranslateExtension")
    parser.add_argument(
        "--jev",
        action="store_true",
        help="ask Jev to route at decision nodes instead of the keyword router",
    )
    args = parser.parse_args(argv)

    jev = Jev() if args.jev else None
    walk = Walk(
        tree_with(TranslateExtension()) if args.extensions else tree_with(),
        jev=jev,
    )
    prompts = args.prompts or [
        "write a parser for the config file",
        "write a parser for the config file",
        "write a parser for the config file",
        "/back",
        "explain how the registry finds parents",
    ]
    return asyncio.run(_demo(walk, prompts, jev))


async def _demo(walk: Walk, lines: list[str], jev: Jev | None) -> int:
    try:
        for line in lines:
            outcome = walk.command(line)
            if isinstance(outcome, Back):
                note = outcome.reason if not outcome.ok else f"{outcome.frm} -> {outcome.to}"
                print(f"{line:<45} back: {note}")
            else:
                step = await walk.astep(line)
                print(f"{line:<45} at {step.frm}: {step.result}")
            print(f"{'':<45} path: {' -> '.join(walk.path)}")
    finally:
        if jev is not None:
            await jev.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
