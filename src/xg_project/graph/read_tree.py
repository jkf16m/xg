"""The read-only graph: xg_origin -> xg_collect -> xg_sort -> xg_response.

A question about the project, answered from the project. Four nodes, one
direction, no branching: each one hands the next a little more than it received.

``xg_origin``
    Where the user states the question. It contributes the goal and nothing
    else. A prompt arriving here moves the walk to collect.

``xg_collect``
    The deterministic half. It reads the whole directory tree — every file not
    excluded by ``.gitignore`` or ``.xgignore`` — and puts the result into the
    state as ``files``: path -> the file's own content, nothing wrapped around
    it. No model is involved, so the same tree always produces the same map.
    The read is what the later nodes are handed; this node decides nothing.

``xg_sort``
    The model half. Jev is asked one ``Noul`` question per file, with the
    collected map as the state and the file path as the question's name, and
    keeps every file whose probability of being relevant is above the threshold.
    What comes out is ``selected``: the same shape as ``files``, smaller. A file
    Jev was not asked about, or did not answer for, is not selected.

``xg_response``
    The last step. It assembles the selected files and the question into one
    text and hands that to the responder — the LLM seam. With no responder
    configured it returns the assembled text unchanged, so the graph is
    runnable and testable without a provider; wiring one is the remaining open
    piece, and this is the only place it is needed.

State flows through ``Walk.data``: whatever a node returns is merged and handed
to the next visit. ``files`` is collected once and read twice — once by sort, and
again by response — which is why it is kept in the state rather than re-read.

The graph is walked, not run to completion: each prompt advances one node, so
the same question is submitted at the origin, then at collect, then at sort, and
the answer arrives at response. Sort and response are async, so the tree as a
whole is walked with ``Walk.astep``; ``Walk.step`` stops at sort with an error
rather than pretending to have asked.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path

from langgraph.graph import END

from xg_project.graph.langgraph_tree import NodeFn, TreeBuilder, TreeState
from xg_project.jev import DEFAULT_SELECT_BATCH_BYTES, RELEVANCE_THRESHOLD, Jev
from xg_project.read import DEFAULT_MAX_BYTES, read_tree

XG_ORIGIN = "xg_origin"
XG_COLLECT = "xg_collect"
XG_SORT = "xg_sort"
XG_RESPONSE = "xg_response"

#: The final step's seam. It receives the assembled context and returns the
#: answer; it may be async, because an LLM client usually is.
Responder = Callable[[str], "str | Awaitable[str]"]


def _origin(state: TreeState) -> dict[str, object]:
    """State the question, and contribute the goal. Nothing is read here."""
    prompt = str(state.get("prompt", ""))
    return {"goal": f"the user asks: {prompt}", "trail": [XG_ORIGIN]}


def _collect(root: Path | str | None, max_bytes: int) -> NodeFn:
    """Build the node that reads the tree.

    The root and the size bound are captured here rather than read from the
    state, so what a run reads is fixed when the graph is built and cannot be
    redirected by anything that arrives later.
    """

    def collect(state: TreeState) -> dict[str, object]:
        tree = read_tree(root if root is not None else ".", max_bytes=max_bytes)
        return {
            "files": tree.files,
            "skipped": list(tree.skipped),
            "trail": [XG_COLLECT],
        }

    return collect


def _sort(jev: Jev | None, threshold: float, batch_bytes: int) -> NodeFn:
    """Build the node that asks Jev which files matter.

    An async node: the question is a network round trip, and the walk awaits it.
    No Jev means the file set is left alone but nothing is selected, because
    selecting is exactly the judgement that was unavailable — passing the whole
    tree on would be answering a different question.
    """

    async def sort(state: TreeState) -> dict[str, object]:
        files = state.get("files") or {}
        if jev is None:
            return {
                "trail": [XG_SORT],
                "selected": {},
                "scores": {},
                "problem": "no Jev attached, so nothing can be selected",
            }
        selection = await jev.select(
            files=files,
            prompt=str(state.get("prompt", "")),
            threshold=threshold,
            batch_bytes=batch_bytes,
        )
        return {
            "trail": [XG_SORT],
            "selected": dict(selection.files),
            "scores": dict(selection.scores),
            "problem": selection.problem or "",
        }

    return sort


def build_prompt(state: TreeState) -> str:
    """Assemble the question and the selected files into one text.

    Plain and labelled rather than JSON: this is the last thing a model reads,
    and a path above its own contents is the only structure the answer needs.
    """
    selected = state.get("selected") or {}
    lines = [f"the user asks: {state.get('prompt', '')}", ""]
    if not selected:
        lines.append("(no files were selected as relevant)")
    for path, content in selected.items():
        lines.append(f"--- {path} ---")
        lines.append(content)
    return "\n".join(lines)


def _response(respond: Responder | None) -> NodeFn:
    """Build the final node: assemble the context, then answer it.

    With no responder the assembled text is the answer. That keeps the node
    honest about what it would be handing over, and makes the graph runnable
    end to end before a provider is chosen.
    """

    async def response(state: TreeState) -> dict[str, object]:
        context = build_prompt(state)
        if respond is None:
            answer: str = context
        else:
            result = respond(context)
            answer = await result if inspect.isawaitable(result) else result
        return {"trail": [XG_RESPONSE], "answer": answer}

    return response


def default_read_tree(
    *,
    jev: Jev | None = None,
    root: Path | str | None = None,
    respond: Responder | None = None,
    threshold: float = RELEVANCE_THRESHOLD,
    max_bytes: int = DEFAULT_MAX_BYTES,
    batch_bytes: int = DEFAULT_SELECT_BATCH_BYTES,
) -> TreeBuilder:
    """Build the read-only graph. Unsealed, so extensions can still register."""
    tree = TreeBuilder()
    tree.set_entry(XG_ORIGIN)
    tree.add_node(
        XG_ORIGIN,
        _origin,
        summary="Where the user states the question about the project.",
    )
    tree.add_node(
        XG_COLLECT,
        _collect(root, max_bytes),
        summary=(
            "Read the whole directory tree, honouring .gitignore and .xgignore, "
            "and put every file's contents into the state as path -> content."
        ),
    )
    tree.add_node(
        XG_SORT,
        _sort(jev, threshold, batch_bytes),
        summary=(
            "Ask Jev, once per file, whether that file is relevant to the "
            "question, and keep the files above the threshold."
        ),
    )
    tree.add_node(
        XG_RESPONSE,
        _response(respond),
        summary="Assemble the selected files and answer the question from them.",
    )

    tree.add_edge(XG_ORIGIN, XG_COLLECT)
    tree.add_edge(XG_COLLECT, XG_SORT)
    tree.add_edge(XG_SORT, XG_RESPONSE)
    tree.add_edge(XG_RESPONSE, END)
    return tree


def read_tree_with(*extensions, **kwargs) -> TreeBuilder:
    """The read-only graph with every extension registered, then sealed."""
    tree = default_read_tree(**kwargs)
    for extension in extensions:
        extension.register(tree)
    tree.seal()
    return tree


def main(argv: list[str] | None = None) -> int:
    """Walk the read-only graph against a real project.

    ``python -m xg_project.graph.read_tree [--root DIR] [--threshold F]
    [--no-jev] [question ...]``
    """
    import argparse
    import asyncio

    from xg_project.graph.walk import Walk

    parser = argparse.ArgumentParser(description="Walk the read-only graph.")
    parser.add_argument("question", nargs="*", help="the question to answer")
    parser.add_argument("--root", default=".", help="the tree to read (default: .)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=RELEVANCE_THRESHOLD,
        help=f"relevance probability above which a file is kept (default: {RELEVANCE_THRESHOLD})",
    )
    parser.add_argument(
        "--no-jev",
        action="store_true",
        help="skip the relevance question, to see what the read collected",
    )
    args = parser.parse_args(argv)

    question = " ".join(args.question) or "what does this project do?"
    return asyncio.run(_walk_graph(question, args.root, args.threshold, args.no_jev))


async def _walk_graph(question: str, root: str, threshold: float, no_jev: bool) -> int:
    from xg_project.graph.walk import Walk
    from xg_project.jev import Jev

    jev = None if no_jev else Jev()
    try:
        walk = Walk(read_tree_with(jev=jev, root=root, threshold=threshold))
        # Four prompts, one per node: the walk is one node per submission, and the
        # same question is what gets resubmitted at each one.
        for _ in range(4):
            step = await walk.astep(question)
            print(f"{step.frm:12} -> {step.to or 'answer'}")

        print(f"\ncollected {len(walk.files)} files")
        if walk.data.get("skipped"):
            print(f"skipped {len(walk.data['skipped'])} (too large or not text)")
        problem = walk.data.get("problem")
        if problem:
            print(f"problem: {problem}")

        scores = walk.data.get("scores") or {}
        if scores:
            print(f"\nkept {len(walk.selected)} above {threshold:.2f}:")
            for path, score in sorted(scores.items(), key=lambda kv: -kv[1]):
                mark = "+" if path in walk.selected else " "
                print(f"  {mark} {score:.3f}  {path}")

        print("\n--- answer ---")
        print(walk.answer)
    finally:
        if jev is not None:
            await jev.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
