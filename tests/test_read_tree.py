"""The read-only graph: origin -> collect -> sort -> response."""

from __future__ import annotations

from pathlib import Path

import pytest

from xg_project.graph.read_tree import (
    XG_COLLECT,
    XG_ORIGIN,
    XG_RESPONSE,
    XG_SORT,
    build_prompt,
    default_read_tree,
    read_tree_with,
)
from xg_project.graph.walk import Walk
from xg_project.jev import RELEVANCE_THRESHOLD, Selection


class StubJev:
    """A Jev that keeps whichever files were listed as relevant."""

    def __init__(self, keep: tuple[str, ...] = ()) -> None:
        self.keep = set(keep)
        self.calls: list[dict] = []

    async def select(self, *, files, prompt, threshold=RELEVANCE_THRESHOLD):
        self.calls.append({"files": sorted(files), "prompt": prompt, "threshold": threshold})
        selected = {path: content for path, content in files.items() if path in self.keep}
        scores = {path: (0.9 if path in self.keep else 0.1) for path in files}
        return Selection(
            files=selected, scores=scores, model="stub", threshold=threshold
        )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A small project: one file to keep, one to ignore, one to neither."""
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("def parse(): ...\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("OTHER = 1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("IGNORED = 1\n", encoding="utf-8")
    return tmp_path


async def walk_all(walk: Walk) -> Walk:
    """Walk the four nodes, returning the session at the end."""
    for _ in range(4):
        await walk.astep("where is the parser?")
    return walk


# -- shape -----------------------------------------------------------------


def test_every_default_node_is_prefixed() -> None:
    names = [node.name for node in default_read_tree().nodes()]
    assert names == [XG_COLLECT, XG_ORIGIN, XG_RESPONSE, XG_SORT]
    assert all(name.startswith("xg_") for name in names)


def test_the_graph_is_the_four_nodes_in_a_line() -> None:
    tree = default_read_tree()
    assert tree.entry == XG_ORIGIN
    assert tree.next_node(XG_ORIGIN) == XG_COLLECT
    assert tree.next_node(XG_COLLECT) == XG_SORT
    assert tree.next_node(XG_SORT) == XG_RESPONSE
    assert tree.next_node(XG_RESPONSE) is None  # its only edge is to END


def test_each_node_has_the_one_parent() -> None:
    tree = default_read_tree()
    assert tree.parent(XG_COLLECT) == XG_ORIGIN
    assert tree.parent(XG_SORT) == XG_COLLECT
    assert tree.parent(XG_RESPONSE) == XG_SORT
    assert tree.parent(XG_ORIGIN) is None


def test_the_graph_cannot_compile_until_it_is_sealed() -> None:
    from xg_project.graph.langgraph_tree import GraphNotSealed

    with pytest.raises(GraphNotSealed):
        default_read_tree().compile()


# -- walking one node per prompt -------------------------------------------


async def test_each_prompt_advances_exactly_one_node(project: Path) -> None:
    walk = Walk(read_tree_with(jev=StubJev(), root=project))
    assert (await walk.astep("q")).to == XG_COLLECT
    assert (await walk.astep("q")).to == XG_SORT
    assert (await walk.astep("q")).to == XG_RESPONSE
    assert walk.path == [XG_ORIGIN, XG_COLLECT, XG_SORT, XG_RESPONSE]


async def test_back_returns_to_the_parent(project: Path) -> None:
    walk = Walk(read_tree_with(jev=StubJev(), root=project))
    await walk.astep("q")
    await walk.astep("q")
    assert walk.back().to == XG_COLLECT
    assert walk.position == XG_COLLECT


def test_step_refuses_an_async_node(project: Path) -> None:
    """sort is a network call, and a synchronous walk cannot pretend otherwise."""
    walk = Walk(read_tree_with(jev=StubJev(), root=project))
    walk.step("q")  # origin is synchronous
    walk.step("q")  # collect is synchronous
    with pytest.raises(RuntimeError, match="is async"):
        walk.step("q")


# -- collect ---------------------------------------------------------------


async def test_collect_reads_the_tree_into_the_state(project: Path) -> None:
    walk = Walk(read_tree_with(jev=StubJev(), root=project))
    await walk.astep("q")
    await walk.astep("q")
    assert set(walk.files) == {"keep.py", "other.py"}
    assert walk.files["keep.py"] == "def parse(): ...\n"


async def test_collect_reports_what_it_skipped(project: Path) -> None:
    (project / "blob.bin").write_bytes(b"\xff\xfe")
    walk = Walk(read_tree_with(jev=StubJev(), root=project))
    await walk.astep("q")
    await walk.astep("q")
    assert any("blob.bin" in entry for entry in walk.data["skipped"])


async def test_the_origin_contributes_the_question_as_the_goal(project: Path) -> None:
    walk = Walk(read_tree_with(jev=StubJev(), root=project))
    await walk.astep("where is the parser?")
    assert walk.goal == "the user asks: where is the parser?"


# -- sort ------------------------------------------------------------------


async def test_sort_asks_jev_about_every_collected_file(project: Path) -> None:
    jev = StubJev(keep=("keep.py",))
    walk = Walk(read_tree_with(jev=jev, root=project))
    await walk_all(walk)

    assert jev.calls[0]["files"] == ["keep.py", "other.py"]
    assert jev.calls[0]["prompt"] == "where is the parser?"
    assert set(walk.selected) == {"keep.py"}


async def test_sort_keeps_the_scores_of_files_it_dropped(project: Path) -> None:
    walk = Walk(read_tree_with(jev=StubJev(keep=("keep.py",)), root=project))
    await walk_all(walk)
    assert walk.data["scores"] == {"keep.py": 0.9, "other.py": 0.1}


async def test_without_jev_nothing_is_selected_and_the_reason_is_kept(
    project: Path,
) -> None:
    walk = Walk(read_tree_with(root=project))
    await walk_all(walk)
    assert walk.selected == {}
    assert "no Jev attached" in str(walk.data["problem"])


# -- response --------------------------------------------------------------


async def test_the_answer_is_built_from_the_selected_files(project: Path) -> None:
    walk = Walk(read_tree_with(jev=StubJev(keep=("keep.py",)), root=project))
    await walk_all(walk)
    assert "def parse(): ..." in (walk.answer or "")
    assert "OTHER = 1" not in (walk.answer or "")


async def test_a_responder_receives_the_assembled_context(project: Path) -> None:
    seen: list[str] = []

    async def respond(context: str) -> str:
        seen.append(context)
        return "the parser is in keep.py"

    walk = Walk(
        read_tree_with(jev=StubJev(keep=("keep.py",)), root=project, respond=respond)
    )
    await walk_all(walk)
    assert walk.answer == "the parser is in keep.py"
    assert "--- keep.py ---" in seen[0]


def test_the_assembled_context_labels_each_path() -> None:
    context = build_prompt(
        {"prompt": "q", "selected": {"a.py": "A", "b.py": "B"}}
    )
    assert "the user asks: q" in context
    assert "--- a.py ---\nA" in context
    assert "--- b.py ---\nB" in context


def test_the_assembled_context_says_when_nothing_was_selected() -> None:
    context = build_prompt({"prompt": "q", "selected": {}})
    assert "no files were selected" in context
