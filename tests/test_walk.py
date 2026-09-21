"""Walking the tree: one node per prompt, and /back to the parent."""

from __future__ import annotations

import pytest

from xg_project.graph.langgraph_tree import (
    GraphNotSealed,
    TranslateExtension,
    default_tree,
    tree_with,
)
from xg_project.graph.walk import BACK_COMMANDS, Back, Step, Walk
from xg_project.jev import Routing


@pytest.fixture
def walk() -> Walk:
    return Walk(tree_with())


def test_walking_needs_a_sealed_tree() -> None:
    """A tree that can still grow is not reproducible, so it cannot be walked."""
    with pytest.raises(GraphNotSealed):
        Walk(default_tree())


def test_session_starts_at_the_entry_with_an_empty_trail(walk: Walk) -> None:
    assert walk.position == "xg_origin"
    assert walk.trail == []
    assert walk.path == ["xg_origin"]


def test_one_prompt_walks_exactly_one_node(walk: Walk) -> None:
    step = walk.step("write a parser")
    assert step.frm == "xg_origin"
    assert step.to == "xg_intent"
    assert walk.position == "xg_intent"
    assert walk.trail == ["xg_origin"]


def test_the_same_prompt_walks_deeper_at_each_node(walk: Walk) -> None:
    """The prompt is reused; where it routes is decided by where the session is."""
    prompt = "write a parser for the config file"
    assert walk.step(prompt).to == "xg_intent"
    assert walk.step(prompt).to == "xg_code_intent"
    assert walk.step(prompt).to == "xg_write_code"


def test_origin_records_the_goal(walk: Walk) -> None:
    walk.step("write a parser")
    assert walk.goal == "the user wants: write a parser"


def test_a_leaf_answers_and_does_not_move(walk: Walk) -> None:
    for _ in range(3):
        walk.step("write a parser for the config file")
    assert walk.position == "xg_write_code"

    step = walk.step("write a parser for the config file")
    assert step.kind == "generative"
    assert step.to is None
    assert walk.position == "xg_write_code"
    assert "would implement" in step.result


def test_prompt_never_moves_up(walk: Walk) -> None:
    """Deposits are forward only; a prompt at the leaf stays at the leaf."""
    for _ in range(3):
        walk.step("write a parser")
    position = walk.position
    walk.step("review this instead")
    assert walk.position == position


def test_back_returns_to_the_parent(walk: Walk) -> None:
    walk.step("write a parser")
    result = walk.back()
    assert result.ok
    assert result.frm == "xg_intent"
    assert result.to == "xg_origin"
    assert walk.position == "xg_origin"


def test_back_pops_the_node_it_leaves_from_the_trail(walk: Walk) -> None:
    prompt = "write a parser"
    walk.step(prompt)
    walk.step(prompt)
    assert walk.trail == ["xg_origin", "xg_intent"]

    walk.back()
    assert walk.trail == ["xg_origin"]
    assert walk.path == ["xg_origin", "xg_intent"]


def test_back_at_the_origin_is_refused(walk: Walk) -> None:
    result = walk.back()
    assert isinstance(result, Back)
    assert not result.ok
    assert "nothing above it" in (result.reason or "")
    assert walk.position == "xg_origin"


def test_b_is_an_alias_for_back(walk: Walk) -> None:
    assert "/b" in BACK_COMMANDS
    walk.step("write a parser")
    outcome = walk.command("/b")
    assert isinstance(outcome, Back)
    assert outcome.ok


def test_back_command_is_a_command_and_a_prompt_is_not(walk: Walk) -> None:
    assert isinstance(walk.command("/back"), Back)
    assert walk.command("write a parser") is None


def test_extension_branch_can_be_walked_and_left() -> None:
    walk = Walk(tree_with(TranslateExtension()))
    walk.step("translate this")
    assert walk.position == "xg_intent"
    step = walk.step("translate this")
    assert isinstance(step, Step)
    assert step.to == "translate"

    back = walk.back()
    assert back.ok
    assert back.to == "xg_intent"


def test_a_two_deep_branch_reaches_its_leaf(walk: Walk) -> None:
    """The explain branch is two nodes deep, so it is reached in two prompts."""
    walk.step("explain the registry")
    step = walk.step("explain the registry")
    assert step.to == "xg_explain"
    assert walk.position == "xg_explain"

    answer = walk.step("explain the registry")
    assert "would answer" in answer.result


# -- Jev routing -----------------------------------------------------------


class StubJev:
    """A Jev that answers from a fixed choice and records what it was asked."""

    def __init__(self, *, node: str | None = None, problem: str | None = None) -> None:
        self.node = node
        self.problem = problem
        self.calls: list[dict] = []

    async def decide(self, *, current, context, prompt, options):
        self.calls.append(
            {"current": current, "context": context, "prompt": prompt, "options": options}
        )
        return Routing(node=self.node, problem=self.problem, confidence=0.9)


async def test_jev_chooses_the_child_and_the_walk_moves() -> None:
    jev = StubJev(node="xg_explain")
    walk = Walk(tree_with(), jev=jev)

    step = await walk.astep("write a parser")
    assert step.to == "xg_intent"

    step = await walk.astep("write a parser")
    assert step.to == "xg_explain"
    assert walk.position == "xg_explain"
    assert step.frm == "xg_intent"


async def test_jev_is_offered_the_children_of_the_current_node() -> None:
    """The option set is the graph's own children, labelled by their summaries."""
    jev = StubJev(node="xg_explain")
    walk = Walk(tree_with(), jev=jev)
    await walk.astep("anything")
    await walk.astep("anything")

    call = jev.calls[0]
    assert call["current"] == "xg_intent"
    assert set(call["options"]) == {"xg_code_intent", "xg_explain", "xg_unknown"}
    assert all(text.strip() for text in call["options"].values())


async def test_jev_is_not_asked_where_there_is_no_choice() -> None:
    """The origin descends by a plain edge, so there is nothing to ask about."""
    jev = StubJev(node="xg_explain")
    walk = Walk(tree_with(), jev=jev)
    await walk.astep("write a parser")
    assert jev.calls == []
    assert walk.position == "xg_intent"


async def test_jev_sees_the_goal_as_context_on_a_later_decision() -> None:
    jev = StubJev(node="xg_code_intent")
    walk = Walk(tree_with(), jev=jev)
    await walk.astep("write a parser")
    await walk.astep("write a parser")
    assert jev.calls[0]["context"] == ["the user wants: write a parser"]


async def test_a_failed_routing_does_not_move_the_session() -> None:
    """Failure is a value: the session stays put and says why."""
    jev = StubJev(problem="could not reach Jev")
    walk = Walk(tree_with(), jev=jev)
    step = await walk.astep("write a parser")
    assert step.to == "xg_intent"

    step = await walk.astep("write a parser")
    assert step.to is None
    assert step.reason == "could not reach Jev"
    assert walk.position == "xg_intent"


async def test_astep_without_jev_is_the_keyword_walk() -> None:
    walk = Walk(tree_with())
    assert (await walk.astep("write a parser")).to == "xg_intent"
    assert (await walk.astep("write a parser")).to == "xg_code_intent"


async def test_jev_choosing_a_non_child_is_refused_by_the_walk() -> None:
    """The walk builds the option set, so it will not move outside it.

    ``Jev.decide`` refuses an un-offered label on its own; this guards the seam
    for any object that implements the same ``decide`` shape, so the walk cannot
    be walked somewhere the graph does not connect.
    """
    jev = StubJev(node="xg_write_code")  # two levels down, not a child of xg_intent
    walk = Walk(tree_with(), jev=jev)
    await walk.astep("write a parser")

    step = await walk.astep("write a parser")
    assert step.to is None
    assert "was not offered" in (step.reason or "")
    assert walk.position == "xg_intent"
