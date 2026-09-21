"""The session: position, movement, and what a prompt does where it arrives."""

from __future__ import annotations

import pytest

from xg_project.graph import Actor, Registry, default_graph
from xg_project.session import Session


@pytest.fixture
def session() -> Session:
    return Session(default_graph())


def test_a_session_starts_at_the_origin(session: Session) -> None:
    assert session.position == "none"


def test_moving_down_updates_position_and_trail(session: Session) -> None:
    moved = session.move("command")
    assert moved.ok
    assert session.position == "command"
    assert session.trail == ("none", "command")


def test_the_trail_starts_where_the_session_did(session: Session) -> None:
    assert session.trail == ("none",)


def test_moving_up_is_allowed_for_the_user(session: Session) -> None:
    session.move("command")
    assert session.move("none", Actor.USER).ok
    assert session.position == "none"


def test_jev_cannot_move_toward_the_origin(session: Session) -> None:
    session.move("command")
    moved = session.move("none", Actor.JEV)
    assert not moved.ok
    assert session.position == "command"
    assert "only the user" in moved.reason


def test_a_refused_move_leaves_position_untouched(session: Session) -> None:
    moved = session.move("nowhere")
    assert not moved.ok
    assert session.position == "none"
    assert session.trail == ("none",)


def test_moving_to_an_empty_name_is_refused(session: Session) -> None:
    moved = session.move("   ")
    assert not moved.ok
    assert moved.reason == "no node named"


def test_node_reflects_position(session: Session) -> None:
    assert session.node.name == "none"
    session.move("command")
    assert session.node.name == "command"


# -- prompts ---------------------------------------------------------------


def test_a_prompt_at_the_origin_is_a_goal(session: Session) -> None:
    run = session.submit("get the tests green")
    assert run.is_goal
    assert session.goal == "get the tests green"


def test_the_goal_survives_a_move(session: Session) -> None:
    session.submit("get the tests green")
    session.move("command")
    assert session.goal == "get the tests green"


def test_a_prompt_at_command_is_a_task_not_a_goal(session: Session) -> None:
    session.move("command")
    run = session.submit("echo hi")
    assert not run.is_goal
    assert run.result == "echo hi"
    assert session.goal is None


def test_the_origin_contributes_the_goal_to_the_context(session: Session) -> None:
    """The origin is a decision node, so what it produces is context, not an answer."""
    run = session.submit("do the thing")
    assert run.result == "the user wants: do the thing"
    assert session.context == ["the user wants: do the thing"]


def test_a_generative_node_adds_nothing_to_the_context(session: Session) -> None:
    session.move("command")
    session.submit("echo hi")
    assert session.context == []


def test_run_records_the_kind_of_node_it_came_from(session: Session) -> None:
    from xg_project.graph import NodeKind

    assert session.submit("a goal").kind is NodeKind.DECISION
    session.move("command")
    assert session.submit("a task").kind is NodeKind.GENERATIVE


def test_a_node_without_a_handler_says_so() -> None:
    """Descriptors with no behaviour are shown, not silently ignored."""
    from xg_project.graph import NodeDeclaration

    registry = Registry()
    registry.add(NodeDeclaration(name="none", level=0, summary="x"))
    session = Session(registry)
    assert "nothing to do" in session.submit("anything").result
