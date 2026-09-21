"""One turn: what a prompt does, by the kind of node it lands on.

The point of these tests is the seam. A generative node answers and the run
stops; a decision node contributes to the context window and hands the choice of
the next node to Jev. A routing that fails must leave the session exactly where it
was, so the user can retype rather than being dropped somewhere arbitrary.
"""

from __future__ import annotations

import pytest

from xg_project.graph import Actor, NodeKind, default_graph
from xg_project.jev import NEXT_NODE, Jev
from xg_project.session import Session
from xg_project.turn import take_turn
from tests.test_jev import FakeClient, response


@pytest.fixture
def session() -> Session:
    return Session(default_graph())


def jev_for(node: str) -> tuple[Jev, FakeClient]:
    client = FakeClient(result=response(choice=node))
    return Jev(client=client), client


# -- generative nodes ------------------------------------------------------


async def test_a_generative_node_answers_and_the_turn_ends(session: Session) -> None:
    session.move("command")
    jev, client = jev_for("command")

    turn = await take_turn(session, jev, "echo hi")

    assert turn.run.kind is NodeKind.GENERATIVE
    assert turn.run.result == "echo hi"
    assert turn.routing is None
    assert turn.moved is None
    assert not turn.descended
    assert session.position == "command"


async def test_a_generative_node_asks_jev_nothing(session: Session) -> None:
    """This is the whole design: an LLM runs once at the leaf, not on the way down."""
    session.move("command")
    jev, client = jev_for("command")

    await take_turn(session, jev, "echo hi")

    assert client.calls == []


async def test_a_generative_node_is_the_end_of_the_context_window(session: Session) -> None:
    session.submit("do the thing")  # contributes at the origin
    session.move("command")
    await take_turn(session, jev_for("command")[0], "echo hi")

    # The origin's contribution is still there; the leaf added nothing.
    assert session.context == ["the user wants: do the thing"]


# -- decision nodes --------------------------------------------------------


async def test_a_decision_node_contributes_then_moves_where_jev_says(session: Session) -> None:
    jev, _ = jev_for("command")

    turn = await take_turn(session, jev, "do the thing")

    assert turn.run.kind is NodeKind.DECISION
    assert session.context == ["the user wants: do the thing"]
    assert turn.routing is not None and turn.routing.ok
    assert turn.moved is not None and turn.moved.ok
    assert turn.descended
    assert session.position == "command"
    assert session.trail == ("none", "command")


async def test_jev_is_offered_the_children_of_the_current_node(session: Session) -> None:
    jev, client = jev_for("command")
    await take_turn(session, jev, "do the thing")

    question = client.calls[0]["questions"][NEXT_NODE]
    assert set(question.criteria) == {"command"}


async def test_jev_sees_the_contribution_the_node_just_made(session: Session) -> None:
    """Jev decides from the context window as it stands, not as it stood before."""
    jev, client = jev_for("command")
    await take_turn(session, jev, "do the thing")

    assert "the user wants: do the thing" in "\n".join(client.calls[0]["state"])


async def test_the_move_jev_makes_is_attributed_to_jev(session: Session) -> None:
    """So the up-only rule applies to it. A routed move is not the user moving."""
    from xg_project.graph import Registry, NodeDeclaration

    registry = Registry()
    registry.add(NodeDeclaration(name="none", level=0, summary="origin", children=("deep",)))
    registry.add(NodeDeclaration(name="deep", level=2, summary="deeper still"))
    session = Session(registry)

    jev, _ = jev_for("deep")
    await take_turn(session, jev, "go deeper")

    assert session.position == "deep"
    # If it had been recorded as a user move, nothing would distinguish the two
    # later; the point is only that Jev was the mover and the policy allowed it.
    assert session.move("none", Actor.JEV).ok is False


# -- routing that fails ----------------------------------------------------


async def test_a_failed_routing_leaves_the_session_where_it_was(session: Session) -> None:
    jev, _ = jev_for("invented")  # not an offered label

    turn = await take_turn(session, jev, "do the thing")

    assert turn.routing is not None and not turn.routing.ok
    assert turn.moved is None
    assert not turn.descended
    assert session.position == "none"
    assert session.trail == ("none",)


async def test_a_failed_routing_keeps_the_contribution_it_already_made(session: Session) -> None:
    """The node did its job before Jev was asked; nothing rolls that back.

    The user retypes, and the goal is contributed again rather than lost, which is
    visible here as a longer context window.
    """
    jev, _ = jev_for("invented")

    await take_turn(session, jev, "do the thing")
    assert session.context == ["the user wants: do the thing"]

    await take_turn(session, jev, "do the thing")
    assert len(session.context) == 2


async def test_a_leaf_turn_that_is_a_decision_node_reports_no_options() -> None:
    """A decision node with no children cannot route, and says so without asking."""
    from xg_project.graph import NodeDeclaration, Registry

    registry = Registry()
    registry.add(NodeDeclaration(name="none", level=0, summary="origin, no children"))
    session = Session(registry)
    jev, client = jev_for("command")

    turn = await take_turn(session, jev, "do the thing")

    assert turn.routing is not None and not turn.routing.ok
    assert "no nodes to route to" in turn.routing.problem
    assert client.calls == []
