"""The session: position, movement, and the state nodes introduce."""

from __future__ import annotations

import pytest

from xg_project.graph import (
    COMMAND,
    FILTER,
    ORIGIN,
    SELECT_MODULE,
    SORT,
    Actor,
    NodeDeclaration,
    Registry,
    default_graph,
)
from xg_project.session import Session, brief


@pytest.fixture
def session() -> Session:
    return Session(default_graph())


def test_a_session_starts_at_the_origin(session: Session) -> None:
    assert session.position == ORIGIN


def test_moving_down_updates_position_and_trail(session: Session) -> None:
    moved = session.move(FILTER)
    assert moved.ok
    assert session.position == FILTER
    # The trail is the path through the graph, so it names the step between the
    # origin and FILTER even though the run jumped straight over it.
    assert session.trail == (ORIGIN, SELECT_MODULE, FILTER)


def test_the_trail_starts_where_the_session_did(session: Session) -> None:
    assert session.trail == (ORIGIN,)


def test_the_trail_shrinks_when_the_run_moves_up(session: Session) -> None:
    """A path, not a history: going back up makes it shorter."""
    session.move(FILTER)
    session.move(SORT)
    assert session.move(FILTER, Actor.USER).ok
    assert session.trail == (ORIGIN, SELECT_MODULE, FILTER)


def test_the_trail_stops_growing_however_often_the_run_comes_back(session: Session) -> None:
    for _ in range(20):
        session.move(COMMAND)
        session.move(ORIGIN)
    assert session.trail == (ORIGIN,)


def test_moving_up_is_allowed_for_the_user(session: Session) -> None:
    session.move(FILTER)
    assert session.move(ORIGIN, Actor.USER).ok
    assert session.position == ORIGIN


def test_jev_cannot_move_toward_the_origin(session: Session) -> None:
    session.move(FILTER)
    moved = session.move(ORIGIN, Actor.JEV)
    assert not moved.ok
    assert session.position == FILTER
    assert "only the user" in moved.reason


def test_a_refused_move_leaves_position_untouched(session: Session) -> None:
    moved = session.move("nowhere")
    assert not moved.ok
    assert session.position == ORIGIN
    assert session.trail == (ORIGIN,)


def test_moving_to_an_empty_name_is_refused(session: Session) -> None:
    moved = session.move("   ")
    assert not moved.ok
    assert moved.reason == "no node named"


def test_node_reflects_position(session: Session) -> None:
    assert session.node.name == ORIGIN
    session.move(FILTER)
    assert session.node.name == FILTER


# -- state -----------------------------------------------------------------


def test_a_session_starts_with_no_state(session: Session) -> None:
    assert session.state == {}


def test_a_node_records_its_state_under_its_own_key(session: Session) -> None:
    session.record(FILTER, {"files": {"a.py": "x"}})
    assert session.state[FILTER] == {"files": {"a.py": "x"}}


def test_running_a_node_again_replaces_its_entry(session: Session) -> None:
    """One entry per node: the state is where the run is now, not a history."""
    session.record(FILTER, {"files": {"a.py": "x"}})
    session.record(FILTER, {"files": {"b.py": "y"}})
    assert session.state[FILTER] == {"files": {"b.py": "y"}}
    assert len(session.state) == 1


def test_the_goal_is_the_origins_state(session: Session) -> None:
    assert session.goal is None
    session.record(ORIGIN, "the user wants: get the tests green")
    assert session.goal == "the user wants: get the tests green"


def test_the_goal_survives_a_move(session: Session) -> None:
    session.record(ORIGIN, "the user wants: get the tests green")
    session.move(FILTER)
    assert session.goal == "the user wants: get the tests green"


def test_the_context_is_the_state_labelled_by_node(session: Session) -> None:
    """Routing text is derived, so it cannot drift from the state it summarizes."""
    session.record(ORIGIN, "the user wants: edit a file")
    session.move(FILTER)
    session.record(FILTER, {"files": {"a.py": "x", "b.py": "y"}})
    assert session.context == [f"{ORIGIN}: the user wants: edit a file", f"{FILTER}: 2 files"]


# -- the state is the path, not a history ----------------------------------


def test_moving_up_drops_the_state_of_the_node_it_leaves(session: Session) -> None:
    """SORT ranked files FILTER has since discarded; EDIT must not find it."""
    session.move(FILTER)
    session.record(FILTER, {"files": {"a.py": "x"}})
    session.move(SORT)
    session.record(SORT, {"files": {"a.py": "x"}})

    session.move(FILTER, Actor.USER)

    assert FILTER in session.state
    assert SORT not in session.state


def test_moving_sideways_drops_the_branch_it_stepped_off(session: Session) -> None:
    """FILTER is not on the path to COMMAND, so its files are not current either."""
    session.move(FILTER)
    session.record(FILTER, {"files": {"a.py": "x"}})

    session.move(COMMAND, Actor.USER)

    assert FILTER not in session.state


def test_the_states_on_the_path_survive_a_move(session: Session) -> None:
    session.record(ORIGIN, "the user wants: edit a file")
    session.move(FILTER)
    session.record(FILTER, {"files": {"a.py": "x"}})
    session.move(SORT)

    assert set(session.state) == {ORIGIN, FILTER}


def test_coming_back_down_does_not_revive_the_state_it_dropped(session: Session) -> None:
    session.move(FILTER)
    session.move(SORT)
    session.record(SORT, {"files": {"a.py": "x"}})
    session.move(ORIGIN, Actor.USER)
    session.move(SORT)

    assert session.state == {}

def test_an_unreachable_node_is_not_pruned_against_a_path_it_never_had() -> None:
    """A defect in a user's graph must not make a legal move throw away state."""
    registry = Registry()
    registry.add(NodeDeclaration(name="root", level=0, summary="root"))
    registry.add(NodeDeclaration(name="island", level=1, summary="island"))
    session = Session(registry)
    session.record("root", "hello")

    session.move("island")

    assert session.trail == ("island",)
    assert session.state == {"root": "hello"}


# -- reading one entry -----------------------------------------------------


def test_brief_reads_a_string_as_itself() -> None:
    assert brief("the user wants: x") == "the user wants: x"


def test_brief_counts_files() -> None:
    assert brief({"files": {"a": "x", "b": "y"}}) == "2 files"


def test_brief_names_a_goal_key() -> None:
    assert brief({"goal": "the user wants: x"}) == "the user wants: x"


def test_brief_surfaces_a_problem() -> None:
    assert brief({"problem": "no root"}) == "problem: no root"


def test_brief_prefers_a_value_that_renders_itself() -> None:
    """A gate or a proposal knows how to describe itself; brief asks it."""

    class SelfDescribing:
        def preview(self) -> str:
            return "pending: edit a.py"

    assert brief(SelfDescribing()) == "pending: edit a.py"


def test_brief_falls_back_for_an_unknown_shape() -> None:
    """A user's node is still legible to Jev without this function knowing it."""
    assert brief(42) == "42"
    assert brief({"anything": True}) == "{'anything': True}"


def test_an_unreachable_origin_state_is_not_a_goal(session: Session) -> None:
    session.record(ORIGIN, {"files": {}})
    assert session.goal is None


def test_a_custom_graph_session_uses_its_own_origin() -> None:
    registry = Registry()
    registry.add(NodeDeclaration(name="root", level=0, summary="root"))
    session = Session(registry)
    session.record("root", "hello")
    assert session.goal == "hello"
