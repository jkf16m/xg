"""The graph: what nodes exist, and who may move between them."""

from __future__ import annotations

import pytest

from xg_project.graph import (
    ORIGIN,
    Actor,
    NodeDeclaration,
    NodeKind,
    Registry,
    UnknownNode,
    default_graph,
)


@pytest.fixture
def graph() -> Registry:
    return default_graph()


def test_default_graph_has_the_two_nodes(graph: Registry) -> None:
    assert [n.name for n in graph.sorted()] == ["none", "command"]


def test_origin_is_the_most_generic_node(graph: Registry) -> None:
    assert graph.origin.name == ORIGIN
    assert graph.origin.level == 0


def test_command_is_reached_from_the_origin(graph: Registry) -> None:
    assert graph.ancestors("command") == ["none", "command"]


def test_criteria_is_name_to_summary(graph: Registry) -> None:
    """The bridge to Jev: the option set is the graph itself."""
    criteria = graph.criteria()
    assert set(criteria) == {"none", "command"}
    assert all(text.strip() for text in criteria.values())


def test_command_needs_no_context_window(graph: Registry) -> None:
    assert graph.get("command").builds_context is False


# -- node kinds ------------------------------------------------------------


def test_the_origin_is_a_decision_node(graph: Registry) -> None:
    """Jev chooses from the origin, so the origin is where it is asked."""
    assert graph.get(ORIGIN).kind is NodeKind.DECISION


def test_command_is_a_generative_node(graph: Registry) -> None:
    assert graph.get("command").kind is NodeKind.GENERATIVE


def test_generative_nodes_do_not_build_context(graph: Registry) -> None:
    """Kind decides this, so a node cannot be one kind and behave like the other."""
    for node in graph.sorted():
        assert node.builds_context is (node.kind is NodeKind.DECISION)


def test_options_are_the_children_of_the_current_node(graph: Registry) -> None:
    assert set(graph.options(ORIGIN)) == {"command"}


def test_a_leaf_offers_no_options(graph: Registry) -> None:
    """Nothing to descend to means nothing to ask."""
    assert graph.options("command") == {}


def test_options_never_include_the_current_node(graph: Registry) -> None:
    for node in graph.sorted():
        assert node.name not in graph.options(node.name)


def test_options_exclude_a_move_the_policy_would_refuse() -> None:
    """An option that is always wrong is a chance to get the answer wrong.

    A same-level child is not a descent, so Jev cannot make that move and must
    not be offered it.
    """
    registry = Registry()
    registry.add(NodeDeclaration(name="none", level=0, summary="origin", children=("side",)))
    registry.add(NodeDeclaration(name="side", level=0, summary="same level as none"))
    assert registry.options("none") == {}


def test_options_text_is_the_summary(graph: Registry) -> None:
    """The criteria text is the model's only input, so it is the node's summary."""
    assert graph.options(ORIGIN) == {"command": graph.get("command").summary}


def test_the_tree_shows_each_nodes_kind(graph: Registry) -> None:
    tree = graph.render_tree(ORIGIN)
    assert "decision" in tree
    assert "generative" in tree


# -- movement policy -------------------------------------------------------


def test_user_may_move_down(graph: Registry) -> None:
    assert graph.check_move("none", "command", Actor.USER) is None


def test_user_may_move_up(graph: Registry) -> None:
    """The graph is bidirectional for the user; only Jev is restricted."""
    assert graph.check_move("command", "none", Actor.USER) is None


def test_jev_may_move_down(graph: Registry) -> None:
    assert graph.check_move("none", "command", Actor.JEV) is None


def test_jev_may_not_move_toward_the_origin(graph: Registry) -> None:
    reason = graph.check_move("command", "none", Actor.JEV)
    assert reason is not None
    assert "only the user" in reason


def test_jev_may_not_move_sideways() -> None:
    """A lateral move is not a descent, so it is not Jev's to make."""
    registry = Registry()
    for name in ("none", "a", "b"):
        registry.add(NodeDeclaration(name=name, level=0 if name == "none" else 1, summary="x"))
    reason = registry.check_move("a", "b", Actor.JEV)
    assert reason is not None
    assert "sideways" in reason


def test_moving_to_the_current_node_is_refused(graph: Registry) -> None:
    assert graph.check_move("none", "none", Actor.USER) == "already at 'none'"


def test_moving_to_an_unknown_node_is_refused(graph: Registry) -> None:
    assert graph.check_move("none", "nope", Actor.USER) == "unknown node 'nope'"


# -- registry rules --------------------------------------------------------


def test_add_refuses_to_clobber(graph: Registry) -> None:
    clash = NodeDeclaration(name="command", level=3, summary="mine", origin="project")
    with pytest.raises(ValueError, match="already defined"):
        graph.add(clash)


def test_add_can_override_when_asked(graph: Registry) -> None:
    replacement = NodeDeclaration(name="command", level=3, summary="mine", origin="project")
    graph.add(replacement, override=True)
    assert graph.get("command").origin == "project"
    assert graph.get("command").level == 3


def test_declaration_rejects_a_negative_level() -> None:
    with pytest.raises(ValueError, match="level must be"):
        NodeDeclaration(name="bad", level=-1, summary="x")


def test_declaration_rejects_an_empty_summary() -> None:
    with pytest.raises(ValueError, match="summary must not be empty"):
        NodeDeclaration(name="bad", level=1, summary="   ")


def test_get_raises_unknown_node(graph: Registry) -> None:
    with pytest.raises(UnknownNode):
        graph.get("nope")


def test_ambiguous_origin_is_an_error() -> None:
    """Two nodes at the lowest level is a defect, not a coin flip."""
    registry = Registry()
    registry.add(NodeDeclaration(name="a", level=0, summary="x"))
    registry.add(NodeDeclaration(name="b", level=0, summary="y"))
    with pytest.raises(ValueError, match="ambiguous origin"):
        _ = registry.origin


def test_unreachable_node_is_an_error() -> None:
    registry = Registry()
    registry.add(NodeDeclaration(name="none", level=0, summary="x"))
    registry.add(NodeDeclaration(name="orphan", level=1, summary="y"))
    with pytest.raises(ValueError, match="not reachable"):
        registry.ancestors("orphan")


def test_a_parent_cycle_is_an_error_not_a_hang() -> None:
    registry = Registry()
    registry.add(NodeDeclaration(name="none", level=0, summary="x", children=("a",)))
    registry.add(NodeDeclaration(name="a", level=1, summary="y", children=("b",)))
    registry.add(NodeDeclaration(name="b", level=2, summary="z", children=("a",)))
    with pytest.raises(ValueError, match="cycle"):
        registry.ancestors("b")


# -- rendering -------------------------------------------------------------


def test_tree_marks_the_current_node(graph: Registry, plain) -> None:
    drawn = plain(graph.render_tree("command"))
    assert "◆ command" in drawn
    assert "← you are here" in drawn
    assert "○ none" in drawn
    assert "origin" in drawn


def test_tree_is_stable_across_calls(graph: Registry) -> None:
    """Rendering is a pure function of graph and position."""
    assert graph.render_tree("none") == graph.render_tree("none")
