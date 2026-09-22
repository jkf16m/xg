"""The graph: what nodes exist, who may move between them, and what they read."""

from __future__ import annotations

import pytest

from xg_project.graph import (
    ANSWER,
    COMMAND,
    EDIT,
    FILTER,
    ORIGIN,
    SELECT_MODULE,
    SORT,
    XG_PREFIX,
    Actor,
    NodeDeclaration,
    NodeKind,
    Registry,
    UnknownNode,
    default_graph,
    xg,
)


@pytest.fixture
def graph() -> Registry:
    return default_graph()


ALL_NODES = [ORIGIN, COMMAND, SELECT_MODULE, FILTER, SORT, EDIT, ANSWER]


#: Words that turn a node's own description into an instruction to its parent.
#: A summary says what the node does; it never tells the router when to pick it.
ROUTING_ADVICE = ("pick this", "pick it", "use this when", "choose this when")


def test_default_graph_has_the_seven_built_in_nodes(graph: Registry) -> None:
    # Sorted by genericity then name, so the two level-4 leaves sort as ANSWER, EDIT.
    assert [n.name for n in graph.sorted()] == [
        ORIGIN,
        COMMAND,
        SELECT_MODULE,
        FILTER,
        SORT,
        ANSWER,
        EDIT,
    ]


def test_every_summary_describes_the_node_and_not_its_parent(graph: Registry) -> None:
    """A node does not know who links to it, so it cannot say when to route there.

    The router matches a request against what each node does. A summary written
    for one particular parent is wrong for the next one — this is the property
    that keeps descriptions generic and the routing honest.
    """
    for node in graph.sorted():
        lowered = node.summary.lower()
        for phrase in ROUTING_ADVICE:
            assert phrase not in lowered, f"{node.name} tells its parent when to pick it"


def test_built_in_keys_are_namespaced(graph: Registry) -> None:
    """Users add nodes and edges, so xg's own keys must not collide with theirs."""
    assert XG_PREFIX == "_XG_"
    assert all(name.startswith(XG_PREFIX) for name in ALL_NODES)
    assert xg("ORIGIN") == "_XG_ORIGIN"


def test_origin_is_the_most_generic_node(graph: Registry) -> None:
    assert graph.origin.name == ORIGIN
    assert graph.origin.level == 0


def test_command_is_reached_from_the_origin(graph: Registry) -> None:
    assert graph.ancestors(COMMAND) == [ORIGIN, COMMAND]


def test_the_edit_path_is_origin_module_filter_sort_edit(graph: Registry) -> None:
    assert graph.ancestors(EDIT) == [ORIGIN, SELECT_MODULE, FILTER, SORT, EDIT]


def test_the_answer_path_is_origin_module_filter_sort_answer(graph: Registry) -> None:
    assert graph.ancestors(ANSWER) == [ORIGIN, SELECT_MODULE, FILTER, SORT, ANSWER]


def test_criteria_is_name_to_summary(graph: Registry) -> None:
    """The bridge to Jev: the option set is the graph itself."""
    criteria = graph.criteria()
    assert set(criteria) == set(ALL_NODES)
    assert all(text.strip() for text in criteria.values())


# -- node kinds ------------------------------------------------------------


def test_the_origin_is_a_decision_node(graph: Registry) -> None:
    """Jev chooses between the workflows from here, so the origin is where it is asked."""
    assert graph.get(ORIGIN).kind is NodeKind.DECISION


def test_the_edit_leaves_are_generative_nodes(graph: Registry) -> None:
    assert graph.get(COMMAND).kind is NodeKind.GENERATIVE
    assert graph.get(EDIT).kind is NodeKind.GENERATIVE
    assert graph.get(ANSWER).kind is NodeKind.GENERATIVE


def test_the_file_steps_are_decision_nodes(graph: Registry) -> None:
    """Filter and sort work, then hand the run to their single child."""
    assert graph.get(FILTER).kind is NodeKind.DECISION
    assert graph.get(SORT).kind is NodeKind.DECISION


def test_the_gated_leaves_propose_a_command_the_user_must_confirm(graph: Registry) -> None:
    assert graph.get(COMMAND).proposes is True
    assert graph.get(EDIT).proposes is True


def test_the_file_steps_do_not_propose(graph: Registry) -> None:
    """They introduce state and move on; only the mutating leaves are gated."""
    assert graph.get(ORIGIN).proposes is False
    assert graph.get(FILTER).proposes is False
    assert graph.get(SORT).proposes is False
    # ANSWER is a leaf like EDIT, but reading proposes nothing to accept.
    assert graph.get(ANSWER).proposes is False


def test_only_a_generative_node_may_propose() -> None:
    """A decision node is not a leaf, so it has no executor to propose with."""
    with pytest.raises(ValueError, match="only a generative node can propose"):
        NodeDeclaration(name="bad", level=1, summary="x", kind=NodeKind.DECISION, proposes=True)


def test_options_are_the_children_of_the_current_node(graph: Registry) -> None:
    assert set(graph.options(ORIGIN)) == {COMMAND, SELECT_MODULE}


def test_the_module_step_sits_above_the_file_steps(graph: Registry) -> None:
    """Choosing the context comes before reading it, and is the only way down."""
    assert graph.get(SELECT_MODULE).kind is NodeKind.DECISION
    assert set(graph.options(SELECT_MODULE)) == {FILTER}


def test_a_single_child_is_still_an_option(graph: Registry) -> None:
    assert set(graph.options(FILTER)) == {SORT}


def test_the_sort_fork_offers_both_project_leaves(graph: Registry) -> None:
    """A question and an edit share the read, and part company here."""
    assert set(graph.options(SORT)) == {EDIT, ANSWER}


def test_a_leaf_offers_no_options(graph: Registry) -> None:
    """Nothing to descend to means the run stops."""
    assert graph.options(COMMAND) == {}
    assert graph.options(EDIT) == {}
    assert graph.options(ANSWER) == {}


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
    assert graph.options(ORIGIN)[COMMAND] == graph.get(COMMAND).summary


def test_the_tree_shows_each_nodes_kind(graph: Registry) -> None:
    tree = graph.render_tree(ORIGIN)
    assert "decision" in tree
    assert "generative" in tree


# -- movement policy -------------------------------------------------------


def test_user_may_move_down(graph: Registry) -> None:
    assert graph.check_move(ORIGIN, FILTER, Actor.USER) is None


def test_user_may_move_up(graph: Registry) -> None:
    """The graph is bidirectional for the user; only Jev is restricted."""
    assert graph.check_move(FILTER, ORIGIN, Actor.USER) is None


def test_jev_may_move_down(graph: Registry) -> None:
    assert graph.check_move(ORIGIN, FILTER, Actor.JEV) is None


def test_jev_may_not_move_toward_the_origin(graph: Registry) -> None:
    reason = graph.check_move(FILTER, ORIGIN, Actor.JEV)
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
    assert graph.check_move(ORIGIN, ORIGIN, Actor.USER) == f"already at {ORIGIN!r}"


def test_moving_to_an_unknown_node_is_refused(graph: Registry) -> None:
    assert graph.check_move(ORIGIN, "nope", Actor.USER) == "unknown node 'nope'"


# -- registry rules --------------------------------------------------------


def test_add_refuses_to_clobber(graph: Registry) -> None:
    clash = NodeDeclaration(name=COMMAND, level=9, summary="mine", origin="project")
    with pytest.raises(ValueError, match="already defined"):
        graph.add(clash)


def test_add_can_override_when_asked(graph: Registry) -> None:
    replacement = NodeDeclaration(name=COMMAND, level=9, summary="mine", origin="project")
    graph.add(replacement, override=True)
    assert graph.get(COMMAND).origin == "project"
    assert graph.get(COMMAND).level == 9


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


# -- parents ---------------------------------------------------------------


def test_parent_of_a_child_is_the_node_that_lists_it(graph: Registry) -> None:
    assert graph.parent(COMMAND) == ORIGIN
    assert graph.parent(FILTER) == SELECT_MODULE


def test_parent_follows_the_edit_path(graph: Registry) -> None:
    assert graph.parent(SORT) == FILTER
    assert graph.parent(EDIT) == SORT
    assert graph.parent(ANSWER) == SORT


def test_the_origin_has_no_parent(graph: Registry) -> None:
    assert graph.parent(ORIGIN) is None


def test_parent_of_an_unknown_node_raises(graph: Registry) -> None:
    with pytest.raises(UnknownNode):
        graph.parent("nope")


# -- rendering -------------------------------------------------------------


def test_tree_marks_the_current_node(graph: Registry, plain) -> None:
    drawn = plain(graph.render_tree(FILTER))
    assert f"◆ {FILTER}" in drawn
    assert "← you are here" in drawn
    assert f"○ {ORIGIN}" in drawn
    assert "origin" in drawn


def test_tree_is_stable_across_calls(graph: Registry) -> None:
    """Rendering is a pure function of graph and position."""
    assert graph.render_tree(ORIGIN) == graph.render_tree(ORIGIN)
