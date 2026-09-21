"""The LangGraph decision tree: routing, the deferred compile, and extension."""

from __future__ import annotations

import pytest
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph

from xg_project.graph.langgraph_tree import (
    DEFAULT_PREFIX,
    GraphNotSealed,
    TranslateExtension,
    TreeBuilder,
    UnknownTarget,
    build_tree,
    default_tree,
)


def walk(prompt: str, *extensions) -> dict:
    """Compile a tree and run one prompt through it."""
    graph: CompiledStateGraph = build_tree(*extensions)
    return graph.invoke({"prompt": prompt, "trail": []})


def test_every_default_node_is_prefixed() -> None:
    """The prefix is how a trail says whether a node is default or extension."""
    names = [node.name for node in default_tree().nodes()]
    assert names
    assert all(name.startswith(f"{DEFAULT_PREFIX}_") for name in names)


def test_compile_refuses_while_the_tree_is_open() -> None:
    """The design point: no compiled structure exists until extensions are done."""
    tree = default_tree()
    with pytest.raises(GraphNotSealed):
        tree.compile()


def test_sealed_tree_compiles() -> None:
    tree = default_tree()
    tree.seal()
    assert isinstance(tree.compile(), CompiledStateGraph)


def test_registration_after_seal_is_refused() -> None:
    tree = default_tree()
    tree.seal()
    with pytest.raises(GraphNotSealed):
        tree.add_node("late", lambda state: {}, summary="too late")


def test_code_request_reaches_the_code_branch() -> None:
    state = walk("write a parser for the config file")
    assert state["trail"] == [
        "xg_origin",
        "xg_intent",
        "xg_code_intent",
        "xg_write_code",
    ]


def test_review_is_split_from_write_at_the_second_decision() -> None:
    state = walk("review the session module for bugs")
    assert state["trail"][-1] == "xg_review_code"


def test_explanation_request_takes_the_prose_branch() -> None:
    state = walk("explain how the registry finds parents")
    assert state["trail"] == ["xg_origin", "xg_intent", "xg_explain"]


def test_unmatched_prompt_falls_back_rather_than_guessing() -> None:
    state = walk("banana")
    assert state["trail"][-1] == "xg_unknown"


def test_origin_records_the_goal() -> None:
    state = walk("write a parser")
    assert state["goal"] == "the user wants: write a parser"


def test_trail_is_accumulated_by_reducer_not_overwritten() -> None:
    """Without the Annotated reducer the last node's list would replace the rest."""
    state = walk("write a parser")
    assert len(state["trail"]) == 4


def test_extension_adds_a_reachable_branch() -> None:
    state = walk("translate this into French", TranslateExtension())
    assert state["trail"] == ["xg_origin", "xg_intent", "translate"]


def test_extension_branch_is_absent_without_the_extension() -> None:
    state = walk("translate this into French")
    assert "translate" not in state["trail"]


def test_extension_routes_delegate_back_to_the_default_decision() -> None:
    """The extension owns one key; everything else still routes as before."""
    state = walk("write a parser", TranslateExtension())
    assert state["trail"][-1] == "xg_write_code"


def test_duplicate_route_key_is_refused() -> None:
    tree = default_tree()
    with pytest.raises(ValueError, match="already declared"):
        tree.add_router("xg_intent", lambda state: "code", {"code": "xg_explain"})


def test_edge_to_an_unknown_node_fails_at_compile() -> None:
    tree = default_tree()
    tree.add_edge("xg_origin", "does_not_exist")
    tree.seal()
    with pytest.raises(UnknownTarget):
        tree.compile()


def test_route_to_an_unknown_node_fails_at_compile() -> None:
    tree = TreeBuilder()
    tree.set_entry("only")
    tree.add_node("only", lambda state: {}, summary="sole node")
    tree.add_router("only", lambda state: "gone", {"gone": "missing"})
    tree.seal()
    with pytest.raises(UnknownTarget):
        tree.compile()


def test_leaf_edges_stop_the_run() -> None:
    """Every default leaf declares an edge to END; reachable leaves are terminal."""
    tree = default_tree()
    leaves = {name for _, name in tree.edges() if name == END}
    assert leaves


def test_a_node_with_two_parents_is_refused() -> None:
    """Back-navigation needs one parent; a second parent would make /back a guess."""
    tree = TreeBuilder()
    tree.set_entry("top")
    tree.add_node("top", lambda state: {}, summary="root")
    tree.add_node("left", lambda state: {}, summary="left")
    tree.add_node("right", lambda state: {}, summary="right")
    tree.add_node("shared", lambda state: {}, summary="shared leaf")
    tree.add_router("top", lambda state: "l", {"l": "left", "r": "right"})
    tree.add_edge("left", "shared")
    tree.add_edge("right", "shared")
    tree.seal()
    with pytest.raises(ValueError, match="two parents"):
        tree.compile()


def test_a_router_and_a_plain_edge_on_one_node_are_refused() -> None:
    """Two ways to walk on from one node make 'the next node' ambiguous."""
    tree = TreeBuilder()
    tree.set_entry("a")
    tree.add_node("a", lambda state: {}, summary="first")
    tree.add_node("b", lambda state: {}, summary="second")
    tree.add_edge("a", "b")
    tree.add_router("a", lambda state: "x", {"x": "b"})
    tree.seal()
    with pytest.raises(ValueError, match="both a router and a plain edge"):
        tree.compile()
