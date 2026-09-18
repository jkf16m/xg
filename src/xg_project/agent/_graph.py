"""The xg agent graph: classify a prompt, then route it.

LangGraph owns the routing. The ``classify`` node runs the Jev first filter
(one ``system_one`` call), and a conditional edge sends the run to a
downstream node chosen by :func:`route_for`. The downstream nodes are
placeholders until their pipelines exist; wiring one in means replacing a node
body, not re-deciding the edges.

    START -> classify -> {clarify, answer, select_files, run_command} -> END
"""

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typesafe_sdk import TypeSafeClient

from xg_project.agent._state import AgentState
from xg_project.jev import Classification, RequestKind, classify

NODES: tuple[str, ...] = ("clarify", "answer", "select_files", "run_command")
"""Downstream nodes the classify node can route to."""


def route_for(classification: Classification) -> str:
    """Pick the next node from a classification.

    Pure, so it can be unit-tested and reasoned about without running a graph.
    Order matters: an unsure classification is clarified before anything else.
    """
    if classification.needs_clarification:
        return "clarify"
    if classification.kind is RequestKind.QUESTION:
        return "answer"
    if classification.kind is RequestKind.COMMAND:
        return "run_command"
    if classification.is_change:
        return "select_files"
    return "answer"


def _terminal(name: str) -> Callable[[AgentState], AgentState]:
    """Build a placeholder node that records which route was taken."""

    def node(state: AgentState) -> AgentState:
        return {"route": name}

    node.__name__ = f"{name}_node"
    return node


def build_graph(
    *, client: TypeSafeClient | None = None, model: str | None = None
) -> CompiledStateGraph:
    """Compile the graph. ``client``/``model`` are captured by the classify node."""
    if client is None:
        from xg_project.jev import build_client

        client = build_client(model)

    def classify_node(state: AgentState) -> AgentState:
        return {"classification": classify(state["prompt"], client=client, model=model)}

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    for name in NODES:
        graph.add_node(name, _terminal(name))
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        lambda state: route_for(state["classification"]),
        list(NODES),
    )
    for name in NODES:
        graph.add_edge(name, END)
    return graph.compile()


def run(
    prompt: str, *, client: TypeSafeClient | None = None, model: str | None = None
) -> AgentState:
    """Classify one prompt and route it, returning the final graph state."""
    graph = build_graph(client=client, model=model)
    return graph.invoke({"prompt": prompt})
