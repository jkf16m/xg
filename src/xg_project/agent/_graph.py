"""The xg agent graph: classify a prompt, research the repository, then route.

LangGraph owns the routing. The ``classify`` node runs the Jev first filter
(one ``system_one`` call). A conditional edge sends the run to one downstream
node; for anything that is not a command and not too ambiguous, that node is
``local_research``, a forced step that finds the files the request needs.

    START -> classify -> {local_research, run_command, clarify} -> END

``local_research`` is the generic research step: it applies to questions and
changes alike, and it is what decides scope, because the number of files it
finds is how far-reaching the request turned out to be. Its output feeds the
generative agent, which is not built yet.
"""

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typesafe_sdk import TypeSafeClient

from xg_project.agent._state import AgentState
from xg_project.jev import Classification, RequestKind, build_client, classify
from xg_project.research import local_research

NODES: tuple[str, ...] = ("local_research", "run_command", "clarify")
"""Nodes the classify node can route to."""


def route_for(classification: Classification) -> str:
    """Pick the next node from a classification.

    Pure, so the routing policy can be reasoned about and tested without a
    graph or an API call. Order matters: an unsure classification is clarified
    before anything else, and a command is run rather than researched.
    """
    if classification.needs_clarification:
        return "clarify"
    if classification.kind is RequestKind.COMMAND:
        return "run_command"
    return "local_research"


def _terminal(name: str) -> Callable[[AgentState], AgentState]:
    """Build a placeholder node that records which route was taken."""

    def node(state: AgentState) -> AgentState:
        return {"route": name}

    node.__name__ = f"{name}_node"
    return node


def build_graph(
    *, client: TypeSafeClient | None = None, model: str | None = None
) -> CompiledStateGraph:
    """Compile the graph. ``client``/``model`` are captured by the Jev nodes."""
    if client is None:
        client = build_client(model)

    def classify_node(state: AgentState) -> AgentState:
        return {"classification": classify(state["prompt"], client=client, model=model)}

    def research_node(state: AgentState) -> AgentState:
        result = local_research(state["prompt"], client=client, model=model)
        return {
            "research": result,
            "relevant_files": result.files,
            "route": "local_research",
        }

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("local_research", research_node)
    graph.add_node("run_command", _terminal("run_command"))
    graph.add_node("clarify", _terminal("clarify"))
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
    """Classify one prompt, research it, and route it."""
    graph = build_graph(client=client, model=model)
    return graph.invoke({"prompt": prompt})
