"""The xg agent graph: classify, research, confirm, and only then generate.

LangGraph owns the routing; Jev owns the decisions. The graph uses no tools: a
turn advances through Jev decisions -- ``classify``, and the per-file questions
inside ``local_research`` -- and stops at a human confirmation before the final
generative step. An LLM is reached only at that last moment, to fill in
generative information; nothing before it needs one.

    START -> classify -> {local_research, run_command, clarify}
    local_research -> confirm -> {classify, END}

``confirm`` is a human-in-the-loop ``interrupt``: the run pauses and surfaces
the research result, and the client resumes it with ``{"action": "continue"}``
or ``{"action": "reprompt", "prompt": "..."}``. A reprompt loops back to
``classify`` with the new prompt; continuing stops the run here, because the
generative node does not exist yet.

Interrupts need a checkpointer, so ``build_graph`` compiles with an
``InMemorySaver`` whose serializer is given an explicit allowlist for the
dataclasses the nodes put in state. Without the allowlist the checkpointer
logs an "unregistered type" warning on every deserialize.

``local_research`` is the generic research step: it applies to questions and
changes alike, and it is what decides scope, because the number of files it
finds is how far-reaching the request turned out to be.
"""

from collections.abc import Callable, Mapping
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, interrupt
from typesafe_sdk import TypeSafeClient

from xg_project.agent._state import AgentState
from xg_project.jev import Classification, RequestKind, build_client, classify
from xg_project.research import local_research

NODES: tuple[str, ...] = ("local_research", "run_command", "clarify")
"""Nodes the classify node can route to."""

CONTINUE = "continue"
REPROMPT = "reprompt"

ALLOWED_STATE_TYPES: tuple[tuple[str, str], ...] = (
    ("xg_project.jev._classify", "ChoiceOutcome"),
    ("xg_project.jev._classify", "ScoreOutcome"),
    ("xg_project.jev._classify", "NoulOutcome"),
    ("xg_project.jev._classify", "Classification"),
    ("xg_project.research", "Research"),
)
"""State types the checkpointer serializes, as ``(module, name)`` pairs."""


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


def route_after_confirm(state: Mapping[str, Any]) -> str:
    """Send a reprompt back to classification; otherwise stop."""
    return "classify" if state.get("decision") == REPROMPT else END


def decision_from(value: Any) -> dict[str, str]:
    """Turn a resume value into a state update.

    Anything that is not a reprompt carrying a non-empty prompt means continue,
    so a malformed resume value cannot strand the run.
    """
    if isinstance(value, Mapping) and value.get("action") == REPROMPT:
        prompt = str(value.get("prompt") or "").strip()
        if prompt:
            return {"decision": REPROMPT, "prompt": prompt, "route": "confirm"}
    return {"decision": CONTINUE, "route": "confirm"}


def _terminal(name: str) -> Callable[[AgentState], AgentState]:
    """Build a placeholder node that records which route was taken."""

    def node(state: AgentState) -> AgentState:
        return {"route": name}

    node.__name__ = f"{name}_node"
    return node


def build_graph(
    *,
    client: TypeSafeClient | None = None,
    model: str | None = None,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Compile the graph. ``client``/``model`` are captured by the Jev nodes."""
    if client is None:
        client = build_client(model)
    if checkpointer is None:
        checkpointer = InMemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES)
        )

    def classify_node(state: AgentState) -> AgentState:
        return {"classification": classify(state["prompt"], client=client, model=model)}

    def research_node(state: AgentState) -> AgentState:
        result = local_research(state["prompt"], client=client, model=model)
        return {
            "research": result,
            "relevant_files": result.files,
            "file_contents": result.contents,
            "route": "local_research",
        }

    def confirm_node(state: AgentState) -> AgentState:
        research = state["research"]
        value = interrupt(
            {
                "question": "Continue with these files?",
                "files": list(research.files),
                "relevance": {
                    path: round(score, 3)
                    for path, score in research.relevance.items()
                },
                "candidates": research.candidates,
                "listed": research.listed,
            }
        )
        return decision_from(value)

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("local_research", research_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("run_command", _terminal("run_command"))
    graph.add_node("clarify", _terminal("clarify"))
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        lambda state: route_for(state["classification"]),
        list(NODES),
    )
    graph.add_edge("local_research", "confirm")
    graph.add_conditional_edges(
        "confirm",
        route_after_confirm,
        {"classify": "classify", END: END},
    )
    graph.add_edge("run_command", END)
    graph.add_edge("clarify", END)
    return graph.compile(checkpointer=checkpointer)


def thread_config(thread_id: str | None = None) -> dict[str, Any]:
    """Config carrying the thread id that ties checkpoints to one run."""
    return {"configurable": {"thread_id": thread_id or uuid4().hex}}


def start(
    graph: CompiledStateGraph, prompt: str, *, thread_id: str | None = None
) -> AgentState:
    """Begin a run. Returns a finished state or one stopped at ``confirm``."""
    return graph.invoke({"prompt": prompt}, thread_config(thread_id))


def resume(
    graph: CompiledStateGraph, decision: Mapping[str, str], *, thread_id: str
) -> AgentState:
    """Continue a stopped run with a human decision."""
    return graph.invoke(Command(resume=decision), thread_config(thread_id))


def pending(state: Mapping[str, Any]) -> Interrupt | None:
    """The interrupt a run is stopped at, or ``None`` when it finished."""
    interrupts = state.get("__interrupt__")
    return interrupts[0] if interrupts else None


def run(
    prompt: str,
    *,
    client: TypeSafeClient | None = None,
    model: str | None = None,
    thread_id: str | None = None,
) -> AgentState:
    """Start one run and return it at its first stop.

    A run stopped at ``confirm`` is only resumable while the graph that produced
    it is alive, so an interactive caller should build the graph once and drive
    it with :func:`start` and :func:`resume`.
    """
    graph = build_graph(client=client, model=model)
    return start(graph, prompt, thread_id=thread_id)
