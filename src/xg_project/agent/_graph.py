"""The xg agent graph: classify, research, confirm, and write.

LangGraph owns the routing; Jev owns the decisions. The graph uses no tools: a
turn advances through Jev decisions -- ``classify`` and the per-file questions
inside ``local_research`` -- and reaches a language model only at the last
moment, to fill in generative information.

    START -> classify -> { local_research, reroute }
    local_research -> confirm -> { classify, write, END }
    write -> check -> { reroute, context }
    context -> generate -> END
    reroute -> { classify, END }

Two nodes ask the user, both as LangGraph interrupts:

``confirm``
    Shown after research. The user continues to the write path or reprompts,
    which re-enters at ``classify`` because a different request can route
    differently.

``reroute``
    Shown when the graph will not guess: ``classify`` could not place the
    request, the model produced no usable path, or the target path already
    exists. The user supplies a clearer request and the run continues from
    ``classify``.

The write path keeps the model out of the collision decision. The model proposes
a path (``write``); plain filesystem code checks it (``check``); only when the
path is free does the graph build a deterministic context window (``context``)
and ask for the file's contents (``generate``). The model never decides whether
a file exists, and it is never handed a path that is already taken.

Interrupts need a checkpointer, so ``build_graph`` compiles with an
``InMemorySaver`` whose serializer is given an explicit allowlist for the
dataclasses the nodes put in state. Without the allowlist the checkpointer logs
an "unregistered type" warning on every deserialize.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, interrupt
from typesafe_sdk import TypeSafeClient

from xg_project.agent._state import AgentState
from xg_project.context import read_entries, render, write_window
from xg_project.jev import (
    Classification,
    Operation,
    build_client,
    classify,
)
from xg_project.llm import (
    build_llm,
    generate_file,
    parse_path,
    propose_path,
    safe_relative,
    strip_fence,
)
from xg_project.research import local_research, repo_files

NODES: tuple[str, ...] = ("local_research", "reroute")
"""Nodes the classify node can route to."""

CONTINUE = "continue"
REPROMPT = "reprompt"

CONFIRMATION = "confirmation"
CLARIFICATION = "clarification"
"""Which question an interrupt is asking, so a client can word its prompt."""

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
    graph or an API call. A request the graph cannot place is sent back to the
    user rather than researched, because research on a request nobody
    understood finds the wrong files.
    """
    if classification.needs_reroute:
        return "reroute"
    return "local_research"


def route_after_confirm(state: Mapping[str, Any]) -> str:
    """Send a reprompt back to classification, a write on to the write path."""
    if state.get("decision") == REPROMPT:
        return "classify"
    classification = state.get("classification")
    if classification is not None and classification.operation_kind is Operation.WRITE:
        return "write"
    return END


def route_after_check(state: Mapping[str, Any]) -> str:
    """Reroute when the check recorded a reason, otherwise build context."""
    return "reroute" if state.get("reason") else "context"


def route_after_reroute(state: Mapping[str, Any]) -> str:
    """Re-classify the user's new request, or stop if they declined."""
    return "classify" if state.get("decision") == REPROMPT else END


def decision_from(value: Any) -> dict[str, str]:
    """Turn a resume value into a state update.

    Anything that is not a reprompt carrying a non-empty prompt means continue,
    so a malformed resume value cannot strand the run.
    """
    if isinstance(value, Mapping) and value.get("action") == REPROMPT:
        prompt = str(value.get("prompt") or "").strip()
        if prompt:
            return {"decision": REPROMPT, "prompt": prompt, "route": "input"}
    return {"decision": CONTINUE, "route": "input"}


def build_graph(
    *,
    client: TypeSafeClient | None = None,
    model: str | None = None,
    llm: ChatOpenRouter | None = None,
    llm_model: str | None = None,
    root: Path | None = None,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Compile the graph.

    ``client``/``model`` are captured by the Jev nodes. ``llm``/``llm_model``
    are captured by the generative nodes and built on first use, so a run that
    never reaches them needs no OpenRouter key.
    """
    if client is None:
        client = build_client(model)
    if checkpointer is None:
        checkpointer = InMemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES)
        )
    workspace = (root or Path.cwd()).resolve()
    cache: dict[str, ChatOpenRouter] = {}

    def chat() -> ChatOpenRouter:
        if llm is not None:
            return llm
        if "llm" not in cache:
            cache["llm"] = build_llm(llm_model, root=workspace)
        return cache["llm"]

    def classify_node(state: AgentState) -> AgentState:
        result = classify(state["prompt"], client=client, model=model)
        route = route_for(result)
        if route == "reroute":
            reason = (
                f"the request could not be placed "
                f"(kind={result.request_kind.label}, "
                f"confidence={result.request_kind.confidence:.2f})"
            )
        else:
            reason = ""
        return {"classification": result, "reason": reason, "route": route}

    def research_node(state: AgentState) -> AgentState:
        result = local_research(
            state["prompt"], client=client, model=model, root=workspace
        )
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
                "awaiting": CONFIRMATION,
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
        decision = decision_from(value)
        if decision["decision"] == CONTINUE:
            decision["route"] = "confirm"
        return decision

    def reroute_node(state: AgentState) -> AgentState:
        value = interrupt(
            {
                "awaiting": CLARIFICATION,
                "question": "Give a clearer, more explicit request.",
                "request": state["prompt"],
                "reason": state.get("reason", ""),
            }
        )
        decision = decision_from(value)
        if decision["decision"] == CONTINUE:
            decision["route"] = "reroute"
        return decision

    def write_node(state: AgentState) -> AgentState:
        raw = propose_path(
            state["prompt"], existing=repo_files(workspace), llm=chat()
        )
        proposed = parse_path(raw)
        target = safe_relative(proposed, workspace)
        return {
            "proposed_path": proposed,
            "target_path": target or "",
            "route": "write",
        }

    def check_node(state: AgentState) -> AgentState:
        target = state.get("target_path") or ""
        if not target:
            reason = (
                f"the model returned no usable path "
                f"(reply: {state.get('proposed_path') or 'empty'!r})"
            )
        elif (workspace / target).exists():
            reason = f"{target} already exists"
        else:
            reason = ""
        return {"reason": reason, "route": "check"}

    def context_node(state: AgentState) -> AgentState:
        entries = read_entries(
            list(state.get("relevant_files") or []), root=workspace
        )
        text = render(entries, request=state["prompt"])
        window = write_window(text)
        return {
            "context_path": str(window),
            "context_files": len(entries),
            "context_bytes": len(text),
            "route": "context",
        }

    def generate_node(state: AgentState) -> AgentState:
        target = state["target_path"]
        window = Path(state["context_path"]).read_text(encoding="utf-8")
        content = strip_fence(
            generate_file(state["prompt"], target, window, llm=chat())
        )
        destination = workspace / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return {
            "written_path": target,
            "written_bytes": len(content),
            "route": "generate",
        }

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("local_research", research_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("reroute", reroute_node)
    graph.add_node("write", write_node)
    graph.add_node("check", check_node)
    graph.add_node("context", context_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify", lambda state: route_for(state["classification"]), list(NODES)
    )
    graph.add_edge("local_research", "confirm")
    graph.add_conditional_edges(
        "confirm",
        route_after_confirm,
        {"classify": "classify", "write": "write", END: END},
    )
    graph.add_edge("write", "check")
    graph.add_conditional_edges(
        "check", route_after_check, {"reroute": "reroute", "context": "context"}
    )
    graph.add_edge("context", "generate")
    graph.add_edge("generate", END)
    graph.add_conditional_edges(
        "reroute", route_after_reroute, {"classify": "classify", END: END}
    )
    return graph.compile(checkpointer=checkpointer)


def thread_config(thread_id: str | None = None) -> dict[str, Any]:
    """Config carrying the thread id that ties checkpoints to one run."""
    return {"configurable": {"thread_id": thread_id or uuid4().hex}}


def start(
    graph: CompiledStateGraph, prompt: str, *, thread_id: str | None = None
) -> AgentState:
    """Begin a run. Returns a finished state or one stopped at an interrupt."""
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


def awaiting(state: Mapping[str, Any]) -> str | None:
    """Which question the run is stopped on, or ``None`` when it finished."""
    stop = pending(state)
    if stop is None or not isinstance(stop.value, Mapping):
        return None
    value = stop.value.get("awaiting")
    return str(value) if value is not None else None


def run(
    prompt: str,
    *,
    client: TypeSafeClient | None = None,
    model: str | None = None,
    thread_id: str | None = None,
) -> AgentState:
    """Start one run and return it at its first stop.

    A stopped run is only resumable while the graph that produced it is alive,
    so an interactive caller should build the graph once and drive it with
    :func:`start` and :func:`resume`.
    """
    graph = build_graph(client=client, model=model)
    return start(graph, prompt, thread_id=thread_id)
