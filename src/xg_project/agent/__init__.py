"""xg_project.agent — the LangGraph agent.

The graph classifies an incoming prompt with Jev, researches the repository for
the files the request needs, asks the user to confirm, and then takes one path:
the write path, which creates a new file. No tools are bound and the graph uses
none; a language model is reached only at the last moment, to fill in generative
information.

Public API
----------
    build_graph(*, client=None, model=None, llm=None, llm_model=None, root=None,
                checkpointer=None) -> CompiledStateGraph
    start(graph, prompt, *, thread_id=None) -> AgentState
    resume(graph, decision, *, thread_id) -> AgentState
    pending(state) -> Interrupt | None
    awaiting(state) -> str | None
    run(prompt, *, client=None, model=None) -> AgentState
    route_for(classification) -> str
    route_after_confirm(state) -> str
    route_after_check(state) -> str
    route_after_reroute(state) -> str
    decision_from(value) -> dict[str, str]
    thread_config(thread_id=None) -> dict
    NODES -> tuple[str, ...]
    AgentState -> the graph state schema
"""

from xg_project.agent._graph import (
    ALLOWED_STATE_TYPES,
    CLARIFICATION,
    CONFIRMATION,
    CONTINUE,
    NODES,
    REPROMPT,
    awaiting,
    build_graph,
    decision_from,
    pending,
    resume,
    route_after_check,
    route_after_confirm,
    route_after_reroute,
    route_for,
    run,
    start,
    thread_config,
)
from xg_project.agent._state import AgentState

__all__ = [
    "ALLOWED_STATE_TYPES",
    "CLARIFICATION",
    "CONFIRMATION",
    "CONTINUE",
    "NODES",
    "REPROMPT",
    "AgentState",
    "awaiting",
    "build_graph",
    "decision_from",
    "pending",
    "resume",
    "route_after_check",
    "route_after_confirm",
    "route_after_reroute",
    "route_for",
    "run",
    "start",
    "thread_config",
]
