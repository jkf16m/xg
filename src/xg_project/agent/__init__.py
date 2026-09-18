"""xg_project.agent — the LangGraph agent.

The graph classifies an incoming prompt with Jev, researches the repository for
the files the request needs, and stops to ask a human to confirm before the
final generative step. Every node before that step is a Jev decision; the graph
uses no tools.

Public API
----------
    build_graph(*, client=None, model=None, checkpointer=None) -> CompiledStateGraph
    start(graph, prompt, *, thread_id=None) -> AgentState
    resume(graph, decision, *, thread_id) -> AgentState
    pending(state) -> Interrupt | None
    run(prompt, *, client=None, model=None) -> AgentState
    route_for(classification) -> str
    route_after_confirm(state) -> str
    decision_from(value) -> dict[str, str]
    thread_config(thread_id=None) -> dict
    NODES -> tuple[str, ...]
    AgentState -> the graph state schema
"""

from xg_project.agent._graph import (
    ALLOWED_STATE_TYPES,
    CONTINUE,
    NODES,
    REPROMPT,
    build_graph,
    decision_from,
    pending,
    resume,
    route_after_confirm,
    route_for,
    run,
    start,
    thread_config,
)
from xg_project.agent._state import AgentState

__all__ = [
    "ALLOWED_STATE_TYPES",
    "CONTINUE",
    "NODES",
    "REPROMPT",
    "AgentState",
    "build_graph",
    "decision_from",
    "pending",
    "resume",
    "route_after_confirm",
    "route_for",
    "run",
    "start",
    "thread_config",
]
