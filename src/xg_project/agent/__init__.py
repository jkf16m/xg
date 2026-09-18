"""xg_project.agent — the LangGraph agent.

The graph classifies an incoming prompt with Jev and routes it to a downstream
node. Downstream nodes are placeholders until their pipelines exist.

Public API
----------
    build_graph(*, client=None, model=None) -> CompiledStateGraph
    run(prompt, *, client=None, model=None) -> AgentState
    route_for(classification) -> str
    NODES -> tuple[str, ...]
    AgentState -> the graph state schema
"""

from xg_project.agent._graph import NODES, build_graph, route_for, run
from xg_project.agent._state import AgentState

__all__ = [
    "NODES",
    "AgentState",
    "build_graph",
    "route_for",
    "run",
]
