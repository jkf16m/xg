"""The graph: what nodes exist, who may move between them, and what they read.

The registry graph lives here. It is the graph the TUI walks: nodes are
declarations, movement is a policy over levels, and every node introduces state
under its own key. :mod:`xg_project.graph._default` is the built-in graph;
extensions register against the same :class:`~xg_project.graph.Registry`.

Two modules in this package are a separate, worked example of the same ideas in
LangGraph rather than the registry, and are not what the TUI runs:

- ``xg_project.graph.langgraph_tree`` — the ``xg_``-prefixed decision tree, the
  ``TreeBuilder`` that stays open to extensions until ``seal()``, and the
  one-shot ``build_tree()``.
- ``xg_project.graph.walk`` — ``Walk``, the session that moves one node per
  prompt and returns to the parent on ``/back`` or ``/b``.
"""

from xg_project.graph._default import (
    ANSWER,
    COMMAND,
    EDIT,
    FILTER,
    ORIGIN,
    SELECT_MODULE,
    SORT,
    default_graph,
)
from xg_project.graph._registry import (
    XG_PREFIX,
    Actor,
    Handler,
    NodeDeclaration,
    NodeInput,
    NodeKind,
    Registry,
    UnknownNode,
    xg,
)

__all__ = [
    "ANSWER",
    "COMMAND",
    "EDIT",
    "FILTER",
    "ORIGIN",
    "SELECT_MODULE",
    "SORT",
    "XG_PREFIX",
    "Actor",
    "Handler",
    "NodeDeclaration",
    "NodeInput",
    "NodeKind",
    "Registry",
    "UnknownNode",
    "default_graph",
    "xg",
]
