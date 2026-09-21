"""The graph: node declarations, the registry, and movement policy.

``default_graph()`` is what xg always has. Location-based discovery will add to
it later; the registry is already built for that, which is why ``add`` refuses
to clobber a name unless told to and every declaration carries ``origin``.

Two LangGraph modules sit beside it and are imported by path rather than
re-exported here, so that ``python -m xg_project.graph.<module>`` runs without
runpy importing the package's own submodule twice:

- ``xg_project.graph.langgraph_tree`` — the ``xg_``-prefixed decision tree, the
  ``TreeBuilder`` that stays open to extensions until ``seal()``, and the
  one-shot ``build_tree()``.
- ``xg_project.graph.walk`` — ``Walk``, the session that moves one node per
  prompt and returns to the parent on ``/back`` or ``/b``.
"""

from xg_project.graph._default import COMMAND, ORIGIN, default_graph
from xg_project.graph._registry import (
    Actor,
    NodeDeclaration,
    NodeKind,
    Registry,
    UnknownNode,
)

__all__ = [
    "COMMAND",
    "ORIGIN",
    "Actor",
    "NodeDeclaration",
    "NodeKind",
    "Registry",
    "UnknownNode",
    "default_graph",
]
