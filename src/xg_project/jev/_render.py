"""Render a Classification for humans or as JSON."""

import json

from xg_project.jev._classify import Classification


def render(result: Classification, *, as_json: bool = False) -> str:
    """Format a classification as text, or as a JSON object."""
    if as_json:
        return json.dumps(result.to_dict(), indent=2, sort_keys=True)

    top = result.complexity.levels - 1
    lines = [
        f"kind        : {result.request_kind.label}  "
        f"({result.request_kind.confidence:.2f}, {result.band.value})",
        f"operation   : {result.operation.label}  "
        f"({result.operation.confidence:.2f})",
        f"complexity  : {result.complexity.score:.2f}/{top}  "
        f"({result.complexity.normalized:.2f})",
        f"changes code: {result.changes_code.probability:.2f}",
        f"destructive : {result.is_destructive.probability:.2f}",
    ]
    if result.needs_reroute:
        lines.append("reroute: ask the user for a clearer request before routing.")
    return "\n".join(lines)
