"""Classify one request from the command line.

    uv run python -m xg_project.jev "add retry to the http client"

Reads ``jev_model`` from the composed ``.xg/config.json`` settings, falling back
to the SDK default (``jev-latest``). Requires ``TYPESAFE_API_KEY``.
"""

import argparse
import json
import sys
from pathlib import Path

from xg_project.config import resolve
from xg_project.jev import Classification, JevError, classify


def _model_from_config() -> str | None:
    return resolve(Path.cwd()).jev_model


def _report(result: Classification) -> None:
    top = result.complexity.levels - 1
    print(
        f"request kind : {result.request_kind.label}  "
        f"(confidence {result.request_kind.confidence:.2f}, {result.band.value})"
    )
    print(
        f"scope        : {result.scope.label}  "
        f"(confidence {result.scope.confidence:.2f})"
    )
    print(
        f"complexity   : {result.complexity.score:.2f}/{top}  "
        f"({result.complexity.normalized:.2f} normalized, "
        f"confidence {result.complexity.confidence:.2f})"
    )
    print(f"changes code : {result.changes_code.probability:.2f}")
    print(f"destructive  : {result.is_destructive.probability:.2f}")
    if result.needs_clarification:
        print("-> low confidence: ask the user to clarify before routing.")
    elif result.is_change:
        print(f"-> route to file selection ({result.scope.label}).")
    else:
        print("-> no repository change requested.")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m xg_project.jev",
        description="Classify a request with TypeSafe Jev.",
    )
    parser.add_argument("request", help="the user request to classify")
    parser.add_argument(
        "--model",
        default=None,
        help="Jev model name (default: .xg config, else the SDK default jev-latest)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full classification as JSON"
    )
    args = parser.parse_args()

    try:
        result = classify(args.request, model=args.model or _model_from_config())
    except (JevError, ValueError) as exc:
        print(f"classification failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
