"""Classify prompts with TypeSafe Jev — the first filter.

One prompt in, one classification out:

    uv run python -m xg_project.jev "add retry to the http client"

Omit the prompt to type them one per line, with the client reused:

    uv run python -m xg_project.jev

The API key comes from ``TYPESAFE_API_KEY`` or ``pass show jev``. The model
comes from ``--model`` or the composed ``.xg/config.json`` (``jev_model``),
falling back to the SDK default (``jev-latest``).
"""

import argparse
import json
import sys
from pathlib import Path

from typesafe_sdk import TypeSafeClient, TypeSafeError

from xg_project.config import resolve
from xg_project.jev import Classification, JevError, build_client, classify


def _model_from_config() -> str | None:
    return resolve(Path.cwd()).jev_model


def _render(result: Classification, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return
    top = result.complexity.levels - 1
    print(
        f"kind        : {result.request_kind.label}  "
        f"({result.request_kind.confidence:.2f}, {result.band.value})"
    )
    print(f"scope       : {result.scope.label}  ({result.scope.confidence:.2f})")
    print(
        f"complexity  : {result.complexity.score:.2f}/{top}  "
        f"({result.complexity.normalized:.2f})"
    )
    print(f"changes code: {result.changes_code.probability:.2f}")
    print(f"destructive : {result.is_destructive.probability:.2f}")
    if result.needs_clarification:
        print("-> low confidence: ask the user to clarify before routing.")
    elif result.is_change:
        print(f"-> route to file selection ({result.scope.label}).")
    else:
        print("-> no repository change requested.")


def _one(prompt: str, client: TypeSafeClient, model: str | None, as_json: bool) -> int:
    try:
        _render(classify(prompt, client=client, model=model), as_json=as_json)
    except (JevError, ValueError) as exc:
        print(f"classification failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _loop(client: TypeSafeClient, model: str | None, as_json: bool) -> int:
    status = 0
    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return status
        if not prompt.strip():
            continue
        status = _one(prompt, client, model, as_json)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m xg_project.jev",
        description="Classify a prompt with TypeSafe Jev.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="the prompt to classify; omit to read prompts from stdin",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Jev model name (default: .xg config, else jev-latest)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full classification as JSON"
    )
    args = parser.parse_args()

    model = args.model or _model_from_config()
    try:
        client = build_client(model)
    except TypeSafeError as exc:
        print(f"classification failed: {exc}", file=sys.stderr)
        return 1

    with client:
        if args.prompt is not None:
            return _one(args.prompt, client, model, args.json)
        return _loop(client, model, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
