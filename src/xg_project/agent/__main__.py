"""Run the xg agent graph from the command line — this is the ``xg`` command.

    xg "add retry to the http client"
    xg

The graph classifies the prompt with Jev and routes it to a downstream node.
Omit the prompt to type them one per line; the client and graph are reused.
"""

import argparse
import json
import sys
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph
from typesafe_sdk import TypeSafeClient, TypeSafeError

from xg_project.agent import AgentState, build_graph
from xg_project.config import resolve
from xg_project.jev import JevError, build_client
from xg_project.jev._render import render


def _model_from_config() -> str | None:
    return resolve(Path.cwd()).jev_model


def _emit(state: AgentState, *, as_json: bool) -> None:
    classification = state["classification"]
    route = state.get("route")
    research = state.get("research")
    files = state.get("relevant_files", [])
    if as_json:
        payload = dict(classification.to_dict())
        payload["route"] = route
        payload["relevant_files"] = files
        if research is not None:
            payload["candidate_count"] = research.candidates
            payload["relevance"] = {
                path: round(probability, 3)
                for path, probability in research.relevance.items()
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(render(classification))
    print(f"route       : {route}")
    if research is not None:
        print(f"files       : {len(files)} of {research.candidates} candidates")
        for path in files:
            print(f"  {research.relevance[path]:.2f}  {path}")


def _one(graph: CompiledStateGraph, prompt: str, *, as_json: bool) -> int:
    try:
        state = graph.invoke({"prompt": prompt})
    except (JevError, ValueError) as exc:
        print(f"classification failed: {exc}", file=sys.stderr)
        return 1
    _emit(state, as_json=as_json)
    return 0


def _loop(graph: CompiledStateGraph, *, as_json: bool) -> int:
    status = 0
    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return status
        if not prompt.strip():
            continue
        status = _one(graph, prompt, as_json=as_json)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="xg",
        description="Classify and route a prompt with TypeSafe Jev.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="the prompt to classify and route; omit to read prompts from stdin",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Jev model name (default: .xg config, else jev-latest)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the classification as JSON"
    )
    args = parser.parse_args()

    model = args.model or _model_from_config()
    try:
        client: TypeSafeClient = build_client(model)
    except TypeSafeError as exc:
        print(f"classification failed: {exc}", file=sys.stderr)
        return 1

    graph = build_graph(client=client, model=model)
    with client:
        if args.prompt is not None:
            return _one(graph, args.prompt, as_json=args.json)
        return _loop(graph, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
