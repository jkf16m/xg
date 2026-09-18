"""Run the xg agent graph from the command line — this is the ``xg`` command.

    xg "add retry to the http client"
    xg

The graph classifies the prompt with Jev, researches the repository, then stops
to ask for confirmation before the generative step. Answer with ``c`` to
continue, ``r`` to reprompt (or just type the new prompt), or ``q`` to quit.

Omit the prompt to type them one per line; the client and graph are reused.
With ``--json`` an interrupted run is printed with ``pending_confirmation`` set
and the command exits, so it stays scriptable.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph
from typesafe_sdk import TypeSafeClient, TypeSafeError

from xg_project.agent import (
    CONTINUE,
    REPROMPT,
    AgentState,
    build_graph,
    pending,
    resume,
    start,
)
from xg_project.config import resolve
from xg_project.jev import JevError, build_client
from xg_project.jev._render import render


def _model_from_config() -> str | None:
    return resolve(Path.cwd()).jev_model


def _emit(state: AgentState, *, as_json: bool, awaiting: bool = False) -> None:
    classification = state["classification"]
    route = state.get("route")
    decision = state.get("decision")
    research = state.get("research")
    files = state.get("relevant_files", [])
    show_files = route in {"local_research", "confirm"} and research is not None
    if as_json:
        payload: dict[str, Any] = dict(classification.to_dict())
        payload["route"] = route
        payload["pending_confirmation"] = awaiting
        if decision is not None:
            payload["decision"] = decision
        if show_files:
            payload["relevant_files"] = files
            payload["candidate_count"] = research.candidates
            payload["listed_count"] = research.listed
            payload["content_bytes"] = sum(
                len(text) for text in state.get("file_contents", {}).values()
            )
            payload["relevance"] = {
                path: round(probability, 3)
                for path, probability in research.relevance.items()
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(render(classification))
    print(f"route       : {route}")
    if decision is not None:
        print(f"decision    : {decision}")
    if show_files:
        print(
            f"files       : {len(files)} of {research.candidates} candidates "
            f"({research.listed} tracked)"
        )
        for path in files:
            print(f"  {research.relevance[path]:.2f}  {path}")
        carried = sum(len(text) for text in state.get("file_contents", {}).values())
        print(f"contents    : {carried} bytes carried in state")
    if awaiting:
        print("awaiting    : confirmation (c=continue, r=reprompt, q=quit)")


_YES = {"", "c", "continue", "y", "yes"}
_NO = {"q", "quit", "n", "no"}
_REPROMPT = {"r", "reprompt"}


def _ask() -> dict[str, str] | None:
    """Ask for a decision. Returns a resume value, or ``None`` to quit the run.

    Bare Enter continues. ``r`` asks for a new prompt; any other text is taken
    as that new prompt directly, so a reprompt is one keystroke away.
    """
    while True:
        try:
            reply = input("\n[c]ontinue  [r]eprompt  [q]uit > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        low = reply.lower()
        if low in _NO:
            return None
        if low in _YES:
            return {"action": CONTINUE}
        if low in _REPROMPT:
            try:
                reply = input("new prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return None
            if not reply:
                continue
        return {"action": REPROMPT, "prompt": reply}


def _one(graph: CompiledStateGraph, prompt: str, *, as_json: bool) -> int:
    thread_id = uuid4().hex
    try:
        state = start(graph, prompt, thread_id=thread_id)
    except (JevError, ValueError) as exc:
        print(f"classification failed: {exc}", file=sys.stderr)
        return 1

    while pending(state) is not None:
        _emit(state, as_json=as_json, awaiting=True)
        if as_json:
            return 0
        decision = _ask()
        if decision is None:
            return 0
        try:
            state = resume(graph, decision, thread_id=thread_id)
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
