"""Run the xg agent graph from the command line — this is the ``xg`` command.

    xg "add a module that parses xgignore files"
    xg

The graph classifies the prompt with Jev, researches the repository, then stops
to ask for confirmation before the write path runs. Answer with ``c`` to
continue, ``r`` to reprompt (or just type the new prompt), or ``q`` to quit.

When the request cannot be placed, or the path the model proposes is already
taken, the run stops and asks for a clearer request instead; there, any text is
the new request.

Omit the prompt to type them one per line; the client and graph are reused.
With ``--json`` a stopped run is printed with ``awaiting`` set and the command
exits, so it stays scriptable.
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
    CLARIFICATION,
    CONTINUE,
    REPROMPT,
    AgentState,
    awaiting,
    build_graph,
    pending,
    resume,
    start,
)
from xg_project.config import resolve
from xg_project.jev import JevError, build_client
from xg_project.jev._render import render
from xg_project.llm import LlmError

FILE_ROUTES = {"local_research", "confirm", "write", "check", "context", "generate"}
"""Routes whose state carries the research result, so the files are worth showing."""


def _config_model(key: str) -> str | None:
    value = getattr(resolve(Path.cwd()), key)
    return value if isinstance(value, str) else None


def _emit(state: AgentState, *, as_json: bool, asking: str | None = None) -> None:
    classification = state["classification"]
    route = state.get("route")
    decision = state.get("decision")
    reason = state.get("reason")
    research = state.get("research")
    files = state.get("relevant_files") or []
    show_files = route in FILE_ROUTES and research is not None
    if as_json:
        payload: dict[str, Any] = dict(classification.to_dict())
        payload["route"] = route
        payload["awaiting"] = asking
        payload["reason"] = reason
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
        for key in (
            "proposed_path",
            "target_path",
            "context_path",
            "context_files",
            "context_bytes",
            "written_path",
            "written_bytes",
        ):
            if state.get(key) is not None:
                payload[key] = state[key]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(render(classification))
    print(f"route       : {route}")
    if decision is not None:
        print(f"decision    : {decision}")
    if reason:
        print(f"reason      : {reason}")
    if show_files:
        print(
            f"files       : {len(files)} of {research.candidates} candidates "
            f"({research.listed} tracked)"
        )
        for path in files:
            print(f"  {research.relevance[path]:.2f}  {path}")
    if state.get("target_path"):
        print(f"target      : {state['target_path']}")
    if state.get("context_path"):
        print(
            f"context     : {state['context_files']} files, "
            f"{state['context_bytes']} bytes -> {state['context_path']}"
        )
    if state.get("written_path"):
        print(f"written     : {state['written_path']} ({state['written_bytes']} bytes)")
    if asking == CLARIFICATION:
        print("awaiting    : a clearer request (type it, or q to quit)")
    elif asking is not None:
        print("awaiting    : confirmation (c=continue, r=reprompt, q=quit)")


_YES = {"c", "continue", "y", "yes"}
_NO = {"q", "quit", "n", "no"}
_REPROMPT = {"r", "reprompt"}


def _prompt_for(asking: str | None) -> dict[str, str] | None:
    """Ask the user. Returns a resume value, or ``None`` to stop the run."""
    if asking == CLARIFICATION:
        try:
            reply = input("\nnew request (q to quit) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not reply or reply.lower() in _NO:
            return None
        return {"action": REPROMPT, "prompt": reply}

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
    except (JevError, ValueError, LlmError) as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1

    while pending(state) is not None:
        asking = awaiting(state)
        _emit(state, as_json=as_json, asking=asking)
        if as_json:
            return 0
        decision = _prompt_for(asking)
        if decision is None:
            return 0
        try:
            state = resume(graph, decision, thread_id=thread_id)
        except (JevError, ValueError, LlmError) as exc:
            print(f"failed: {exc}", file=sys.stderr)
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
        "--llm-model",
        default=None,
        help="generative model name (default: .xg config, else @preset/mimo)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the classification as JSON"
    )
    args = parser.parse_args()

    model = args.model or _config_model("jev_model")
    try:
        client: TypeSafeClient = build_client(model)
    except TypeSafeError as exc:
        print(f"classification failed: {exc}", file=sys.stderr)
        return 1

    graph = build_graph(
        client=client,
        model=model,
        llm_model=args.llm_model or _config_model("model"),
    )
    with client:
        if args.prompt is not None:
            return _one(graph, args.prompt, as_json=args.json)
        return _loop(graph, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
