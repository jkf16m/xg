"""Try the Jev pipeline planner end to end: research the repo, then plan it.

Runs the real two-stage pipeline against this repository:

1. ``local_research`` -- one ``Noul`` per candidate file ("is this file
   needed?"), selecting the files whose probability clears 0.5.
2. ``planner.plan``  -- the round loop. Each round sends the plan built so far
   and asks ``done`` (``Noul``) and ``next`` (one composite ``Choice`` whose
   options are ``edit:<path>``, ``read:<path>``, ``write:new_file``, ``answer``).

Prints the selected files, then the plan, round by round, so the loop's
decisions are visible rather than just the final list.

    uv run python scripts/jev_planner_test.py
    uv run python scripts/jev_planner_test.py "your own request"
"""

import argparse
import time
from pathlib import Path

from xg_project.config import resolve
from xg_project.jev import build_client
from xg_project.planner import MAX_STEPS, StepKind, build_questions, plan
from xg_project.research import local_research, repo_files

ROOT = Path(__file__).resolve().parent.parent

PROMPT = (
    "Explain how a request flows through the xg agent graph: what each node "
    "does, and where a Jev decision happens. I want the explanation to trace "
    "the actual code in this repository."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default=PROMPT)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    model = resolve(ROOT).jev_model
    print(f"model      : {model or 'jev-latest (SDK default)'}")
    print(f"listed     : {len(repo_files(ROOT))} files")
    print(f"request    : {args.prompt}\n")

    with build_client(model) as client:
        started = time.perf_counter()
        research = local_research(
            args.prompt, client=client, model=model, root=ROOT
        )
        elapsed = time.perf_counter() - started
        print(
            f"research   : {research.candidates} candidates -> "
            f"{len(research.files)} selected in {elapsed:.1f}s"
        )
        for path in research.files:
            print(f"             {research.relevance[path]:.3f}  {path}")
        print()

        started = time.perf_counter()

        # The loop is transparent: wrap the client so each round prints the
        # questions asked and the answers returned before plan() consumes them.
        # The original method is bound first; calling ``client.system_one`` from
        # inside the wrapper would otherwise recurse.
        original_system_one = client.system_one

        def traced_system_one(*, state, questions, model=None):
            print(f"round {len(traces) + 1}")
            print(f"  plan in state : {state['plan']}")
            response = original_system_one(state=state, questions=questions, model=model)
            done = response.nouls["done"].noul
            nxt = response.choices["next"]
            print(f"  done          : {done:.3f}")
            print(f"  next          : {nxt.choice} ({nxt.confidence:.3f})")
            traces.append(response)
            return response

        traces = []
        client.system_one = traced_system_one  # type: ignore[method-assign]
        result = plan(
            args.prompt,
            files=research.contents,
            client=client,
            model=model,
            max_steps=args.max_steps,
        )
        elapsed = time.perf_counter() - started

    print(f"\nplan       : {result.rounds} rounds in {elapsed:.1f}s")
    print(f"stopped    : {result.stopped}")
    if result.is_empty:
        print("             (no steps -- the request needs nothing done)")
    for index, step in enumerate(result.steps):
        if step.new_file:
            where = "a new file (path chosen at generate time)"
        elif step.kind is StepKind.ANSWER:
            where = "no file"
        else:
            where = step.target
        print(f"  {index}. {step.kind.value:<7} {where}")

    # The questions are stable across rounds; print one set so the option
    # space is visible. This rebuilds it rather than replaying a captured call.
    questions = build_questions(research.contents)
    options = sorted(questions["next"].criteria)
    print(f"\nnext options: {len(options)} -> {', '.join(options)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
