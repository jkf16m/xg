"""Two-step file selection with Jev: Noul filtering, then relevance sorting.

The prompt is a realistic new-feature request against this repository. Every
tracked file is a candidate, sent with its full text, exactly as
:mod:`xg_project.research` does it.

Step 1 (filter)  — one ``Noul`` per file: is it needed at all? Files below 0.5
                   are not considered further.
Step 2 (sort)    — among the survivors, rank by relevance. Tried two ways:
                   one ``Score`` per file against a shared rubric, and one
                   ``Noul`` per pair ("is A more relevant than B?") feeding a
                   client-side comparison sort.

Both steps are graded against hand-labelled tiers for this feature:

    MUST    the feature cannot be implemented without reading/changing it
    SHOULD  must be read to implement it correctly
    REJECT  unrelated to the feature

Metrics: false rejects (MUST/SHOULD filtered out) and false accepts (REJECT
kept) for the filter; the ranked list and tier inversions for the sort.

    uv run python scripts/jev_file_relevance_test.py
"""

import argparse
import itertools
import time
from functools import cmp_to_key
from pathlib import Path

from typesafe_sdk import Noul, Score, TypeSafeClient

from xg_project.jev import build_client
from xg_project.research import MAX_FILE_BYTES, repo_files

ROOT = Path(__file__).resolve().parent.parent

PROMPT = (
    "I want to implement the `answer` path in xg. Today a request classified "
    "with operation `answer` still stops at `confirm` and then ends, so those "
    "requests have no path at all. I want them to run research, build the "
    "context window, ask the generative model for a plain-text reply, and print "
    "that reply to the terminal — no file is written."
)

MUST = [
    "src/xg_project/agent/_graph.py",      # add the answer node and the routes
    "src/xg_project/agent/__main__.py",    # CLI: print the reply, drive the flow
    "src/xg_project/llm/__init__.py",      # the generative reply call
    "src/xg_project/agent/_state.py",      # state field for the reply text
]

SHOULD = [
    "src/xg_project/agent/__init__.py",    # re-export the new node/helpers
    "src/xg_project/context/__init__.py",  # the answer path reuses the window
    "src/xg_project/jev/_classify.py",     # where Operation.ANSWER comes from
]

# Everything not in MUST/SHOULD and not listed here is treated as REJECT when
# counting false accepts; this list names the ones most likely to tempt a model.
CLEAR_REJECT = [
    "src/xg_project/session/__init__.py",
    "src/xg_project/interface.py",
    "src/xg_project/terminal.py",
    "src/xg_project/config/__init__.py",
    "src/xg_project/research/__init__.py",
    "src/xg_project/jev/_taxonomy.py",
    "src/xg_project/jev/_client.py",
    "src/xg_project/jev/_render.py",
    "src/xg_project/jev/__main__.py",
    "src/xg_project/jev/test_jev.py",
    "src/xg_project/jev/test_integration.py",
    "src/xg_project/jev/__init__.py",
    "src/xg_project/__init__.py",
    "build.py",
    "install.py",
    "pyproject.toml",
    "README.md",
    "uv.lock",
    "AGENTS.md",
    ".gitignore",
    ".python-version",
    ".xgignore",
]

TIER = {path: 0 for path in MUST}
TIER.update({path: 1 for path in SHOULD})
TIER.update({path: 2 for path in CLEAR_REJECT})


def tier(path: str) -> int:
    """The hand label for a path; anything unlisted counts as REJECT."""
    return TIER.get(path, 2)


def candidates() -> dict[str, str]:
    """Tracked files with readable, non-huge contents, as path -> text."""
    loaded: dict[str, str] = {}
    for relative in repo_files(ROOT):
        try:
            text = (ROOT / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if len(text.encode("utf-8")) > MAX_FILE_BYTES:
            continue
        loaded[relative] = text
    return loaded


def filter_step(
    client: TypeSafeClient, files: dict[str, str], model: str | None
) -> dict[str, float]:
    """Step 1: one Noul per file — is it needed at all?"""
    questions = {
        f"file_{index}": Noul(
            instructions=(
                f"Does implementing the request require reading or changing the "
                f"repository file {path!r}? Its full text is in the state under "
                f"`files[{path!r}]`. Answer yes only if the file must be read or "
                f"edited to implement the feature, not merely because it belongs "
                f"to the same package or shares vocabulary with the request."
            ),
            criteria={
                "true": "The file must be read or changed to implement the feature.",
                "false": "The file is not needed for this feature.",
            },
        )
        for index, path in enumerate(files)
    }
    started = time.perf_counter()
    response = client.system_one(
        state={"request": PROMPT, "files": files},
        questions=questions,
        model=model,
        timeout=120.0,
    )
    print(
        f"  filter call: {time.perf_counter() - started:.2f}s, "
        f"{response.usage.input_tokens} in / {response.usage.output_tokens} out"
    )
    return {
        path: response.nouls[f"file_{index}"].noul
        for index, path in enumerate(files)
    }


def score_sort(
    client: TypeSafeClient, files: dict[str, str], model: str | None
) -> list[tuple[str, float, float]]:
    """Step 2a: one Score per file on a shared relevance rubric."""
    rubric = [
        "Not relevant: nothing in the file is needed to implement the request.",
        "Background: the file explains a concept the feature relies on, but it "
        "does not need to change.",
        "Supporting: the file must be read so the feature is implemented "
        "correctly, and may need a small change.",
        "Core: the feature is implemented here, or the file must change for it.",
    ]
    questions = {
        f"rel_{index}": Score(
            instructions=(
                f"How relevant is the repository file {path!r} to implementing "
                f"the request? Its full text is in the state under `files[{path!r}]`."
            ),
            criteria=rubric,
        )
        for index, path in enumerate(files)
    }
    started = time.perf_counter()
    response = client.system_one(
        state={"request": PROMPT, "files": files},
        questions=questions,
        model=model,
        timeout=120.0,
    )
    print(
        f"  score call : {time.perf_counter() - started:.2f}s, "
        f"{response.usage.input_tokens} in / {response.usage.output_tokens} out"
    )
    rows = []
    for index, path in enumerate(files):
        answer = response.scores[f"rel_{index}"]
        rows.append((path, answer.score, answer.confidence))
    return sorted(rows, key=lambda row: (-row[1], -row[2], row[0]))


def pairwise_sort(
    client: TypeSafeClient, files: dict[str, str], model: str | None
) -> list[str]:
    """Step 2b: one Noul per pair, then a client-side comparison sort."""
    paths = list(files)
    pairs = list(itertools.combinations(paths, 2))
    questions = {
        f"more::{a}::{b}": Noul(
            instructions=(
                f"To implement the request, is the repository file {a!r} more "
                f"relevant than the repository file {b!r}? Both full texts are in "
                f"the state under `files`. Count a file as more relevant when it "
                f"must be read or changed sooner to implement the feature."
            ),
            criteria={
                "true": f"`{a}` is more relevant than `{b}`.",
                "false": f"`{b}` is more relevant than `{a}`.",
            },
        )
        for a, b in pairs
    }
    started = time.perf_counter()
    response = client.system_one(
        state={"request": PROMPT, "files": files},
        questions=questions,
        model=model,
        timeout=120.0,
    )
    print(
        f"  pair call  : {time.perf_counter() - started:.2f}s, "
        f"{response.usage.input_tokens} in / {response.usage.output_tokens} out "
        f"({len(pairs)} pairs)"
    )

    prob: dict[tuple[str, str], float] = {}
    for a, b in pairs:
        value = response.nouls[f"more::{a}::{b}"].noul
        prob[(a, b)] = value
        prob[(b, a)] = 1.0 - value

    def compare(a: str, b: str) -> int:
        return -1 if prob[(a, b)] >= 0.5 else 1

    return sorted(paths, key=cmp_to_key(compare))


def tier_inversions(order: list[str]) -> int:
    """Pairs the model ordered against the tiers (lower tier must come first).

    ``combinations`` yields earlier-before-later pairs, so a pair is an
    inversion exactly when the earlier item has the higher (worse) tier.
    """
    return sum(
        1 for a, b in itertools.combinations(order, 2) if tier(a) > tier(b)
    )


def tier_pairs(order: list[str]) -> int:
    return sum(
        1 for a, b in itertools.combinations(order, 2) if tier(a) != tier(b)
    )


def show(order: list[str], detail: dict[str, tuple[float, float]]) -> None:
    labels = {0: "MUST  ", 1: "SHOULD", 2: "reject"}
    for index, path in enumerate(order, 1):
        value, confidence = detail[path]
        print(
            f"  {index:>2}. [{labels[tier(path)]}] {value:>4.2f} "
            f"(conf {confidence:.2f})  {path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="jev-latest", help="Jev model (default: jev-latest)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Noul filter threshold"
    )
    args = parser.parse_args()

    files = candidates()
    total = sum(len(text) for text in files.values())
    print(f"prompt: {PROMPT}\n")
    print(f"candidates: {len(files)} files, {total} bytes")

    with build_client(args.model) as client:
        relevance = filter_step(client, files, args.model)

        print("\nfilter relevance (all candidates, descending):")
        for path, probability in sorted(
            relevance.items(), key=lambda item: (-item[1], item[0])
        ):
            labels = {0: "MUST  ", 1: "SHOULD", 2: "reject"}
            print(f"  {probability:.2f}  [{labels[tier(path)]}]  {path}")

        selected = [
            path
            for path in files
            if relevance[path] >= args.threshold
        ]
        selected.sort(key=lambda path: (-relevance[path], path))
        selected_files = {path: files[path] for path in selected}

        rejects = [
            path for path in files if relevance[path] < args.threshold
        ]
        false_rejects = sorted(p for p in rejects if tier(p) <= 1)
        false_accepts = sorted(
            p for p in selected if p in CLEAR_REJECT
        )
        print(
            f"\nfilter: {len(selected)} selected, {len(rejects)} rejected "
            f"(threshold {args.threshold})"
        )
        print(f"  false rejects (MUST/SHOULD dropped): {false_rejects}")
        print(f"  false accepts (clear REJECT kept)  : {false_accepts}")

        print("\nsort A — Score per file (rubric 0..3):")
        scored = score_sort(client, selected_files, args.model)
        show([row[0] for row in scored], {row[0]: (row[1], row[2]) for row in scored})
        order_a = [row[0] for row in scored]

        print("\nsort B — pairwise Noul comparisons:")
        order_b = pairwise_sort(client, selected_files, args.model)
        show(order_b, {path: (relevance[path], 0.0) for path in order_b})

    for label, order in (("A (Score)", order_a), ("B (pairwise)", order_b)):
        top4 = [path for path in order[:4]]
        print(
            f"\n{label}: MUST in top 4 = "
            f"{len(set(MUST) & set(top4))}/4; "
            f"tier inversions = {tier_inversions(order)}/{tier_pairs(order)} "
            f"comparable pairs"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
