"""Can Jev sort data? A small, measurable test with a known ground truth.

Two independent methods are tried against the same two datasets:

1. ``choice``     — one Choice question over one option per record. Per
   https://docs.typesafe.ai/cookbooks/skill_suggestion the Choice probabilities
   are the ranking, so the records are ordered by descending probability.
2. ``pairwise``   — one Noul question per unordered pair ("does A come before
   B?"). The answers are used as a comparator and the records are sorted
   client-side with ``functools.cmp_to_key``. This is sorting as an algorithm:
   the model only ever answers a comparison.

Both are compared to the real sorted order (exact match + Kendall tau distance),
and the pairwise run also reports per-pair comparison accuracy and whether the
model's comparisons form a consistent total order (no cycles).

    uv run python scripts/jev_sort_test.py
"""

import argparse
import itertools
import time
from functools import cmp_to_key

from typesafe_sdk import Choice, Noul, TypeSafeClient

from xg_project.jev import build_client

# Scrambled on purpose: an input that already is the answer would let a model
# that echoes the state score well without sorting anything.
WORDS = ["fig", "banana", "kiwi", "apple", "elderberry", "cherry", "grape", "date"]
NUMBERS = [42, 7, 256, 3, 128, 19, 1, 64]


def ground_truth(values: list[str | int]) -> list:
    return sorted(values)


def kendall_tau_distance(order: list, truth: list) -> int:
    """Number of discordant pairs: how far ``order`` is from ``truth``."""
    rank = {item: i for i, item in enumerate(truth)}
    return sum(
        1
        for a, b in itertools.combinations(order, 2)
        if rank[a] > rank[b]
    )


def _records(values: list[str | int]) -> list[dict]:
    return [{"id": f"r{i}", "value": v} for i, v in enumerate(values)]


def choice_ranking(
    client: TypeSafeClient,
    values: list[str | int],
    *,
    direction: str,
    model: str | None,
) -> tuple[list, dict]:
    """Method 1: one Choice over all records; probabilities give the order."""
    records = _records(values)
    question = Choice(
        instructions=(
            f"In the state, `records` is a list and `sort_by` names the field to "
            f"order them by. Rank the records in {direction} order of `value`. "
            f"Put the most probability on the record that comes first."
        ),
        criteria={
            record["id"]: {"value": record["value"]} for record in records
        },
    )
    started = time.perf_counter()
    response = client.system_one(
        state={"sort_by": "value", "direction": direction, "records": records},
        questions={"ranking": question},
        model=model,
        timeout=60.0,
    )
    elapsed = time.perf_counter() - started

    answer = response.choices["ranking"]
    by_id = {record["id"]: record["value"] for record in records}
    order = [
        by_id[label]
        for label, _ in sorted(
            answer.probabilities.items(), key=lambda item: item[1], reverse=True
        )
    ]
    return order, {
        "elapsed_s": elapsed,
        "inputs": response.usage.input_tokens,
        "outputs": response.usage.output_tokens,
        "confidence": answer.confidence,
    }


def pairwise_ranking(
    client: TypeSafeClient,
    values: list[str | int],
    *,
    direction: str,
    model: str | None,
) -> tuple[list, dict]:
    """Method 2: one Noul per pair; sort client-side from the comparisons."""
    records = _records(values)
    pairs = list(itertools.combinations([record["id"] for record in records], 2))
    questions = {
        f"before::{a}::{b}": Noul(
            instructions=(
                f"In the state, `records` is a list and `sort_by` names the field "
                f"to order them by. When sorting in {direction} order of `value`, "
                f"does the record with id `{a}` come before the record with "
                f"id `{b}`?"
            ),
            criteria={
                "true": f"`{a}` before `{b}`.",
                "false": f"`{b}` before `{a}`.",
            },
        )
        for a, b in pairs
    }
    started = time.perf_counter()
    response = client.system_one(
        state={"sort_by": "value", "direction": direction, "records": records},
        questions=questions,
        model=model,
        timeout=60.0,
    )
    elapsed = time.perf_counter() - started

    # P(a before b) from the answer for (a, b); P(b before a) = 1 - that.
    prob: dict[tuple[str, str], float] = {}
    for a, b in pairs:
        value = response.nouls[f"before::{a}::{b}"].noul
        prob[(a, b)] = value
        prob[(b, a)] = 1.0 - value

    def compare(a: str, b: str) -> int:
        return -1 if prob[(a, b)] >= 0.5 else 1

    ids = [record["id"] for record in records]
    ordered_ids = sorted(ids, key=cmp_to_key(compare))
    by_id = {record["id"]: record["value"] for record in records}
    order = [by_id[record_id] for record_id in ordered_ids]

    truth = ground_truth(values)
    rank = {value: i for i, value in enumerate(truth)}
    agree = sum(
        1
        for a, b in pairs
        if (prob[(a, b)] >= 0.5) == (rank[by_id[a]] < rank[by_id[b]])
    )
    cycles = _has_cycle(
        ids, [(a, b) for a, b in pairs if prob[(a, b)] >= 0.5]
    )
    return order, {
        "elapsed_s": elapsed,
        "inputs": response.usage.input_tokens,
        "outputs": response.usage.output_tokens,
        "pair_accuracy": agree / len(pairs),
        "pairs": len(pairs),
        "consistent_total_order": not cycles,
    }


def _has_cycle(nodes: list[str], edges: list[tuple[str, str]]) -> bool:
    """Whether the 'before' edges contain a cycle (an inconsistent total order)."""
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for a, b in edges:
        adjacency[a].append(b)
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        if state.get(node, 0) == 1:
            return True
        if state.get(node, 0) == 2:
            return False
        state[node] = 1
        for neighbour in adjacency[node]:
            if visit(neighbour):
                return True
        state[node] = 2
        return False

    return any(visit(node) for node in nodes)


def report(label: str, run, truth: list) -> None:
    order, meta = run
    print(f"  {label:<9} order : {order}")
    print(
        f"  {'':<9} exact : {order == truth}   "
        f"kendall: {kendall_tau_distance(order, truth)} discordant pairs"
    )
    details = "  ".join(f"{k}={v}" for k, v in meta.items())
    print(f"  {'':<9} {details}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="jev-latest", help="Jev model (default: jev-latest)"
    )
    args = parser.parse_args()

    with build_client(args.model) as client:
        for name, values, direction in [
            ("words", WORDS, "alphabetical ascending"),
            ("numbers", NUMBERS, "numeric ascending"),
        ]:
            truth = ground_truth(values)
            print(f"\n{name}: {values}")
            print(f"  truth    : {truth}")
            report(
                "choice",
                choice_ranking(
                    client, values, direction=direction, model=args.model
                ),
                truth,
            )
            report(
                "pairwise",
                pairwise_ranking(
                    client, values, direction=direction, model=args.model
                ),
                truth,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
