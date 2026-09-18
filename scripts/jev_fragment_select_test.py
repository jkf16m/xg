"""Selecting a contiguous text fragment with Jev — four methods, one document.

Document: ``src/xg_project/agent/_graph.py`` (358 lines), tagged line by line
with ids (``L014|``) kept in the state as ordinary text, the scheme the
semantic-search cookbook uses.

A ``Choice`` accepts at most 255 options (docs.typesafe.ai/primitives/choice),
so a 358-line document cannot be pointed at in one Choice question; the
coarse-to-fine methods chunk it first.

Methods, all against the same gold spans (exact, from ``ast``; each query is
described by what the target code does, never by its name):

``boundaries`` (2 requests)
    Chunk ``Choice``, then two ``Choice`` questions for the first and last
    line of the target inside that chunk.

``anchor`` (same 2 requests, different answer read)
    Read the third ``Choice`` answer instead: the single most representative
    line. Code expands that anchor to its enclosing definition with ``ast``.
    The model points; code decides the boundaries.

``line-noul`` (1 request, one question per line)
    One ``Noul`` per line; the longest run above threshold is the fragment.

``hunk-noul`` (1 request, one question per blank-line-separated hunk)
    Same at hunk granularity.

    uv run python scripts/jev_fragment_select_test.py
"""

import argparse
import ast
from pathlib import Path

from typesafe_sdk import Choice, Noul, TypeSafeClient

from xg_project.jev import build_client

ROOT = Path(__file__).resolve().parent.parent
DOCUMENT = ROOT / "src/xg_project/agent/_graph.py"
CHUNK_LINES = 255

# description -> function whose ast span is the gold fragment. The description
# deliberately never names the function or any identifier in it.
QUERIES: dict[str, str] = {
    "route_for": "chooses the next node from a request's classification",
    "route_after_confirm": (
        "decides where a run goes after the user answers the confirmation prompt"
    ),
    "check_node": (
        "checks whether the path the model proposed is already taken by an "
        "existing file"
    ),
    "generate_node": "writes the generated file contents to disk",
}


def definitions() -> list[tuple[str, int, int]]:
    """Every function definition as (name, first, last), innermost included."""
    found: list[tuple[str, int, int]] = []
    for node in ast.walk(ast.parse(DOCUMENT.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append((node.name, node.lineno, node.end_lineno))
    return found


def gold_spans() -> dict[str, tuple[int, int]]:
    return {name: (first, last) for name, first, last in definitions()}


def enclosing(anchor: int, spans: list[tuple[str, int, int]]) -> tuple[int, int]:
    """The smallest definition span containing ``anchor``, else the line itself."""
    containing = [
        (first, last)
        for _, first, last in spans
        if first <= anchor <= last
    ]
    if not containing:
        return (anchor, anchor)
    return min(containing, key=lambda span: span[1] - span[0])


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def block_span(lines: list[str], anchor: int) -> tuple[int, int]:
    """Enclosing block of a line by indentation alone — no parser.

    Scan up to the nearest less-indented non-blank line (the header), then down
    to the last more-indented non-blank line. Purely lexical: works for Python,
    and for anything else indentation delimits.
    """

    def indent(number: int) -> int:
        return indent_of(lines[number - 1])

    depth = indent(anchor)
    header = anchor
    probe = anchor - 1
    while probe >= 1:
        if lines[probe - 1].strip() and indent(probe) < depth:
            header = probe
            break
        probe -= 1

    top = indent(header)
    end = header
    probe = header + 1
    while probe <= len(lines):
        if lines[probe - 1].strip():
            if indent(probe) <= top:
                break
            end = probe
        probe += 1
    return (header, end)


def tag(lines: list[str]) -> tuple[list[str], str]:
    ids = [f"L{index:03d}" for index in range(1, len(lines) + 1)]
    body = "\n".join(
        f"{line_id}|{line}" for line_id, line in zip(ids, lines, strict=True)
    )
    return ids, body


def line_of(line_id: str) -> int:
    return int(line_id[1:])


def hunks_of(lines: list[str]) -> list[tuple[int, int]]:
    """Blank-line-separated runs of non-blank lines, as (first, last)."""
    hunks: list[tuple[int, int]] = []
    start: int | None = None
    for number, line in enumerate(lines, start=1):
        if line.strip():
            start = number if start is None else start
        elif start is not None:
            hunks.append((start, number - 1))
            start = None
    if start is not None:
        hunks.append((start, len(lines)))
    return hunks


def chunks_of(hunks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedy groups of hunks, each at most CHUNK_LINES lines."""
    chunks: list[tuple[int, int]] = []
    start, end = hunks[0]
    for first, last in hunks[1:]:
        if last - start + 1 > CHUNK_LINES:
            chunks.append((start, end))
            start = first
        end = last
    chunks.append((start, end))
    return chunks


def longest_run(selected: list[int]) -> list[int]:
    """The longest run of consecutive line numbers. A single value is a run."""
    if not selected:
        return []
    best: list[int] = [selected[0]]
    current: list[int] = [selected[0]]
    for value in selected[1:]:
        current = [*current, value] if value == current[-1] + 1 else [value]
        if len(current) > len(best):
            best = current
    return best


def span_set(span: tuple[int, int] | None) -> set[int]:
    return set() if span is None else set(range(span[0], span[1] + 1))


def score(guess: set[int], gold: tuple[int, int]) -> dict:
    truth = set(range(gold[0], gold[1] + 1))
    overlap = len(guess & truth)
    precision = overlap / len(guess) if guess else 0.0
    recall = overlap / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(guess | truth)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": overlap / union if union else 0.0,
        "exact": guess == truth,
    }


def fmt(metrics: dict) -> str:
    return (
        f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
        f"F1={metrics['f1']:.2f} IoU={metrics['iou']:.2f} exact={metrics['exact']}"
    )


def choice_probe(
    client: TypeSafeClient,
    ids: list[str],
    body: str,
    chunks: list[tuple[int, int]],
    query: str,
    model: str,
) -> dict:
    """Two requests: pick a chunk, then ask for start, end, and an anchor."""
    options = {
        f"C{index:02d}": {"lines": f"L{first:03d}| through L{last:03d}|"}
        for index, (first, last) in enumerate(chunks, start=1)
    }
    coarse = client.system_one(
        state=body,
        questions={
            "chunk": Choice(
                instructions=(
                    "The state is a source file whose lines are tagged with ids "
                    "like `L014|`. Which chunk of lines contains the code that "
                    f"{query}? Pick the chunk that holds it."
                ),
                criteria=options,
            )
        },
        model=model,
        timeout=120.0,
    )
    chunk_index = int(coarse.choices["chunk"].choice[1:]) - 1
    first, last = chunks[chunk_index]
    chunk_ids = ids[first - 1 : last]
    chunk_lines = [
        line
        for line in body.splitlines()
        if first <= line_of(line.split("|", 1)[0]) <= last
    ]
    chunk_body = "\n".join(chunk_lines)

    fine = client.system_one(
        state=chunk_body,
        questions={
            "start": Choice(
                instructions=(
                    "The state is a source file whose lines are tagged with ids "
                    "like `L014|`. Which line id is the FIRST line of the code "
                    f"that {query}?"
                ),
                criteria=dict.fromkeys(chunk_ids),
            ),
            "end": Choice(
                instructions=(
                    "The state is a source file whose lines are tagged with ids "
                    "like `L014|`. Which line id is the LAST line of the code "
                    f"that {query}?"
                ),
                criteria=dict.fromkeys(chunk_ids),
            ),
            "anchor": Choice(
                instructions=(
                    "The state is a source file whose lines are tagged with ids "
                    "like `L014|`. Which single line id is the most "
                    "representative line of the code that "
                    f"{query}? Pick a line that could only belong to that code."
                ),
                criteria=dict.fromkeys(chunk_ids),
            ),
            "exists": Noul(
                instructions=f"Does this file contain a block of code that {query}?",
                criteria={
                    "true": "The file contains such a block.",
                    "false": "No part of the file does this.",
                },
            ),
        },
        model=model,
        timeout=120.0,
    )

    def top(name: str) -> int:
        probabilities = fine.choices[name].probabilities
        return line_of(max(probabilities, key=probabilities.get))

    start, end = sorted((top("start"), top("end")))
    return {
        "span": (start, end),
        "anchor": top("anchor"),
        "chunk": chunk_index + 1,
        "chunk_span": (first, last),
        "start_conf": fine.choices["start"].confidence,
        "end_conf": fine.choices["end"].confidence,
        "anchor_conf": fine.choices["anchor"].confidence,
        "exists": fine.nouls["exists"].noul,
    }


def per_unit_nouls(
    client: TypeSafeClient,
    body: str,
    units: list[tuple[str, int, int]],
    query: str,
    model: str,
) -> tuple[list[int], float, int]:
    """One Noul per unit. Returns selected lines, the peak probability, and count."""
    questions = {
        f"in::{name}": Noul(
            instructions=(
                "The state is a source file whose lines are tagged with ids. Does "
                f"the line range {first}-{last} belong to code that {query}?"
            ),
            criteria={
                "true": "This range is part of that code.",
                "false": "This range is not part of it.",
            },
        )
        for name, first, last in units
    }
    response = client.system_one(
        state=body, questions=questions, model=model, timeout=180.0
    )
    selected: list[int] = []
    for name, first, last in units:
        if response.nouls[f"in::{name}"].noul >= 0.5:
            selected.extend(range(first, last + 1))
    peak = max(response.nouls[f"in::{name}"].noul for name, _, _ in units)
    return selected, peak, len(units)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="jev-latest")
    args = parser.parse_args()

    lines = DOCUMENT.read_text(encoding="utf-8").splitlines()
    ids, body = tag(lines)
    gold = gold_spans()
    definitions_list = definitions()
    hunks = hunks_of(lines)
    chunks = chunks_of(hunks)
    print(
        f"document: {DOCUMENT.relative_to(ROOT)}  {len(lines)} lines, "
        f"{len(hunks)} hunks, {len(chunks)} chunks {chunks}\n"
    )

    totals: dict[str, list[float]] = {
        "boundaries": [],
        "anchor+ast": [],
        "anchor+indent": [],
        "line-noul": [],
        "hunk-noul": [],
    }
    with build_client(args.model) as client:
        for function, query in QUERIES.items():
            truth = gold[function]
            print(f"query: {query}")
            print(f"  gold        : lines {truth[0]}-{truth[1]}  ({function})")

            probe = choice_probe(client, ids, body, chunks, query, args.model)
            metrics = score(span_set(probe["span"]), truth)
            totals["boundaries"].append(metrics["f1"])
            print(
                f"  boundaries  : lines {probe['span'][0]}-{probe['span'][1]}  "
                f"C{probe['chunk']:02d} conf=({probe['start_conf']:.2f},"
                f"{probe['end_conf']:.2f}) {fmt(metrics)}"
            )

            anchor_span = enclosing(probe["anchor"], definitions_list)
            metrics = score(span_set(anchor_span), truth)
            totals["anchor+ast"].append(metrics["f1"])
            print(
                f"  anchor+ast  : lines {anchor_span[0]}-{anchor_span[1]} "
                f"(anchor {probe['anchor']}) conf={probe['anchor_conf']:.2f} "
                f"{fmt(metrics)}"
            )

            indent_span = block_span(lines, probe["anchor"])
            metrics = score(span_set(indent_span), truth)
            totals["anchor+indent"].append(metrics["f1"])
            print(
                f"  anchor+indent: lines {indent_span[0]}-{indent_span[1]} "
                f"(anchor {probe['anchor']}) {fmt(metrics)}"
            )

            line_units = [(i, line_of(i), line_of(i)) for i in ids]
            selected, peak, count = per_unit_nouls(
                client, body, line_units, query, args.model
            )
            run = longest_run(sorted(selected))
            metrics = score(span_set((run[0], run[-1])) if run else set(), truth)
            totals["line-noul"].append(metrics["f1"])
            print(
                f"  line-noul   : {len(selected)} lines (max_p={peak:.2f}, "
                f"{count} questions) run="
                f"{f'{run[0]}-{run[-1]}' if run else 'none'} {fmt(metrics)}"
            )

            hunk_units = [
                (f"H{index:02d}", first, last)
                for index, (first, last) in enumerate(hunks, start=1)
            ]
            selected, peak, count = per_unit_nouls(
                client, body, hunk_units, query, args.model
            )
            metrics = score(set(selected), truth)
            totals["hunk-noul"].append(metrics["f1"])
            print(
                f"  hunk-noul   : {len(selected)} lines (max_p={peak:.2f}, "
                f"{count} questions) {fmt(metrics)}"
            )
            print()

    print("mean F1 over queries:")
    for name, values in totals.items():
        print(f"  {name:<12} {sum(values) / len(values):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
