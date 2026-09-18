"""Jev as the selector, an LLM as the processor — two selection jobs.

Goal 1 (context): one ``Noul`` per repository file — is it needed? The target
file must survive.
Goal 2 (region): inside the target file, which lines does the processor need?

Region methods:

``hunk-noul``   one ``Noul`` per blank-line hunk: does this range need editing?
                Selected hunks are handed over as-is.
``hunk+block``  the same hunks, each expanded to its enclosing block in code.
                Selection stays with Jev; the boundary stays with code.
``anchor``      one ``Choice`` for the most representative line (chunked first,
                since Choice caps at 255 options), expanded to its block —
                always exactly one region.

The LLM then receives the request and only the selected regions, and returns
the new text of each; code splices them in. The whole file is never sent. A
per-request checker parses the result, so the LLM is free in formatting but not
in outcome. A whole-file LLM edit is the baseline for correctness and size.

    uv run python scripts/jev_edit_pipeline_test.py
"""

import argparse
import ast
import difflib
from dataclasses import dataclass
from pathlib import Path

from typesafe_sdk import Choice, Noul, TypeSafeClient

from xg_project.jev import build_client
from xg_project.llm import build_llm, complete, strip_fence
from xg_project.research import MAX_FILE_BYTES, repo_files

ROOT = Path(__file__).resolve().parent.parent
CHUNK_LINES = 255


def check_path_max(text: str) -> tuple[bool, str]:
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "PATH_MAX" for t in node.targets
        ):
            value = getattr(node.value, "value", None)
            return value == 300, f"PATH_MAX={value}"
    return False, "PATH_MAX not found"


def check_confirm(text: str) -> tuple[bool, str]:
    if "Continue with these files?" in text:
        return False, "old wording still present"
    if "Proceed with these files?" not in text:
        return False, "new wording missing"
    return True, "ok"


def check_ordering(text: str) -> tuple[bool, str]:
    """newest_first is keyword-only, so it lives in kwonlyargs, not args."""
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == "read_entries":
            args = node.args
            positional = zip(
                args.args[len(args.args) - len(args.defaults) :],
                args.defaults,
                strict=True,
            )
            paired = list(positional)
            paired += [
                (arg, default)
                for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
                if default is not None
            ]
            for arg, default in paired:
                if arg.arg == "newest_first":
                    value = getattr(default, "value", None)
                    return value is False, f"newest_first={value}"
            return False, "newest_first not in signature"
    return False, "read_entries not found"


def check_taxonomy(text: str) -> tuple[bool, str]:
    """REQUEST_KIND_CRITERIA is an annotated assignment, not a plain one."""
    members: list[str] | None = None
    keys: list[str] | None = None
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == "RequestKind":
            members = [
                item.value.value
                for item in node.body
                if isinstance(item, ast.Assign)
                and isinstance(item.value, ast.Constant)
            ]
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(getattr(t, "id", None) == "REQUEST_KIND_CRITERIA" for t in targets):
            keys = [k.value for k in node.value.keys]
    if members is None:
        return False, "RequestKind missing"
    if keys is None:
        return False, "REQUEST_KIND_CRITERIA missing"
    if len(members) != 11:
        return False, f"{len(members)} members, expected 11"
    if sorted(members) != sorted(keys):
        return False, "criteria keys do not match members"
    return True, "ok"


@dataclass(frozen=True)
class Edit:
    request: str
    target: str
    gold: list[tuple[int, int]]
    """Line ranges that must be inside the selected region."""
    check: object


EDITS: list[Edit] = [
    Edit(
        request=(
            "Raise the maximum length allowed for a model-proposed file path "
            "from 200 characters to 300."
        ),
        target="src/xg_project/llm/__init__.py",
        gold=[(39, 39)],
        check=check_path_max,
    ),
    Edit(
        request=(
            "In the confirmation prompt shown after research, change the wording "
            "from 'Continue with these files?' to 'Proceed with these files?'."
        ),
        target="src/xg_project/agent/_graph.py",
        gold=[(196, 196)],
        check=check_confirm,
    ),
    Edit(
        request=(
            "Make context windows order files oldest-modified first by default, "
            "keeping the parameter's name."
        ),
        target="src/xg_project/context/__init__.py",
        gold=[(34, 34)],
        check=check_ordering,
    ),
    Edit(
        request=(
            "Add a request kind for dependency upgrades to the taxonomy, "
            "including its criteria entry."
        ),
        target="src/xg_project/jev/_taxonomy.py",
        gold=[(30, 42), (45, 94)],
        check=check_taxonomy,
    ),
]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def lines_of(text: str) -> list[str]:
    return text.splitlines()


def hunks_of(lines: list[str]) -> list[tuple[int, int]]:
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
    chunks: list[tuple[int, int]] = []
    start, end = hunks[0]
    for first, last in hunks[1:]:
        if last - start + 1 > CHUNK_LINES:
            chunks.append((start, end))
            start = first
        end = last
    chunks.append((start, end))
    return chunks


OPEN, CLOSE = "([{", ")]}"


def depth_profile(lines: list[str]) -> list[int]:
    """Bracket depth before each line, ignoring strings and comments.

    Needed because a statement can span lines with the closing bracket at a
    lower indent than its body — which is exactly where a pure indentation scan
    cuts a signature in half.
    """
    before: list[int] = []
    depth = 0
    quote: str | None = None
    for line in lines:
        before.append(depth)
        index = 0
        while index < len(line):
            char = line[index]
            if quote in ('"""', "'''"):
                if line.startswith(quote, index):
                    quote = None
                    index += 3
                    continue
                index += 1
                continue
            if quote in ("'", '"'):
                if char == "\\":
                    index += 2
                    continue
                if char == quote:
                    quote = None
                index += 1
                continue
            if char == "#":
                break
            if line.startswith('"""', index):
                quote = '"""'
                index += 3
                continue
            if line.startswith("'''", index):
                quote = "'''"
                index += 3
                continue
            if char in "\"'":
                quote = char
                index += 1
                continue
            if char in OPEN:
                depth += 1
            elif char in CLOSE:
                depth -= 1
            index += 1
    return before


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def enclosing_block(lines: list[str], first: int, last: int) -> tuple[int, int]:
    """Expand a selected range to the block that contains it.

    A definition or compound statement that starts the range is the block; any
    other line is expanded up to the innermost block that encloses it.
    """
    depth = depth_profile(lines)
    start = first
    while start > 1 and depth[start - 1] > 0:
        start -= 1

    if lines[start - 1].rstrip().endswith(":"):
        header = start
    else:
        header = start
        probe = start - 1
        while probe >= 1:
            if (
                lines[probe - 1].strip()
                and depth[probe - 1] == 0
                and indent_of(lines[probe - 1]) < indent_of(lines[start - 1])
            ):
                header = probe
                break
            probe -= 1

    top = indent_of(lines[header - 1])
    after = header
    while after < len(lines) and depth[after] > 0:
        after += 1

    end = after
    probe = after + 1
    while probe <= len(lines):
        if lines[probe - 1].strip() and depth[probe - 1] == 0:
            if indent_of(lines[probe - 1]) <= top:
                break
            end = probe
        else:
            end = max(end, probe)
        probe += 1
    while end < len(lines) and depth[end] > 0:
        end += 1
    return (header, max(end, after))


def merged(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for first, last in sorted(regions):
        if out and first <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], last))
        else:
            out.append((first, last))
    return out


def coverage(regions: list[tuple[int, int]], gold: list[tuple[int, int]]) -> dict:
    chosen = {n for first, last in regions for n in range(first, last + 1)}
    wanted = {n for first, last in gold for n in range(first, last + 1)}
    return {
        "covered": wanted <= chosen,
        "missing": sorted(wanted - chosen),
        "precision": len(chosen & wanted) / len(chosen) if chosen else 0.0,
        "selected": len(chosen),
    }


def context_step(
    client: TypeSafeClient, files: dict[str, str], request: str, model: str
) -> dict[str, float]:
    response = client.system_one(
        state={"request": request, "files": files},
        questions={
            f"file_{index}": Noul(
                instructions=(
                    f"Does carrying out the request require reading or changing "
                    f"the repository file {path!r}? Its full text is in the state "
                    f"under `files[{path!r}]`."
                ),
                criteria={
                    "true": "The file must be read or changed.",
                    "false": "The file is not needed.",
                },
            )
            for index, path in enumerate(files)
        },
        model=model,
        timeout=120.0,
    )
    return {
        path: response.nouls[f"file_{index}"].noul
        for index, path in enumerate(files)
    }


def region_hunk_noul(
    client: TypeSafeClient, path: str, text: str, request: str, model: str
) -> list[tuple[int, int]]:
    lines = lines_of(text)
    hunks = hunks_of(lines)
    body = "\n".join(
        f"L{number:03d}|{line}" for number, line in enumerate(lines, start=1)
    )
    response = client.system_one(
        state={"request": request, "path": path, "lines": body},
        questions={
            f"h{index:02d}": Noul(
                instructions=(
                    f"`lines` holds the file at `path`, each line tagged by its id. "
                    f"Does carrying out `request` require editing the line range "
                    f"L{first:03d}|-L{last:03d}|?"
                ),
                criteria={
                    "true": "These lines must be edited for the request.",
                    "false": "These lines do not need to change.",
                },
            )
            for index, (first, last) in enumerate(hunks)
        },
        model=model,
        timeout=120.0,
    )
    return merged(
        [
            hunk
            for index, hunk in enumerate(hunks)
            if response.nouls[f"h{index:02d}"].noul >= 0.5
        ]
    )


def region_anchor(
    client: TypeSafeClient, path: str, text: str, request: str, model: str
) -> list[tuple[int, int]]:
    lines = lines_of(text)
    ids = [f"L{number:03d}" for number in range(1, len(lines) + 1)]
    body = "\n".join(
        f"{line_id}|{line}" for line_id, line in zip(ids, lines, strict=True)
    )
    chunks = chunks_of(hunks_of(lines))
    options = {
        f"C{index:02d}": {"lines": f"L{first:03d}| through L{last:03d}|"}
        for index, (first, last) in enumerate(chunks, start=1)
    }
    coarse = client.system_one(
        state={"request": request, "path": path, "lines": body},
        questions={
            "chunk": Choice(
                instructions=(
                    "`lines` holds the file at `path`, each line tagged by its id. "
                    "Which chunk contains the code that must change to carry out "
                    "`request`?"
                ),
                criteria=options,
            )
        },
        model=model,
        timeout=120.0,
    )
    chunk = chunks[int(coarse.choices["chunk"].choice[1:]) - 1]
    chunk_ids = ids[chunk[0] - 1 : chunk[1]]
    chunk_body = "\n".join(
        line
        for line in body.splitlines()
        if chunk[0] <= int(line.split("|", 1)[0][1:]) <= chunk[1]
    )
    fine = client.system_one(
        state={"request": request, "path": path, "lines": chunk_body},
        questions={
            "anchor": Choice(
                instructions=(
                    "`lines` holds part of the file at `path`, each line tagged by "
                    "its id. Which single line id is the most representative line "
                    "of the code that must change for `request`?"
                ),
                criteria=dict.fromkeys(chunk_ids),
            )
        },
        model=model,
        timeout=120.0,
    )
    probabilities = fine.choices["anchor"].probabilities
    anchor = int(max(probabilities, key=probabilities.get)[1:])
    return [enclosing_block(lines, anchor, anchor)]


def region_text(lines: list[str], regions: list[tuple[int, int]]) -> str:
    return "\n".join(
        f"L{number:03d}|{lines[number - 1]}"
        for first, last in regions
        for number in range(first, last + 1)
    )


REGION_SYSTEM = """You edit regions of a source file.
You are given a request and the current text of each region, with line-number
tags. Reply with ONLY the complete new text of that region, without line-number
tags, without prose, and without code fences.
Change only what the request requires. Keep every other line exactly as it is."""

FILE_SYSTEM = """You edit a source file.
Reply with ONLY the complete new file contents, without prose and without code
fences. Change only what the request requires."""


def llm_region_edit(
    llm, path: str, lines: list[str], regions: list[tuple[int, int]], request: str
) -> str:
    edited = list(lines)
    for first, last in sorted(regions, reverse=True):
        current = region_text(edited, [(first, last)])
        prompt = (
            f"Request: {request}\n\n"
            f"File: {path}\n\n"
            f"Region (L{first:03d}| to L{last:03d}|):\n{current}\n\n"
            "New region text:"
        )
        reply = strip_fence(complete(llm, REGION_SYSTEM, prompt))
        edited[first - 1 : last] = reply.splitlines()
    return "\n".join(edited) + "\n"


def llm_whole_file(llm, path: str, text: str, request: str) -> str:
    prompt = f"Request: {request}\n\nFile: {path}\n\n{text}\n\nNew file contents:"
    return strip_fence(complete(llm, FILE_SYSTEM, prompt))


def touched_lines(original: str, edited: str) -> set[int]:
    touched: set[int] = set()
    matcher = difflib.SequenceMatcher(None, lines_of(original), lines_of(edited))
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag != "equal":
            touched.update(range(i1 + 1, i2 + 1))
    return touched


def gold_lines(gold: list[tuple[int, int]]) -> set[int]:
    return {n for first, last in gold for n in range(first, last + 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--no-llm", action="store_true", help="selection metrics only")
    args = parser.parse_args()

    files: dict[str, str] = {}
    for relative in repo_files(ROOT):
        try:
            text = (ROOT / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if len(text.encode("utf-8")) <= MAX_FILE_BYTES:
            files[relative] = text
    llm = None if args.no_llm else build_llm()

    with build_client(args.model) as client:
        for edit in EDITS:
            print(f"request: {edit.request}")
            print(f"  target      : {edit.target}  gold={edit.gold}")

            relevance = context_step(client, files, edit.request, args.model)
            kept = sorted(
                (p for p, prob in relevance.items() if prob >= 0.5),
                key=lambda p: -relevance[p],
            )
            rank = kept.index(edit.target) + 1 if edit.target in kept else None
            print(
                f"  context     : {len(kept)} files kept, target "
                f"{'rank ' + str(rank) if rank else 'MISSING'} "
                f"(p={relevance[edit.target]:.2f})"
            )

            text = files[edit.target]
            lines = lines_of(text)
            hunks = region_hunk_noul(client, edit.target, text, edit.request, args.model)
            blocks = merged(
                [enclosing_block(lines, first, last) for first, last in hunks]
            )
            anchor = region_anchor(client, edit.target, text, edit.request, args.model)
            methods = {
                "hunk-noul": hunks,
                "hunk+block": blocks,
                "anchor": anchor,
            }
            for name, regions in methods.items():
                metrics = coverage(regions, edit.gold)
                print(
                    f"  {name:<10}: {regions} covered={metrics['covered']} "
                    f"missing={metrics['missing'][:6]}"
                    f"{'...' if len(metrics['missing']) > 6 else ''} "
                    f"precision={metrics['precision']:.2f}"
                )

            if llm is None:
                print()
                continue

            gold = gold_lines(edit.gold)
            for name, regions in methods.items():
                if not regions:
                    print(f"  LLM {name:<10}: no region selected")
                    continue
                edited = llm_region_edit(llm, edit.target, lines, regions, edit.request)
                try:
                    ok, detail = edit.check(edited)
                except SyntaxError as exc:
                    ok, detail = False, f"SyntaxError: {exc}"
                touched = touched_lines(text, edited)
                print(
                    f"  LLM {name:<10}: verified={ok} ({detail})  "
                    f"touched={len(touched)} outside_gold={sorted(touched - gold)}"
                )

            whole = llm_whole_file(llm, edit.target, text, edit.request)
            try:
                ok, detail = edit.check(whole)
            except SyntaxError as exc:
                ok, detail = False, f"SyntaxError: {exc}"
            region_chars = len(region_text(lines, methods["hunk+block"]))
            print(
                f"  LLM whole-file : verified={ok} ({detail})  "
                f"prompt={len(text)} chars, reply={len(whole)} chars, "
                f"hunk+block prompt={region_chars} chars"
            )
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
