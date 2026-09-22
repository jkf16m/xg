"""The built-in graph: the workflows xg ships with.

Every node is named through :func:`~xg_project.graph._registry.xg`, so the
built-in keys are ``_XG_ORIGIN``, ``_XG_FILTER``, and so on. A node a user adds
does not carry the prefix, which is how the two are told apart.

The graph is three workflows that share an origin.

``_XG_ORIGIN``
    The origin, and a decision node. The user states the goal here; the node
    introduces it into the state as a sentence, and Jev chooses the branch.

**The project workflow**, ``_XG_ORIGIN -> _XG_SELECT_MODULE -> _XG_FILTER -> _XG_SORT``,
reads the project and then either answers a question about it or edits a file:

``_XG_SELECT_MODULE``
    Finds the context module the request belongs to. A context module is a folder
    that declares what it exposes in a ``.xg/module.json`` beside it; once one is
    selected, its exposures and its imports' become the only files the rest of
    the workflow may read. One module is the context without being asked about,
    several are put to Jev and the most important is taken, and no modules at all
    means the whole project is the context — the behaviour the graph had before
    modules existed, reached by the absence of a declaration rather than a
    choice.

``_XG_FILTER``
    Reads the files in scope — what the selected module exposes, or the whole tree
    when no module was selected, minus whatever ``.gitignore`` and ``.xgignore``
    exclude — and then asks Jev, once per file, whether that file is one of the
    files the goal is about. It introduces the files above the threshold as the
    candidates, so what survives is a decision Jev made rather than a directory
    listing.

    This is the gathering step, and it is deliberately not the ranking one. The
    question is about belonging, not importance, so a file that supports the goal
    without being central to it is kept — a request to explain something is
    answered out of the supporting files as much as the defining one. Ranking is
    the next node's job, over exactly this set.

``_XG_SORT``
    Ranks what FILTER kept. It asks Jev a second, different question — how
    central each file is to fulfilling the goal — and introduces the same files
    ordered most important first. Only the order is decided here; nothing is
    dropped, because the files may be needed for an answer as well as an edit.
    SORT is where the project workflow branches, and Jev chooses the leaf.

``_XG_EDIT``
    A generative leaf. It takes the file SORT ranked first and asks the executor
    for one edit to it — a forced tool call carrying ``old_text`` and
    ``new_text``. That is a proposal: nothing is written until the user accepts.

``_XG_ANSWER``
    A generative leaf. It hands SORT's ranked files and the goal to the executor
    and introduces the prose answer. Nothing is proposed and nothing is gated:
    reading is the one thing that does not change anything.

**The command workflow**, ``_XG_ORIGIN -> _XG_COMMAND``, runs one shell command
with no project context at all:

``_XG_COMMAND``
    A generative leaf. The executor proposes one shell command, which the user
    accepts or rejects before anything runs.

**The add workflow**, ``_XG_ORIGIN -> _XG_ADD``, creates a new file, also with no
project context:

``_XG_ADD``
    A generative leaf. The executor proposes a new file — a path that does not
    exist yet, and the whole content for it — as a forced tool call. That is a
    proposal: nothing is created until the user accepts, and a path that already
    exists is refused rather than overwritten.

Both of those are self-contained: the request says what to do and there is
nothing in the project to consult about it. The project workflow is the one that
reads, which is why it is the one with several steps.

Every summary describes what its node *does*. None of them says when to route to
it: a node does not know who links to it, and a description written for one
parent is wrong for the next. Jev is shown those descriptions and decides by
matching the request against the effects, so a description of the node's own
behaviour is the whole contract.
"""

from __future__ import annotations

from collections.abc import Mapping

from xg_project.graph._registry import (
    NodeDeclaration,
    NodeInput,
    NodeKind,
    Registry,
    xg,
)
from xg_project.jev import (
    IMPORTANCE_CRITERIA,
    IMPORTANCE_INSTRUCTIONS,
    MODULE_CRITERIA,
    MODULE_INSTRUCTIONS,
)
from xg_project.llm import AddProposal, Answer, CommandProposal, EditProposal
from xg_project.module import Module, Modules, Scope, discover, scope
from xg_project.read import read_tree

ORIGIN = xg("ORIGIN")
COMMAND = xg("COMMAND")
ADD = xg("ADD")
SELECT_MODULE = xg("SELECT_MODULE")
FILTER = xg("FILTER")
SORT = xg("SORT")
EDIT = xg("EDIT")
ANSWER = xg("ANSWER")


def _origin(inp: NodeInput) -> str:
    """State the goal, and introduce it as the state the origin owns.

    Stating the goal is all that happens here. Which workflow handles it is the
    decision Jev is asked for immediately afterwards.
    """
    return f"the user wants: {inp.prompt}"


async def _select_module(inp: NodeInput) -> dict[str, object]:
    """Choose the context module the goal belongs to, and bound the read to it.

    Discovery is deterministic and selection is not, which is FILTER's shape one
    level up. Every ``.xg/module.json`` below the root is found by walking the
    same candidate set the reader uses, so a manifest inside a folder the ignores
    exclude is not found either; a manifest that cannot be used becomes a line in
    the state instead of a failed run.

    One module needs no decision — it is the only declared context, and asking
    would spend a round trip to be told what the tree already says. Several are
    put to Jev, and the highest scoring one above the threshold is taken. None
    above the threshold is reported as no module rather than as the best of a bad
    set: the selected module is the only context the rest of the run may read, so
    a wrong selection hides the right files completely, and the whole tree is the
    safer thing to fall back to.

    The state separates three outcomes that are easy to confuse. ``module`` empty
    means nothing bounded the read, so the whole project is in scope.
    ``module`` set with ``includes`` empty means a module was chosen and it
    admits nothing. ``missing`` and ``problems`` are the declarations that did not
    resolve and the manifests that did not parse.
    """
    if inp.root is None:
        return _selection(None, modules=None, problem="no project root is configured")

    modules = discover(inp.root)
    if not modules.found:
        return _selection(None, modules=modules)

    chosen, scores, problem = await _choose_module(inp, modules)
    if chosen is None:
        return _selection(None, modules=modules, scores=scores, problem=problem)
    return _selection(chosen, modules=modules, scores=scores, resolved=scope(modules, chosen))


async def _choose_module(
    inp: NodeInput, modules: Modules
) -> tuple[Module | None, dict[str, float], str]:
    """The module the request belongs to, or why none was taken.

    The modules' own declarations are the question's state, keyed by module id,
    so the model judges what each module says about itself and nothing else — no
    file contents are read to make this decision.
    """
    if len(modules.found) == 1:
        return modules.found[0], {}, ""
    if inp.jev is None:
        return None, {}, "no Jev is attached, so no module can be chosen"

    selection = await inp.jev.select(
        files=modules.briefs(),
        prompt=_goal(inp),
        instructions=MODULE_INSTRUCTIONS,
        criteria=MODULE_CRITERIA,
    )
    if selection.problem:
        return None, dict(selection.scores), selection.problem

    by_id = {module.id(modules.root): module for module in modules.found}
    kept = [name for name in selection.files if name in by_id]
    if not kept:
        return (
            None,
            dict(selection.scores),
            "no module's context matched the request, so the whole project is in scope",
        )
    best = max(kept, key=lambda name: selection.scores.get(name, 0.0))
    return by_id[best], dict(selection.scores), ""


def _selection(
    chosen: Module | None,
    *,
    modules: Modules | None,
    scores: Mapping[str, float] | None = None,
    resolved: Scope | None = None,
    problem: str = "",
) -> dict[str, object]:
    """Assemble SELECT_MODULE's state from a choice, or from the lack of one."""
    return {
        "module": "" if chosen is None or modules is None else chosen.id(modules.root),
        "name": "" if chosen is None else chosen.name,
        "modules": {} if modules is None else modules.briefs(),
        "scores": dict(scores or {}),
        "includes": [] if resolved is None else list(resolved.files),
        "missing": [] if resolved is None else list(resolved.missing),
        "problems": [] if modules is None else list(modules.problems),
        "problem": problem,
    }


def _bound_files(value: object) -> list[str] | None:
    """The files a selected module admits, or ``None`` when nothing bounded the read.

    ``None`` and ``[]`` are opposites here: no scope means the whole project, and
    an empty scope means the module admits nothing. Reading `includes` alone
    would collapse the two, and a module that exposes nothing would silently
    widen the read to everything it was meant to narrow.
    """
    if isinstance(value, Mapping) and value.get("module"):
        includes = value.get("includes")
        if isinstance(includes, (list, tuple)):
            return [str(path) for path in includes]
        return []
    return None


async def _command(inp: NodeInput) -> CommandProposal:
    """Ask the executor for one shell command, and introduce it as a proposal.

    No files are read and the state is not consulted: this workflow's defining
    property is that the request is self-contained.
    """
    if inp.executor is None:
        return CommandProposal(problem="no executor is attached, so no command can be proposed")
    return await inp.executor.propose_command(prompt=inp.prompt, context=_context(inp))


async def _add(inp: NodeInput) -> AddProposal:
    """Ask the executor for one new file, and introduce it as a proposal.

    No files are read and the state is not consulted, for the same reason COMMAND
    does not read anything: the request is self-contained. The file being asked
    for is one that does not exist yet, so there is nothing in the project to read
    about it — the path is the model's to choose, and the workflow's only job is
    to hold the result until the user decides.
    """
    if inp.executor is None:
        return AddProposal(problem="no executor is attached, so no file can be proposed")
    return await inp.executor.propose_add(prompt=inp.prompt, context=_context(inp))


async def _filter(inp: NodeInput) -> dict[str, object]:
    """Ignore deterministically, then ask Jev which surviving files the request is about.

    Two stages, in this order, and the order is the point. `read_tree` is the
    deterministic stage: it walks the root honouring ``.gitignore`` and
    ``.xgignore`` — and, when a context module was selected, opening only the
    files that module exposes. Jev is the inference stage: it answers one
    ``Noul`` question per file that survived the ignore rules, and only those
    above the threshold are introduced as candidates. A file Jev did not answer
    for is dropped rather than passed on with a caveat — its relevance is unknown,
    which is not the same as relevant.

    This node gathers and does not rank. Its question asks whether a file belongs
    to the set the request is about, so a file that supports the request without
    being the centre of it is kept. Ranking is SORT's question, asked next and
    over exactly this set: splitting them is what lets a request to explain or
    change something arrive with all the files that explain it, and still leaves
    one file nominated to be edited.

    ``read`` is how many files the deterministic stage produced, kept beside the
    kept set so the state shows both halves of the step: what was in scope, and
    what the model judged worth keeping.
    """
    if inp.root is None:
        return {
            "files": {},
            "scores": {},
            "read": 0,
            "skipped": [],
            "dropped": [],
            "problem": "no project root is configured",
        }

    tree = read_tree(inp.root, include=_bound_files(inp.state.get(SELECT_MODULE)))
    read = len(tree.files)
    if not tree.files:
        return {
            "files": {},
            "scores": {},
            "read": 0,
            "skipped": list(tree.skipped),
            "dropped": [],
            "problem": "",
        }
    if inp.jev is None:
        return {
            "files": {},
            "scores": {},
            "read": read,
            "skipped": list(tree.skipped),
            "dropped": [],
            "problem": "no Jev is attached, so no file can be selected",
        }

    selection = await inp.jev.select(files=tree.files, prompt=_goal(inp))
    return {
        "files": dict(selection.files),
        "scores": dict(selection.scores),
        "read": read,
        "skipped": list(tree.skipped),
        "dropped": list(selection.dropped),
        "problem": selection.problem or "",
    }


async def _sort(inp: NodeInput) -> dict[str, object]:
    """Ask Jev how central each filtered file is, and introduce them ranked.

    This is a second question, not a repeat of FILTER's: FILTER asked whether a
    file belongs at all, SORT asks which of the survivors matters most. The
    scores order the map; every file is kept, only the order changes, so EDIT can
    be asked for any of them.
    """
    files = _files(inp.state.get(FILTER))
    if not files:
        return {"files": {}, "scores": {}, "problem": "there are no filtered files to rank"}
    if inp.jev is None:
        return {
            "files": files,
            "scores": {},
            "problem": "no Jev is attached, so the files cannot be ranked",
        }

    prompt = _goal(inp)
    selection = await inp.jev.select(
        files=files,
        prompt=prompt,
        instructions=IMPORTANCE_INSTRUCTIONS,
        criteria=IMPORTANCE_CRITERIA,
    )
    ranked = {
        path: files[path]
        for path in sorted(files, key=lambda name: selection.scores.get(name, 0.0), reverse=True)
    }
    return {
        "files": ranked,
        "scores": dict(selection.scores),
        "dropped": list(selection.dropped),
        "problem": selection.problem or "",
    }


async def _edit(inp: NodeInput) -> EditProposal:
    """Ask the executor for one edit to the top-ranked file, as a proposal.

    The file comes from SORT's state, not from the model: EDIT edits the file the
    workflow chose. With an empty ranking there is nothing to edit, and that is
    reported rather than guessed at.
    """
    ranked = _files(inp.state.get(SORT))
    if not ranked:
        return EditProposal(problem="no file was ranked to edit")

    path, content = next(iter(ranked.items()))
    if inp.executor is None:
        return EditProposal(
            path=path, problem="no executor is attached, so no edit can be proposed"
        )
    return await inp.executor.propose_edit(
        path=path, content=content, prompt=_goal(inp), context=_context(inp)
    )


async def _answer(inp: NodeInput) -> Answer:
    """Answer the goal in prose, from the files SORT ranked.

    The files come from SORT when it has run and FILTER otherwise, so the node
    is still usable if a user walked straight to it. No files means the answer
    has no ground to stand on, and that is reported rather than guessed at — an
    answer invented without the project is the one failure mode reading exists to
    prevent.
    """
    files = _files(inp.state.get(SORT)) or _files(inp.state.get(FILTER))
    if not files:
        return Answer(problem="no files were selected, so there is nothing to answer from")
    if inp.executor is None:
        return Answer(problem="no executor is attached, so no answer can be produced")
    return await inp.executor.answer(files=files, prompt=_goal(inp), context=_context(inp))


def _files(value: object) -> dict[str, str]:
    """Read a ``{"files": {path: content}}`` entry, or an empty map."""
    if isinstance(value, Mapping):
        files = value.get("files")
        if isinstance(files, Mapping):
            return {str(path): str(content) for path, content in files.items()}
    return {}


def _goal(inp: NodeInput) -> str:
    """The prompt a later node should work from: the origin's goal if it is there."""
    value = inp.state.get(ORIGIN)
    return value if isinstance(value, str) else inp.prompt


def _context(inp: NodeInput) -> list[str]:
    """The state so far, as lines, for a model that is asked to produce something."""
    return [f"{name}: {value}" for name, value in inp.state.items() if isinstance(value, str)]


def default_graph() -> Registry:
    """Build the graph xg always has, before discovery adds to it."""
    registry = Registry()
    registry.add(
        NodeDeclaration(
            name=ORIGIN,
            level=0,
            kind=NodeKind.DECISION,
            summary=(
                "Where a run starts: records the user's request as the goal. "
                "Nothing is read and nothing runs here."
            ),
            children=(COMMAND, ADD, SELECT_MODULE),
            handle=_origin,
        )
    )
    registry.add(
        NodeDeclaration(
            name=ADD,
            level=1,
            kind=NodeKind.GENERATIVE,
            summary=(
                "Adds one new file to the project: it proposes a path that does "
                "not exist yet, together with the whole content for it. It reads "
                "nothing from the project and cannot change a file that is "
                "already there; it is for a request that explicitly asks for a "
                "new file to be created."
            ),
            handle=_add,
            proposes=True,
        )
    )
    registry.add(
        NodeDeclaration(
            name=SELECT_MODULE,
            level=1,
            kind=NodeKind.DECISION,
            summary=(
                "Chooses which context module the request belongs to — a folder "
                "that declares what it exposes in .xg/module.json — and bounds "
                "every later read to the files that module exposes. With no "
                "module declared, the whole project stays in scope."
            ),
            children=(FILTER,),
            handle=_select_module,
        )
    )
    registry.add(
        NodeDeclaration(
            name=COMMAND,
            level=1,
            kind=NodeKind.GENERATIVE,
            summary=(
                "Runs one shell command and reports its output. It reads nothing "
                "from the project and cannot answer a question about it; it is "
                "for a request that explicitly asks for a command to be run."
            ),
            handle=_command,
            proposes=True,
        )
    )
    registry.add(
        NodeDeclaration(
            name=FILTER,
            level=2,
            kind=NodeKind.DECISION,
            summary=(
                "Reads the files in scope — what the chosen context module "
                "exposes, otherwise the whole project, less anything .gitignore "
                "or .xgignore excludes — and gathers every file the request is "
                "about."
            ),
            children=(SORT,),
            handle=_filter,
        )
    )
    registry.add(
        NodeDeclaration(
            name=SORT,
            level=3,
            kind=NodeKind.DECISION,
            summary=(
                "Orders the files that were kept by how central each one is to "
                "the request, most important first."
            ),
            children=(EDIT, ANSWER),
            handle=_sort,
        )
    )
    registry.add(
        NodeDeclaration(
            name=EDIT,
            level=4,
            kind=NodeKind.GENERATIVE,
            summary=(
                "Proposes one edit to the most important project file: a literal "
                "old_text and the new_text that replaces it."
            ),
            handle=_edit,
            proposes=True,
        )
    )
    registry.add(
        NodeDeclaration(
            name=ANSWER,
            level=4,
            kind=NodeKind.GENERATIVE,
            summary=("Answers the request in prose, from the files that were kept."),
            handle=_answer,
        )
    )
    return registry
