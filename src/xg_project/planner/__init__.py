"""xg_project.planner — build a plan with Jev, one node at a time.

Jev cannot emit a plan as text: its answers are ``Choice``/``Noul``/``Score``
and none of the primitives has an ordering operation. A plan is therefore
assembled the way the TypeSafe docs describe larger workflows -- "your code can
inspect and combine its answers into predictable workflows"
(https://docs.typesafe.ai/concepts/system-one).

The mechanism is a loop. Each round sends Jev a state that contains the plan
built so far, and asks the same two questions about that state:

    done    Noul    is the plan complete?
    next    Choice  the single next node, as ``edit:<path>``, ``read:<path>``,
                    ``write:new_file``, or ``answer``

Code then either stops -- ``done`` fired, the node was too uncertain, the node
repeats one already in the plan, or the round budget ran out -- or appends it
and loops. The plan is the transcript of those decisions; no round ever sees a
task it has not been handed in the state.

Why one composite ``next`` question instead of a ``step`` kind plus a separate
``target`` file: questions in a ``system_one`` call are evaluated in parallel
against the same state, so two questions about the same node cannot see each
other's answer. A live run produced ``step: edit`` together with ``target:
none`` -- a node that changes a file, but no file. One label that carries both
the kind and the target makes that contradiction unrepresentable. The cost is
``2 * len(files) + 2`` options, so the file set is capped at ``MAX_TARGETS``.

Two consequences of building a plan this way, both deliberate:

*   A node can name a file that already exists (``edit:``/``read:`` with a path
    from the state) or say that a *new* file is needed (``write:new_file``). It
    cannot say the new file's name, because a ``Choice`` label must come from
    the criteria map we send. Naming a new path stays with the generative model
    at the last moment, exactly as ``agent/_graph.py`` already does.
*   Ordering is the loop, not an answer. Round N sees rounds 0..N-1 in its
    state, so "B after A" is expressed by A already being in the plan.

Public API
----------
    plan(request, *, files, client=None, model=None, ...) -> Plan
    build_state(request, files, plan, results=None) -> dict[str, object]
    build_questions(files, plan) -> dict[str, Choice | Noul]
    step_criteria(files) -> dict[str, JSONContent]

Types:
    Plan, Step, StepKind

Errors:
    JevError -> a TypeSafe-side failure the caller can retry (from xg_project.jev)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typesafe_sdk import Choice, JSONContent, Noul, TypeSafeClient, TypeSafeError

from xg_project.jev import JevError, build_client

NEW_FILE = "new_file"
"""Target segment for a ``write`` node whose path is named later, by the model."""

MAX_TARGETS = 120
"""Cap on the file set. Each file contributes two options; ``Choice`` takes 255."""

MAX_STEPS = 6
"""Hard cap on rounds. The loop is serial, so each step is another round trip."""

NODE_FLOOR = 0.4
"""Minimum confidence for a ``next`` node to be added to the plan.

Deliberately lower than ``jev.CONFIDENCE_FLOOR`` (0.5), which gates the
10-way ``request_kind`` question. A ``next`` node is a composite over
``2 * len(files) + 2`` options, so probability legitimately spreads across
related options (``edit:a`` vs ``read:a`` vs ``edit:b``) even when the top
option is right; a live run returned the correct node at 0.47-0.53. The plan
is shown to the user before anything runs, so a moderately confident node is
actionable; a low one still stops the loop."""

DONE_THRESHOLD = 0.5
"""At or above this ``done`` probability the plan is considered complete."""


class StepKind(StrEnum):
    """What a plan node does. A subset of the graph's capability vocabulary.

    ``answer`` and ``read`` are in the graph already (research reads files).
    ``write`` and ``edit`` are the mutations. A kind is only listed when the
    pipeline has -- or will have -- a node for it; offering a label no node can
    execute would route to nothing, which is why the request taxonomy also
    leaves out ``command`` (see ``jev/_taxonomy.py``).
    """

    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    ANSWER = "answer"


def step_criteria(
    files: dict[str, str], plan: tuple[Step, ...] | list[Step] = ()
) -> dict[str, JSONContent]:
    """The ``next`` option map: one ``edit`` and one ``read`` per remaining file.

    Nodes already in ``plan`` are removed from the option space. A live run had
the model answer ``done=0.18`` (more work needed) while proposing
``edit:_graph.py`` again at confidence 0.89, because the plan being in the
state did not stop it from re-picking the most salient file. Excluding planned
nodes makes progress a property of the question, not of the model's
cooperation; the loop then stops when this returns empty.

    Criteria are objects with ``what``/``not_for``/``examples`` because the
    docs report that structured descriptions separate confusable options better
    than one-line strings (https://docs.typesafe.ai/primitives/choice).
    """
    if len(files) > MAX_TARGETS:
        raise ValueError(
            f"at most {MAX_TARGETS} files are supported, got {len(files)}"
        )
    planned = {(step.kind, step.target) for step in plan}
    criteria: dict[str, JSONContent] = {}
    for path in files:
        if (StepKind.EDIT, path) not in planned:
            criteria[f"{StepKind.EDIT.value}:{path}"] = {
                "what": (
                    f"Change the contents of the existing file {path!r} so the "
                    "request is satisfied."
                ),
                "not_for": "A file that must be created; a node that only reads.",
                "examples": ["Add a flag to an existing CLI module"],
            }
        if (StepKind.READ, path) not in planned:
            criteria[f"{StepKind.READ.value}:{path}"] = {
                "what": f"Read {path!r} to gather evidence. Nothing is created or changed.",
                "not_for": "The request asks for this file to be created or changed.",
                "examples": ["Open a module to see how it works before explaining it"],
            }
    if (StepKind.WRITE, "") not in planned:
        criteria[f"{StepKind.WRITE.value}:{NEW_FILE}"] = {
            "what": "Create a file that does not exist yet; its path is chosen later.",
            "not_for": "Changing a file that already exists.",
            "examples": ["Create a new module the request asks for"],
        }
    if (StepKind.ANSWER, "") not in planned:
        criteria[StepKind.ANSWER.value] = {
            "what": "Produce the reply from what is already in the state. No file is changed.",
            "not_for": "The request asks for a file to be created or changed.",
            "examples": ["Explain what a function does"],
        }
    return criteria


@dataclass(frozen=True)
class Step:
    """One planned node: what it does and which file it does it to.

    ``target`` is a repository-relative path for a node on an existing file, or
    ``""`` when the node is an ``answer`` or a ``write`` whose new path is not
    yet named.
    """

    kind: StepKind
    target: str = ""

    @property
    def new_file(self) -> bool:
        """Whether this node needs a path the generative model must supply."""
        return self.kind is StepKind.WRITE and not self.target

    def to_dict(self) -> dict[str, object]:
        return {"step": self.kind.value, "target": self.target or None}


@dataclass(frozen=True)
class Plan:
    """The ordered nodes Jev built, and why the loop stopped."""

    request: str
    steps: tuple[Step, ...] = ()
    stopped: str = ""
    """Why the loop ended: ``done``, ``budget``, or a guard's explanation."""
    rounds: int = 0
    """How many ``system_one`` calls were made."""
    model: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.steps

    def to_dict(self) -> dict[str, object]:
        """A JSON-serializable view, for logging and a CLI."""
        return {
            "request": self.request,
            "model": self.model,
            "stopped": self.stopped,
            "rounds": self.rounds,
            "steps": [step.to_dict() for step in self.steps],
        }


def _step_dicts(plan: tuple[Step, ...] | list[Step]) -> list[dict[str, object]]:
    return [
        {"index": index, **step.to_dict()} for index, step in enumerate(plan)
    ]


def build_state(
    request: str,
    files: dict[str, str],
    plan: tuple[Step, ...] | list[Step] = (),
    results: dict[int, str] | None = None,
) -> dict[str, object]:
    """Compose the state one planning round is evaluated against.

    The plan so far is the load-bearing part: it is how a round knows which
    nodes already exist, so the model does not propose them again and can order
    the next one relative to them. ``results`` carries the outcome of nodes the
    caller has already executed, keyed by node index, when there are any.
    """
    state: dict[str, object] = {
        "request": request,
        "files": dict(files),
        "plan": _step_dicts(plan),
    }
    if results:
        state["results"] = {str(index): text for index, text in results.items()}
    return state


def build_questions(
    files: dict[str, str], plan: tuple[Step, ...] | list[Step] = ()
) -> dict[str, Choice | Noul]:
    """The question set asked in each planning round.

    The plan is not interpolated into the criteria text: it is in the shared
    state, and keeping it there means one question set can be reused for every
    round while the answers stay comparable across rounds.
    """
    done_instructions = (
        "Is the plan in the state complete enough to carry out the request? "
        "Answer yes only when no further node is needed. Answer no when at "
        "least one more node must be added."
    )
    if plan:
        done_instructions += (
            " Nodes already in the plan must not be planned again; plan what "
            "comes after them."
        )
    return {
        "done": Noul(
            instructions=done_instructions,
            criteria={
                "true": "The plan already covers everything the request needs.",
                "false": "At least one more node is still required.",
            },
        ),
        "next": Choice(
            instructions=(
                "What is the single next node in the plan for `request`? Pick "
                "the next step that has to happen, not the whole remaining "
                "task. Nodes already listed in the state's plan are removed "
                "from the options because they are done; choose among the rest. "
                "If `request` asks for a file to be created or changed, the "
                "node is `edit:<path>` or `write:new_file`, never `answer`."
            ),
            criteria=step_criteria(files, plan),
        ),
    }


def _ask(
    client: TypeSafeClient,
    request: str,
    files: dict[str, str],
    plan: tuple[Step, ...] | list[Step],
    results: dict[int, str] | None,
    model: str | None,
):
    """Send one round. Raises ``JevError`` on a transport-side failure."""
    try:
        return client.system_one(
            state=build_state(request, files, plan, results),
            questions=build_questions(files, plan),
            model=model,
        )
    except TypeSafeError as exc:
        raise JevError(str(exc)) from exc


def _parse(label: str) -> Step | None:
    """Turn a ``next`` label into a :class:`Step`, or ``None`` when unknown.

    This only reads the label; it does not check the target against the file
    set, which ``_build`` does so the stop reason can name the bad target.
    """
    if label == StepKind.ANSWER.value:
        return Step(StepKind.ANSWER, "")
    kind_label, _, target = label.partition(":")
    try:
        kind = StepKind(kind_label)
    except ValueError:
        return None
    return Step(kind, target)


def plan(
    request: str,
    *,
    files: dict[str, str],
    client: TypeSafeClient | None = None,
    model: str | None = None,
    max_steps: int = MAX_STEPS,
    done_threshold: float = DONE_THRESHOLD,
    node_floor: float = NODE_FLOOR,
    results: dict[int, str] | None = None,
) -> Plan:
    """Build a plan by asking Jev "done?" and "which node next?" each round.

    ``files`` is the known-file map the ``next`` options are drawn from --
    ``Research.contents`` is the intended source, so planning sees the same
    file text research selected on.

    Every guard stops the loop rather than guessing: an unknown label, a
    ``read``/``edit`` target that is not a known file, a ``write`` that did not
    say ``new_file``, a node choice below ``node_floor``, a node that repeats
    one already in the plan, or the ``max_steps`` budget. ``Plan.stopped``
    records which. ``client`` lets a caller reuse a connection; when omitted,
    one is built and closed.
    """
    if not request.strip():
        raise ValueError("request must not be empty")

    if client is not None:
        return _build(
            request, files, client, model, max_steps, done_threshold, node_floor, results
        )
    with build_client(model) as owned:
        return _build(
            request, files, owned, model, max_steps, done_threshold, node_floor, results
        )


def _build(
    request: str,
    files: dict[str, str],
    client: TypeSafeClient,
    model: str | None,
    max_steps: int,
    done_threshold: float,
    node_floor: float,
    results: dict[int, str] | None,
) -> Plan:
    """Run the planning loop against one client."""
    steps: list[Step] = []
    stopped = "budget"
    used_model = ""
    rounds = 0

    while rounds < max_steps:
        if not step_criteria(files, steps):
            stopped = "no unplanned node remains"
            break
        response = _ask(client, request, files, steps, results, model)
        rounds += 1
        used_model = response.model

        if response.nouls["done"].noul >= done_threshold:
            stopped = "done"
            break

        chosen = response.choices["next"]
        node = _parse(chosen.choice)
        if node is None:
            stopped = f"unexecutable node {chosen.choice!r}"
            break
        if chosen.confidence < node_floor:
            stopped = "the node choice was too uncertain to act on"
            break

        if node.kind is StepKind.WRITE:
            if node.target != NEW_FILE:
                stopped = (
                    f"write node named {node.target!r} instead of {NEW_FILE!r}"
                )
                break
            node = Step(StepKind.WRITE, "")
        elif node.kind in {StepKind.READ, StepKind.EDIT} and node.target not in files:
            stopped = f"target {node.target!r} is not a known file"
            break

        if node in steps:
            stopped = "the model proposed a node already in the plan"
            break
        steps.append(node)

    return Plan(
        request=request,
        steps=tuple(steps),
        stopped=stopped,
        rounds=rounds,
        model=used_model,
    )


__all__ = [
    "DONE_THRESHOLD",
    "MAX_STEPS",
    "MAX_TARGETS",
    "NEW_FILE",
    "NODE_FLOOR",
    "Plan",
    "Step",
    "StepKind",
    "build_questions",
    "build_state",
    "plan",
    "step_criteria",
]
