"""Request taxonomy for the xg classifier.

The classifier answers the primary ``Choice`` question (``request_kind``), a
second one about what fulfilling the request will *do* (``operation``), and the
signals the router needs: how hard the request is (``complexity``), whether the
repository must be modified at all (``changes_code``), and whether the request
is risky (``is_destructive``).

``request_kind`` and ``operation`` are different axes and both are needed. The
kind says what the user is asking about (a bug, a refactor, a question); the
operation says what the work will produce (a new file, a change to an existing
file, or nothing but an answer). A refactor can write a new module.

``command`` is deliberately absent for now: the graph has no command path, so
offering the label would route requests to a node that does not exist. Requests
that would have been commands fall to ``other`` and are rerouted to the user.

How far-reaching the request is (*scope*) is deliberately not asked here. It is
an outcome of the ``local_research`` step, which finds the relevant files; the
number of files it finds is the scope.

Criteria text is the model's only input for these decisions, so each option
says what it *is* and, where two options are easy to confuse, what belongs to
the neighbour instead. See https://docs.typesafe.ai/primitives/choice.
"""

from enum import StrEnum


class RequestKind(StrEnum):
    """What the user is asking for."""

    QUESTION = "question"
    EDIT_FEATURE = "edit_feature"
    NEW_FEATURE = "new_feature"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"
    CONFIG = "config"
    CONFUSED = "confused"
    OTHER = "other"


REQUEST_KIND_CRITERIA: dict[str, str] = {
    "question": (
        "The user asks about the codebase, its behavior, or how to use it and "
        "does not ask for any file to change. Answering may involve reading "
        "code, but no edit is wanted."
    ),
    "edit_feature": (
        "The user wants existing behavior of an existing feature changed. The "
        "feature already exists and its behavior is being altered, extended, or "
        "corrected in a way that is not purely a defect fix, a restructure, or "
        "new functionality."
    ),
    "new_feature": (
        "The user wants functionality that does not exist yet to be added. "
        "Contrast with edit_feature: there, the behavior already exists and is "
        "being changed."
    ),
    "bug_fix": (
        "The user reports that something is broken, failing, or behaving "
        "incorrectly, and wants the defect corrected. The behavior is intended "
        "to exist and currently does not work as intended."
    ),
    "refactor": (
        "The user wants the code restructured, renamed, reorganized, or "
        "cleaned up without changing observable behavior. No new behavior and "
        "no defect fix."
    ),
    "test": (
        "The user wants tests added, updated, fixed, or run. The primary "
        "deliverable is test code or a test result."
    ),
    "docs": (
        "The user wants documentation written or updated: READMEs, docstrings, "
        "comments, or guides. The primary deliverable is prose, not behavior."
    ),
    "config": (
        "The user wants project tooling, dependencies, build, packaging, or "
        "settings changed: pyproject, lockfiles, CI, linters, or .xg settings."
    ),
    "confused": (
        "The request cannot be understood well enough to act on. Essential "
        "information is missing (no target, no goal, or an unnamed thing), the "
        "request contradicts itself, or it is not a request at all: a fragment, "
        "small talk, or a note to self. Choose this rather than guessing."
    ),
    "other": (
        "The request can be understood, but it fits none of the options above, "
        "or it spans several kinds with no clear primary."
    ),
}


class Operation(StrEnum):
    """What fulfilling the request will do to the repository."""

    WRITE = "write"
    EDIT = "edit"
    ANSWER = "answer"


OPERATION_CRITERIA: dict[str, str] = {
    "write": (
        "The deliverable is a file that does not exist yet: the user wants "
        "something new created. Contrast with edit: there, the file already "
        "exists and its contents are being changed."
    ),
    "edit": (
        "The deliverable is a change to one or more files that already exist. "
        "No new file is needed."
    ),
    "answer": (
        "The request is served by reading and explaining. No file is created, "
        "changed, or deleted."
    ),
}


COMPLEXITY_LEVELS: tuple[str, ...] = (
    "One trivial, mechanical, or obvious change with no design decision.",
    "A small change contained in one area, with no design decision beyond the obvious.",
    "A moderate change touching several call sites, or requiring one design choice.",
    "A large change across the repository, or several interacting design decisions.",
)


class ConfidenceBand(StrEnum):
    """How much to trust a classification, per docs.typesafe.ai/confidence."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


CONFIDENCE_HIGH = 0.75
"""At or above: route automatically."""

CONFIDENCE_FLOOR = 0.5
"""Below: do not route on the classification; ask the user to clarify."""


def band(confidence: float) -> ConfidenceBand:
    """Map a Choice/Score confidence to a routing band."""
    if confidence >= CONFIDENCE_HIGH:
        return ConfidenceBand.HIGH
    if confidence >= CONFIDENCE_FLOOR:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW
