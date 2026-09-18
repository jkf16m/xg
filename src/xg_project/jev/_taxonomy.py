"""Request taxonomy for the xg classifier.

The classifier answers one primary ``Choice`` question (``request_kind``) plus
the signals the router needs to pick a path: how wide the change is
(``scope``), how hard it is (``complexity``), whether the repository must be
modified at all (``changes_code``), and whether the request is risky
(``is_destructive``).

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
    COMMAND = "command"
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
    "command": (
        "The user wants a shell command or an operational action performed "
        "(build, install, run, deploy, git operation) rather than a code edit."
    ),
    "other": (
        "The request does not fit any option above, is too vague to place, or "
        "spans several kinds with no clear primary."
    ),
}


class Scope(StrEnum):
    """How much of the repository the request is expected to touch."""

    SINGLE_FILE = "single_file"
    FEW_FILES = "few_files"
    MODULE = "module"
    REPO_WIDE = "repo_wide"
    UNCLEAR = "unclear"


SCOPE_CRITERIA: dict[str, str] = {
    "single_file": "One file, or one file plus its test, needs to change.",
    "few_files": "A handful of files in one area or one package need to change.",
    "module": "A whole module or subsystem needs to change, or several packages.",
    "repo_wide": "The change is repository-wide or spans unrelated subsystems.",
    "unclear": (
        "The request does not say enough to estimate how far-reaching the "
        "change is, or no change is needed."
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
