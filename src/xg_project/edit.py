"""Applying an edit proposal to the file it names.

The edit's shape is the point: a literal ``old_text`` and the ``new_text`` that
replaces it. That keeps the change reviewable — the user sees exactly which
substring of which file is swapped — and it makes each failure a sentence
instead of a corrupted file.

Two rules are enforced here rather than trusted to the model:

- ``old_text`` must appear in the file at all, or there is nothing to replace.
- it must appear exactly *once*. If it appears twice, the model's intent is
  ambiguous and replacing the first occurrence could be the wrong one; refusing
  is safer than guessing.

:func:`apply_add` is the other half and the mirror of those rules: it creates a
file, and its one rule is that the path must not already exist. An edit can lose
information by replacing the wrong text; an add can lose a whole file, so the
check that cannot be made in advance — the file was not there when the proposal
was written — is made at the moment of writing instead.

Failure is a value, like everywhere else: a missing file, an ambiguous match, and
a write that fails all come back as an :class:`EditOutcome` with ``problem`` set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xg_project.llm import AddProposal, EditProposal


@dataclass(frozen=True)
class EditOutcome:
    """What applying an edit did, or why it could not be applied."""

    path: str
    applied: bool = False
    problem: str | None = None
    created: bool = False
    """Whether the file was created rather than rewritten.

    The two read differently in a status line — "created a.py" against "applied
    to a.py" — and they are not the same act, so the outcome says which one it
    was rather than leaving the caller to infer it from the proposal's type.
    """

    @property
    def ok(self) -> bool:
        """Whether the file was written."""
        return self.problem is None and self.applied

    def explain(self) -> str:
        """A one-line account, for the status line."""
        if self.problem is not None:
            return self.problem
        return f"created {self.path}" if self.created else f"applied to {self.path}"


def apply_edit(proposal: EditProposal, *, root: str | Path | None = None) -> EditOutcome:
    """Replace ``proposal.old_text`` with ``proposal.new_text`` in its file.

    ``root`` is the directory the path is relative to; the bare path is used when
    it is ``None``, which is what an absolute path in a proposal would need.
    """
    path_value = proposal.path
    if not path_value:
        return EditOutcome(path="", problem="the proposal named no file")
    old_text = proposal.old_text
    new_text = proposal.new_text
    if not old_text:
        return EditOutcome(path=path_value, problem="the proposal had no old_text")
    if new_text is None:
        return EditOutcome(path=path_value, problem="the proposal had no new_text")

    path = Path(root) / path_value if root is not None else Path(path_value)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        return EditOutcome(path=path_value, problem=f"could not read {path_value}: {error}")

    occurrences = content.count(old_text)
    if occurrences == 0:
        return EditOutcome(
            path=path_value,
            problem=f"old_text does not appear in {path_value}; nothing was changed",
        )
    if occurrences > 1:
        return EditOutcome(
            path=path_value,
            problem=(
                f"old_text appears {occurrences} times in {path_value}; "
                "ambiguous, so nothing was changed"
            ),
        )

    try:
        path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    except OSError as error:
        return EditOutcome(path=path_value, problem=f"could not write {path_value}: {error}")
    return EditOutcome(path=path_value, applied=True)


def apply_add(proposal: AddProposal, *, root: str | Path | None = None) -> EditOutcome:
    """Create the new file ``proposal`` describes.

    The path must not already exist, and that is the whole rule. Every other
    check an edit makes is about matching text that is already on disk; an add has
    no such text, so the only thing that can go wrong is aiming at a file somebody
    already has — and a proposal written minutes ago cannot know whether that is
    still true. The check therefore happens here, immediately before the write,
    rather than when the preview was drawn.

    Parent directories are created, because a path with a directory in it is a
    normal thing to propose and refusing one would make a nested new file
    impossible to add. ``root`` is the directory the path is relative to; the bare
    path is used when it is ``None``.
    """
    path_value = proposal.path
    if not path_value:
        return EditOutcome(path="", problem="the proposal named no file")
    content = proposal.content
    if content is None:
        return EditOutcome(path=path_value, problem="the proposal had no content")

    path = Path(root) / path_value if root is not None else Path(path_value)
    if path.exists():
        return EditOutcome(
            path=path_value,
            problem=f"{path_value} already exists; nothing was written",
        )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        return EditOutcome(path=path_value, problem=f"could not write {path_value}: {error}")
    return EditOutcome(path=path_value, applied=True, created=True)
