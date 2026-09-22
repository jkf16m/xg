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

Failure is a value, like everywhere else: a missing file, an ambiguous match, and
a write that fails all come back as an :class:`EditOutcome` with ``problem`` set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xg_project.llm import EditProposal


@dataclass(frozen=True)
class EditOutcome:
    """What applying an edit did, or why it could not be applied."""

    path: str
    applied: bool = False
    problem: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the file was rewritten."""
        return self.problem is None and self.applied

    def explain(self) -> str:
        """A one-line account, for the status line."""
        if self.problem is not None:
            return self.problem
        return f"applied to {self.path}"


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
