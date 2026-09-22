"""Applying an edit: the two guards, and how each failure reads.

Everything happens in a tmp_path; nothing here touches a file outside it.
"""

from __future__ import annotations

from pathlib import Path

from xg_project.edit import EditOutcome, apply_edit
from xg_project.llm import EditProposal


def write(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_an_edit_replaces_the_substring_and_writes_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, "a.py", "x = 1\ny = 2\n")
    outcome = apply_edit(
        EditProposal(path="a.py", old_text="x = 1", new_text="x = 42"), root=tmp_path
    )
    assert outcome.ok
    assert outcome.applied
    assert outcome.path == "a.py"
    assert path.read_text(encoding="utf-8") == "x = 42\ny = 2\n"


def test_only_the_first_occurrence_of_a_unique_substring_is_replaced(tmp_path: Path) -> None:
    path = write(tmp_path, "a.py", "keep\nold\nkeep\n")
    apply_edit(EditProposal(path="a.py", old_text="old", new_text="new"), root=tmp_path)
    assert path.read_text(encoding="utf-8") == "keep\nnew\nkeep\n"


def test_an_old_text_that_is_not_there_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    path = write(tmp_path, "a.py", "x = 1\n")
    outcome = apply_edit(
        EditProposal(path="a.py", old_text="not in the file", new_text="y"), root=tmp_path
    )
    assert not outcome.ok
    assert "does not appear" in outcome.problem
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_an_ambiguous_old_text_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    """Replacing the first of two matches could be the wrong one, so neither is."""
    path = write(tmp_path, "a.py", "dup\ndup\n")
    outcome = apply_edit(EditProposal(path="a.py", old_text="dup", new_text="x"), root=tmp_path)
    assert not outcome.ok
    assert "appears 2 times" in outcome.problem
    assert path.read_text(encoding="utf-8") == "dup\ndup\n"


def test_a_missing_file_is_a_problem_not_a_raise(tmp_path: Path) -> None:
    outcome = apply_edit(EditProposal(path="nope.py", old_text="x", new_text="y"), root=tmp_path)
    assert not outcome.ok
    assert "could not read" in outcome.problem


def test_a_path_is_joined_to_the_root(tmp_path: Path) -> None:
    write(tmp_path, "src/deep/a.py", "x = 1\n")
    outcome = apply_edit(
        EditProposal(path="src/deep/a.py", old_text="x = 1", new_text="x = 2"), root=tmp_path
    )
    assert outcome.ok
    assert (tmp_path / "src/deep/a.py").read_text(encoding="utf-8") == "x = 2\n"


def test_a_proposal_with_no_path_is_refused() -> None:
    outcome = apply_edit(EditProposal(old_text="x", new_text="y"))
    assert not outcome.ok
    assert "no file" in outcome.problem


def test_a_proposal_with_no_old_text_is_refused() -> None:
    outcome = apply_edit(EditProposal(path="a.py", new_text="y"))
    assert not outcome.ok
    assert "no old_text" in outcome.problem


def test_a_proposal_with_no_new_text_is_refused(tmp_path: Path) -> None:
    write(tmp_path, "a.py", "x = 1\n")
    outcome = apply_edit(EditProposal(path="a.py", old_text="x = 1"), root=tmp_path)
    assert not outcome.ok
    assert "no new_text" in outcome.problem


def test_explain_reads_the_problem_when_there_is_one() -> None:
    assert EditOutcome(path="a.py", applied=True).explain() == "applied to a.py"
    assert EditOutcome(path="a.py", problem="boom").explain() == "boom"
