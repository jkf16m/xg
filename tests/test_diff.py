"""A proposed change as a patch, and what a proposal says about its own change.

The patch is checked against git itself in two ways: the ``index`` line against
``git hash-object``, and the whole patch against ``git apply``. Those are what
make it a patch rather than something patch-shaped, and they are the two things
one can only find out by asking git.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from xg_project.diff import unified_patch
from xg_project.llm import AddProposal, CommandProposal, EditProposal
from xg_project.turn import Gate

NEW_FILE = 'def greet():\n    return "hi"\n'
"""The content of a file that is not there yet, for the creation patches."""

BEFORE = (
    "def greet(name):\n"
    '    print("hello " + name)\n'
    "    return None\n"
    "\n"
    "\n"
    "def farewell(name):\n"
    '    print("bye " + name)\n'
)
AFTER = BEFORE.replace('"hello " + name', 'f"hello {name}"')

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def test_the_patch_opens_the_way_git_opens_one() -> None:
    lines = unified_patch(path="greet.py", before=BEFORE, after=AFTER).splitlines()
    assert lines[0] == "diff --git a/greet.py b/greet.py"
    assert lines[1].startswith("index ") and lines[1].endswith(" 100644")
    assert lines[2] == "--- a/greet.py"
    assert lines[3] == "+++ b/greet.py"


def test_the_hunk_carries_the_change_and_the_lines_around_it() -> None:
    patch = unified_patch(path="greet.py", before=BEFORE, after=AFTER)
    lines = patch.splitlines()
    assert any(line.startswith("@@ ") for line in lines)
    assert '-    print("hello " + name)' in lines
    assert '+    print(f"hello {name}")' in lines
    # A change is read against what stayed the same, so the context is the point.
    assert " def greet(name):" in lines
    assert "     return None" in lines


def test_the_patch_ends_with_a_newline_because_git_requires_one() -> None:
    assert unified_patch(path="a.py", before="x\n", after="y\n").endswith("\n")


def test_content_that_did_not_change_has_no_patch() -> None:
    assert unified_patch(path="a.py", before="x\n", after="x\n") == ""


def test_context_can_be_widened_or_narrowed() -> None:
    patch = unified_patch(path="a.py", before="1\n2\n3\n4\n5\n", after="1\n2\nX\n4\n5\n", context=0)
    assert "@@ -3 +3 @@" in patch.splitlines()


@needs_git
def test_the_index_line_is_the_hash_git_would_have_written(tmp_path: Path) -> None:
    (tmp_path / "greet.py").write_text(BEFORE, encoding="utf-8")
    expected = subprocess.run(
        ["git", "hash-object", "greet.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    patch = unified_patch(path="greet.py", before=BEFORE, after=AFTER)
    index = next(line for line in patch.splitlines() if line.startswith("index "))

    assert index.split()[1].split("..")[0] == expected[:7]
    assert all(character in "0123456789abcdef" for character in expected[:7])


@needs_git
def test_git_apply_accepts_the_patch_and_produces_the_new_content(tmp_path: Path) -> None:
    """The property that separates a patch from a picture of one."""
    (tmp_path / "greet.py").write_text(BEFORE, encoding="utf-8")
    (tmp_path / "change.patch").write_text(
        unified_patch(path="greet.py", before=BEFORE, after=AFTER), encoding="utf-8"
    )

    result = subprocess.run(
        ["git", "apply", "-p1", "change.patch"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "greet.py").read_text(encoding="utf-8") == AFTER


# -- creating a file -------------------------------------------------------


def test_a_creation_patch_is_marked_as_a_new_file() -> None:
    """Git wants a different header for a creation, so that is what is written."""
    lines = unified_patch(path="pkg/new.py", before="", after=NEW_FILE, new_file=True).splitlines()

    assert lines[0] == "diff --git a/pkg/new.py b/pkg/new.py"
    assert lines[1] == "new file mode 100644"
    assert lines[2].startswith("index 0000000..")
    assert "--- /dev/null" in lines
    assert "+++ b/pkg/new.py" in lines
    assert "@@ -0,0 +1,2 @@" in lines
    assert "+def greet():" in lines


def test_an_empty_new_file_has_no_patch() -> None:
    """No hunks, so nothing to show; the proposal's own line says it is empty."""
    assert unified_patch(path="pkg/__init__.py", before="", after="", new_file=True) == ""


def test_a_creation_patch_ends_with_a_newline_like_any_other() -> None:
    patch = unified_patch(path="pkg/new.py", before="", after=NEW_FILE, new_file=True)
    assert patch.endswith("\n")
    assert not patch.endswith("\n\n")


@needs_git
def test_the_creation_index_hash_is_the_hash_git_gives_the_content(tmp_path: Path) -> None:
    expected = subprocess.run(
        ["git", "hash-object", "--stdin"],
        input=NEW_FILE,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    patch = unified_patch(path="pkg/new.py", before="", after=NEW_FILE, new_file=True)
    index = next(line for line in patch.splitlines() if line.startswith("index "))

    # The from-side is all zeros because there is no file to have hashed.
    assert index.split()[1] == f"{'0' * 7}..{expected[:7]}"


@needs_git
def test_git_apply_creates_the_file_from_a_creation_patch(tmp_path: Path) -> None:
    """A creation is a different patch, and git has to accept it as one."""
    (tmp_path / "change.patch").write_text(
        unified_patch(path="pkg/new.py", before="", after=NEW_FILE, new_file=True),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["git", "apply", "-p1", "change.patch"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "pkg/new.py").read_text(encoding="utf-8") == NEW_FILE


# -- what a proposal says about itself -------------------------------------


def proposal(**overrides) -> EditProposal:
    base = {
        "path": "a.py",
        "content": "x = 1\ny = 2\n",
        "old_text": "x = 1",
        "new_text": "x = 10",
        "model": "deepseek/something",
    }
    return EditProposal(**{**base, **overrides})


def test_an_edit_proposal_shows_its_change_as_a_patch() -> None:
    patch = proposal().detail()
    assert patch.startswith("diff --git a/a.py b/a.py")
    assert "-x = 1" in patch
    assert "+x = 10" in patch
    assert " y = 2" in patch


def test_a_proposal_with_nothing_to_diff_against_shows_no_patch() -> None:
    """A proposal built without the content it was shown has no context to draw."""
    assert proposal(content=None).detail() == ""


def test_an_old_text_that_is_not_in_the_file_shows_no_patch() -> None:
    """`apply_edit` will refuse it, so drawing a change would be a promise it breaks."""
    assert proposal(old_text="nowhere in the file").detail() == ""


def test_an_ambiguous_old_text_shows_no_patch() -> None:
    """Two matches is the other refusal `apply_edit` makes."""
    assert proposal(content="x = 1\nx = 1\n").detail() == ""


def test_a_proposal_that_changes_nothing_shows_no_patch() -> None:
    assert proposal(new_text="x = 1").detail() == ""


def test_a_command_proposal_has_nothing_more_to_say() -> None:
    assert CommandProposal(command="ls").detail() == ""


def test_the_one_line_preview_never_carries_the_patch() -> None:
    """The preview is also what Jev is shown when it routes: one line, always."""
    assert "\n" not in proposal().preview()
    assert "\n" not in Gate(proposal=proposal()).preview()


def test_the_gate_shows_what_the_proposal_would_change() -> None:
    gate = Gate(proposal=proposal())
    assert gate.preview().startswith("pending: edit a.py")
    assert "diff --git a/a.py b/a.py" in gate.detail()


def test_the_gate_shows_the_file_a_creation_would_add() -> None:
    """The gate delegates to the proposal, so it renders a creation too."""
    gate = Gate(proposal=AddProposal(path="pkg/new.py", content=NEW_FILE))

    assert gate.preview() == "pending: add pkg/new.py: 2 lines"
    assert "new file mode 100644" in gate.detail()
    assert "+def greet():" in gate.detail()
