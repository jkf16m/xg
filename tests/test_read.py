"""The deterministic read: what is read, what is ignored, and what is skipped."""

from __future__ import annotations

from pathlib import Path

from xg_project.read import read_tree


def write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_every_readable_file_below_the_root_is_read(tmp_path: Path) -> None:
    write(tmp_path, "a.py", "a")
    write(tmp_path, "pkg/b.py", "b")
    tree = read_tree(tmp_path)
    assert tree.files == {"a.py": "a", "pkg/b.py": "b"}


def test_the_mapping_is_ordered_by_path(tmp_path: Path) -> None:
    """Determinism: the same tree produces the same order, so a run replays."""
    write(tmp_path, "z.py", "z")
    write(tmp_path, "a.py", "a")
    write(tmp_path, "m/n.py", "n")
    assert list(read_tree(tmp_path).files) == ["a.py", "m/n.py", "z.py"]


def test_gitignore_excludes_a_root_pattern(tmp_path: Path) -> None:
    write(tmp_path, ".gitignore", "*.log\n")
    write(tmp_path, "keep.py", "k")
    write(tmp_path, "noise.log", "n")
    assert set(read_tree(tmp_path).files) == {"keep.py"}


def test_gitignore_excludes_a_directory(tmp_path: Path) -> None:
    write(tmp_path, ".gitignore", "__pycache__/\n")
    write(tmp_path, "keep.py", "k")
    write(tmp_path, "__pycache__/x.pyc", "x")
    assert set(read_tree(tmp_path).files) == {"keep.py"}


def test_xgignore_wins_over_gitignore(tmp_path: Path) -> None:
    """The more specific file has the final say, so a negation there re-includes."""
    write(tmp_path, ".gitignore", "*.txt\n")
    write(tmp_path, ".xgignore", "!notes.txt\n")
    write(tmp_path, "notes.txt", "keep me")
    write(tmp_path, "other.txt", "drop me")
    assert set(read_tree(tmp_path).files) == {"notes.txt"}


def test_a_nested_gitignore_applies_to_its_own_directory(tmp_path: Path) -> None:
    """A pattern without a slash means any depth below the file that declares it."""
    write(tmp_path, ".gitignore", "# root has no rules\n")
    write(tmp_path, "pkg/.gitignore", "*.tmp\n")
    write(tmp_path, "pkg/keep.py", "k")
    write(tmp_path, "pkg/drop.tmp", "d")
    write(tmp_path, "other.tmp", "not covered by pkg/.gitignore")
    tree = read_tree(tmp_path)
    assert "pkg/drop.tmp" not in tree.files
    assert "other.tmp" in tree.files
    assert tree.files["pkg/keep.py"] == "k"


def test_a_nested_anchored_pattern_is_anchored_to_its_directory(tmp_path: Path) -> None:
    write(tmp_path, "pkg/.gitignore", "/local.py\n")
    write(tmp_path, "pkg/local.py", "anchored here")
    write(tmp_path, "pkg/deep/local.py", "not anchored here")
    tree = read_tree(tmp_path)
    assert "pkg/local.py" not in tree.files
    assert "pkg/deep/local.py" in tree.files


def test_the_git_directory_and_the_ignore_files_are_never_read(tmp_path: Path) -> None:
    write(tmp_path, ".gitignore", "*.log\n")
    write(tmp_path, ".xgignore", "*.bak\n")
    write(tmp_path, ".git/HEAD", "ref: refs/heads/main\n")
    # A vendored checkout or submodule carries its own .git directory. It is
    # git's storage rather than project content, the same as the root one, and it
    # is skipped wherever it appears rather than only at the root.
    write(tmp_path, "vendor/.git/HEAD", "ref: refs/heads/main\n")
    write(tmp_path, "vendor/lib.py", "l")
    write(tmp_path, "keep.py", "k")
    assert set(read_tree(tmp_path).files) == {"keep.py", "vendor/lib.py"}


def test_a_binary_file_is_skipped_and_named(tmp_path: Path) -> None:
    write(tmp_path, "keep.py", "k")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")
    tree = read_tree(tmp_path)
    assert set(tree.files) == {"keep.py"}
    assert any("blob.bin" in entry and "not UTF-8" in entry for entry in tree.skipped)


def test_an_oversize_file_is_skipped_and_its_size_reported(tmp_path: Path) -> None:
    write(tmp_path, "big.txt", "x" * 100)
    write(tmp_path, "small.txt", "ok")
    tree = read_tree(tmp_path, max_bytes=10)
    assert set(tree.files) == {"small.txt"}
    assert any("big.txt" in entry and "over 10" in entry for entry in tree.skipped)


def test_a_clean_tree_skips_nothing(tmp_path: Path) -> None:
    write(tmp_path, "a.py", "a")
    assert read_tree(tmp_path).skipped == ()


def test_content_is_the_file_itself_not_a_wrapper(tmp_path: Path) -> None:
    """The value is the direct content: no header, no line numbers, no JSON."""
    write(tmp_path, "a.py", "line one\nline two\n")
    assert read_tree(tmp_path).files["a.py"] == "line one\nline two\n"


def test_this_projects_own_venv_and_caches_are_not_read() -> None:
    """The reader against the repository it lives in.

    .venv holds thousands of files and every __pycache__ holds a copy of a
    source file; if the project's own .gitignore stopped applying, the context
    would be mostly vendor code. This is the test that notices.
    """
    root = Path(__file__).resolve().parents[1]
    tree = read_tree(root)
    assert tree.files, "the repository should have readable files"
    for path in tree.files:
        assert not path.startswith(".venv/"), path
        assert "__pycache__" not in path, path
        assert not path.startswith(".pytest_cache/"), path
        assert not path.startswith(".git/"), path
        assert not path.endswith((".pyc", ".pyo")), path
    assert not any(".venv" in entry for entry in tree.skipped)
