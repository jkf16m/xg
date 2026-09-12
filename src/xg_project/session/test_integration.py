"""Integration tests for xg_project.session."""

import time
from pathlib import Path

import pytest

from xg_project.session import LAST_FILE, create, load


@pytest.fixture
def session_dir(tmp_path):
    """Provide a temporary directory for session tests."""
    return tmp_path / "sessions"


# --- create() tests ---


@pytest.mark.integration
def test_create_returns_path(session_dir):
    """create() returns a Path."""
    path = create(session_dir)
    assert isinstance(path, Path)


@pytest.mark.integration
def test_create_file_has_jsonl_extension(session_dir):
    """create() returns a path ending with .jsonl."""
    path = create(session_dir)
    assert path.suffix == ".jsonl"


@pytest.mark.integration
def test_create_file_name_is_timestamp(session_dir):
    """create() names the file with a UNIX timestamp."""
    before = int(time.time())
    path = create(session_dir)
    after = int(time.time())
    name = int(path.stem)
    assert before <= name <= after


@pytest.mark.integration
def test_create_makes_directory(session_dir):
    """create() creates the directory if it doesn't exist."""
    create(session_dir)
    assert session_dir.exists()


@pytest.mark.integration
def test_create_makes_file(session_dir):
    """create() creates the session file."""
    path = create(session_dir)
    assert path.exists()


@pytest.mark.integration
def test_create_file_is_empty(session_dir):
    """create() returns an empty file."""
    path = create(session_dir)
    assert path.read_text() == ""


@pytest.mark.integration
def test_create_makes_last_pointer(session_dir):
    """create() creates a .last pointer file."""
    create(session_dir)
    last = session_dir / LAST_FILE
    assert last.exists()


@pytest.mark.integration
def test_create_last_points_to_new_file(session_dir):
    """create() makes .last point to the new session file."""
    path = create(session_dir)
    last = session_dir / LAST_FILE
    assert last.read_text() == path.name


@pytest.mark.integration
def test_create_last_overwrites_on_new_session(session_dir):
    """create() updates .last when creating a new session."""
    first = create(session_dir)
    time.sleep(1.1)
    second = create(session_dir)
    last = session_dir / LAST_FILE
    assert last.read_text() == second.name
    assert first.name != second.name


@pytest.mark.integration
def test_create_requires_path():
    """create() raises TypeError if not given a Path."""
    with pytest.raises(TypeError):
        create("/not/a/path")  # type: ignore[arg-type]


# --- load() tests ---


@pytest.mark.integration
def test_load_returns_path(session_dir):
    """load() returns a Path."""
    create(session_dir)
    path = load(session_dir)
    assert isinstance(path, Path)


@pytest.mark.integration
def test_load_returns_same_file_as_create(session_dir):
    """load() returns the same file that create() made."""
    created = create(session_dir)
    loaded = load(session_dir)
    assert created == loaded


@pytest.mark.integration
def test_load_returns_latest_session(session_dir):
    """load() returns the most recent session after multiple creates."""
    first = create(session_dir)
    time.sleep(1.1)
    second = create(session_dir)
    loaded = load(session_dir)
    assert loaded == second
    assert loaded != first


@pytest.mark.integration
def test_load_file_exists(session_dir):
    """load() returns a path that exists on disk."""
    create(session_dir)
    path = load(session_dir)
    assert path.exists()


@pytest.mark.integration
def test_load_raises_if_no_session(session_dir):
    """load() raises FileNotFoundError if no session exists."""
    with pytest.raises(FileNotFoundError):
        load(session_dir)


@pytest.mark.integration
def test_load_requires_path():
    """load() raises TypeError if not given a Path."""
    with pytest.raises(TypeError):
        load("/not/a/path")  # type: ignore[arg-type]
