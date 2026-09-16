"""Integration tests for xg_project.session."""

import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from xg_project.session import (
    append,
    configure,
    create,
    file_context,
    load,
    messages,
    remove,
    stream,
)


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    """Use a temporary database for each test."""
    db_path = tmp_path / "test.db"
    configure(db_path)
    yield
    configure(None)


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
def test_create_stores_in_db(session_dir):
    """create() stores the session in the database."""
    path = create(session_dir)
    # Verify the session is queryable
    loaded = load(session_dir)
    assert loaded == path


@pytest.mark.integration
def test_create_has_no_messages(session_dir):
    """create() creates a session with no messages."""
    path = create(session_dir)
    assert list(messages(path)) == []


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
    """load() returns the same path that create() made."""
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
def test_load_file_has_no_messages(session_dir):
    """load() returns a session with no messages."""
    create(session_dir)
    path = load(session_dir)
    assert list(messages(path)) == []


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


# --- append() tests ---


@pytest.mark.integration
def test_append_adds_message(session_dir):
    """append() adds a message to the session."""
    path = create(session_dir)
    msg = HumanMessage(content="hello")
    append(path, msg)
    result = list(messages(path))
    assert len(result) == 1


@pytest.mark.integration
def test_append_multiple_messages(session_dir):
    """append() can add multiple messages."""
    path = create(session_dir)
    append(path, HumanMessage(content="hello"))
    append(path, AIMessage(content="hi there"))
    result = list(messages(path))
    assert len(result) == 2


# --- messages() tests ---


@pytest.mark.integration
def test_messages_returns_iterator(session_dir):
    """messages() returns an iterator of BaseMessage objects."""
    path = create(session_dir)
    append(path, HumanMessage(content="hello"))
    append(path, AIMessage(content="hi there"))
    result = list(messages(path))
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], AIMessage)


@pytest.mark.integration
def test_messages_preserves_content(session_dir):
    """messages() preserves message content."""
    path = create(session_dir)
    append(path, HumanMessage(content="test content"))
    result = list(messages(path))
    assert result[0].content == "test content"


@pytest.mark.integration
def test_messages_empty_session(session_dir):
    """messages() returns nothing for an empty session."""
    path = create(session_dir)
    result = list(messages(path))
    assert result == []


@pytest.mark.integration
def test_messages_does_not_mutate_data(session_dir):
    """messages() should not mutate the underlying data."""
    path = create(session_dir)
    append(path, HumanMessage(content="test"))
    first = list(messages(path))
    second = list(messages(path))
    assert len(first) == len(second)
    assert first[0].content == second[0].content


# --- remove() tests ---


@pytest.mark.integration
def test_remove_by_id(session_dir):
    """remove() removes a message by its id."""
    path = create(session_dir)
    msg = HumanMessage(content="to remove", id="msg-to-remove")
    append(path, msg)
    append(path, AIMessage(content="to keep", id="msg-to-keep"))
    remove(path, msg.id)
    result = list(messages(path))
    assert len(result) == 1
    assert result[0].content == "to keep"


@pytest.mark.integration
def test_remove_nonexistent_id(session_dir):
    """remove() with nonexistent id leaves session unchanged."""
    path = create(session_dir)
    append(path, HumanMessage(content="keep me"))
    remove(path, "nonexistent-id")
    result = list(messages(path))
    assert len(result) == 1


# --- stream() tests ---


@pytest.mark.integration
def test_stream_returns_bytes(session_dir):
    """stream() yields bytes."""
    path = create(session_dir)
    append(path, HumanMessage(content="hello"))
    result = list(stream(path))
    assert len(result) == 1
    assert isinstance(result[0], bytes)


@pytest.mark.integration
def test_stream_contains_message(session_dir):
    """stream() yields the serialized message."""
    path = create(session_dir)
    append(path, HumanMessage(content="stream me"))
    result = list(stream(path))
    assert b"stream me" in result[0]


# --- file_context() tests ---


@pytest.mark.integration
def test_file_context_returns_list():
    """file_context() returns a list."""
    result = file_context(Path("."))
    assert isinstance(result, list)


@pytest.mark.integration
def test_file_context_contains_messages():
    """file_context() returns BaseMessage objects."""
    from langchain_core.messages import BaseMessage

    result = file_context(Path("."))
    for msg in result:
        assert isinstance(msg, BaseMessage)


@pytest.mark.integration
def test_file_context_has_tool_calls():
    """file_context() has AIMessage with tool calls."""
    result = file_context(Path("."))
    ai_msgs = [m for m in result if isinstance(m, AIMessage)]
    assert len(ai_msgs) > 0
    assert ai_msgs[0].tool_calls


@pytest.mark.integration
def test_file_context_uses_session_if_exists(tmp_path):
    """file_context() loads from session if messages exist."""
    from xg_project.config import Config

    session_dir = tmp_path / "sessions"
    path = create(session_dir)
    append(path, HumanMessage(content="existing"))

    config = Config(session_path=path)
    result = file_context(tmp_path, config)
    assert len(result) == 1
    assert result[0].content == "existing"


# --- configure() tests ---


@pytest.mark.integration
def test_configure_resets_to_default(tmp_path):
    """configure(None) resets to default path."""
    custom = tmp_path / "custom.db"
    configure(custom)
    configure(None)
    from xg_project.session import DEFAULT_DB_PATH
    from xg_project.session import _db_path as current_path
    assert current_path == DEFAULT_DB_PATH


# --- directory isolation tests ---


@pytest.mark.integration
def test_sessions_are_isolated_by_directory(tmp_path):
    """load() returns the most recent session per directory, not globally."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    path_a = create(dir_a)
    time.sleep(0.1)
    path_b = create(dir_b)

    loaded_a = load(dir_a)
    loaded_b = load(dir_b)
    assert loaded_a == path_a
    assert loaded_b == path_b
