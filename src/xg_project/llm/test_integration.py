"""Integration tests for xg_project.llm."""


import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from xg_project.config import Config
from xg_project.llm import (
    TOOLS,
    add_message,
    execute_tool,
    tool_result,
)

# --- add_message() tests ---


@pytest.mark.integration
def test_add_message_returns_list():
    """add_message() returns a list."""
    result = add_message([], "hello")
    assert isinstance(result, list)


@pytest.mark.integration
def test_add_message_appends_human():
    """add_message() appends a HumanMessage."""
    result = add_message([], "hello")
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "hello"


@pytest.mark.integration
def test_add_message_preserves_existing():
    """add_message() preserves existing messages."""
    existing = [HumanMessage(content="first")]
    result = add_message(existing, "second")
    assert len(result) == 2
    assert result[0].content == "first"
    assert result[1].content == "second"


@pytest.mark.integration
def test_add_message_does_not_mutate():
    """add_message() does not mutate the original list."""
    original = [HumanMessage(content="first")]
    _ = add_message(original, "second")
    assert len(original) == 1


# --- execute_tool() tests ---


@pytest.mark.integration
def test_execute_tool_returns_tool_message():
    """execute_tool() returns a ToolMessage."""
    tc = {"name": "read_file", "args": {"path": "pyproject.toml"}, "id": "test-1"}
    result = execute_tool(tc)
    assert isinstance(result, ToolMessage)


@pytest.mark.integration
def test_execute_tool_preserves_id():
    """execute_tool() preserves the tool_call_id."""
    tc = {"name": "read_file", "args": {"path": "pyproject.toml"}, "id": "my-id"}
    result = execute_tool(tc)
    assert result.tool_call_id == "my-id"


@pytest.mark.integration
def test_execute_tool_preserves_name():
    """execute_tool() preserves the tool name."""
    tc = {"name": "read_file", "args": {"path": "pyproject.toml"}, "id": "test-1"}
    result = execute_tool(tc)
    assert result.name == "read_file"


@pytest.mark.integration
def test_execute_tool_error_returns_error_status():
    """execute_tool() returns error status on failure."""
    tc = {"name": "read_file", "args": {"path": "/nonexistent"}, "id": "test-err"}
    result = execute_tool(tc)
    assert result.status == "error"


# --- tool_result() tests ---


@pytest.mark.integration
def test_tool_result_returns_list():
    """tool_result() returns a list."""
    tm = ToolMessage(content="result", tool_call_id="t1")
    result = tool_result([], tm)
    assert isinstance(result, list)


@pytest.mark.integration
def test_tool_result_appends_message():
    """tool_result() appends the ToolMessage."""
    tm = ToolMessage(content="result", tool_call_id="t1")
    result = tool_result([], tm)
    assert len(result) == 1
    assert isinstance(result[0], ToolMessage)


@pytest.mark.integration
def test_tool_result_preserves_existing():
    """tool_result() preserves existing messages."""
    existing = [HumanMessage(content="question")]
    tm = ToolMessage(content="result", tool_call_id="t1")
    result = tool_result(existing, tm)
    assert len(result) == 2


# --- TOOLS tests ---


@pytest.mark.integration
def test_tools_is_list():
    """TOOLS is a list."""
    assert isinstance(TOOLS, list)


@pytest.mark.integration
def test_tools_have_names():
    """TOOLS have name attributes."""
    for tool in TOOLS:
        assert hasattr(tool, "name")


# --- Config tests ---


@pytest.mark.integration
def test_config_default_session_path():
    """Config() has None session_path by default."""
    config = Config()
    assert config.session_path is None


@pytest.mark.integration
def test_config_with_session_path(tmp_path):
    """Config() accepts session_path."""
    session_path = tmp_path / "session.jsonl"
    config = Config(session_path=session_path)
    assert config.session_path == session_path
