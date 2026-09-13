"""Global integration test for xg_project."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from xg_project.config import Config
from xg_project.llm import (
    add_message,
    execute_tool,
    tool_result,
)
from xg_project.session import (
    append,
    create,
    file_context,
    load,
    messages,
)


@pytest.mark.integration
def test_full_workflow(tmp_path):
    """Test the full workflow: config -> session -> llm."""
    # 1. Setup session
    session_dir = tmp_path / "sessions"
    path = create(session_dir)

    # 2. Create config with session path
    config = Config(session_path=path)

    # 3. Get file context (creates initial messages)
    project_dir = Path(".")
    msgs = file_context(project_dir, config)
    assert len(msgs) > 0

    # 4. User appends a message to session
    human_msg = HumanMessage(content="What is this project about?")
    append(path, human_msg)

    # 5. Reload messages from session
    msgs = list(messages(path))
    assert len(msgs) > 1

    # 6. Add another message (without persisting)
    msgs = add_message(msgs, "Tell me more")

    # 7. Simulate tool execution
    tc = {"name": "read_file", "args": {"path": "pyproject.toml"}, "id": "call-1"}
    tool_msg = execute_tool(tc)
    msgs = tool_result(msgs, tool_msg)

    # 8. Verify final state
    assert isinstance(msgs[-1], HumanMessage)
    assert msgs[-1].content == "Tell me more"


@pytest.mark.integration
def test_session_persistence(tmp_path):
    """Test that session persists across loads."""
    session_dir = tmp_path / "sessions"

    # First session
    path1 = create(session_dir)
    append(path1, HumanMessage(content="first", id="msg-1"))

    # Second session
    path2 = create(session_dir)
    append(path2, HumanMessage(content="second", id="msg-2"))

    # Load latest
    latest_path = load(session_dir)
    assert latest_path == path2

    msgs = list(messages(latest_path))
    assert len(msgs) == 1
    assert msgs[0].content == "second"


@pytest.mark.integration
def test_file_context_caching(tmp_path):
    """Test that file_context returns cached messages if available."""
    session_dir = tmp_path / "sessions"
    path = create(session_dir)

    # Add some messages
    append(path, HumanMessage(content="cached", id="cached-1"))
    append(path, AIMessage(content="response", id="cached-2"))

    config = Config(session_path=path)
    msgs = file_context(tmp_path, config)

    # Should return cached messages, not create new ones
    assert len(msgs) == 2
    assert msgs[0].content == "cached"
    assert msgs[1].content == "response"


@pytest.mark.integration
def test_config_without_session(tmp_path):
    """Test that functions work without session config."""
    config = Config()

    # file_context without session creates fresh messages
    msgs = file_context(tmp_path, config)
    assert len(msgs) > 0

    # add_message works independently
    msgs = add_message(msgs, "hello")
    assert msgs[-1].content == "hello"


@pytest.mark.integration
def test_tool_execution_flow():
    """Test complete tool execution flow."""
    # Execute a tool
    tc = {"name": "read_file", "args": {"path": "pyproject.toml"}, "id": "call-1"}
    result = execute_tool(tc)

    # Verify result
    assert result.name == "read_file"
    assert result.tool_call_id == "call-1"
    assert "xg-project" in result.content

    # Add to conversation
    msgs = [HumanMessage(content="read the config")]
    msgs = tool_result(msgs, result)
    assert len(msgs) == 2
