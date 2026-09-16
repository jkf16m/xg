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
    configure,
    create,
    file_context,
    load,
    messages,
)


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    """Use a temporary database for each test."""
    db_path = tmp_path / "test.db"
    configure(db_path)
    yield
    configure(None)


@pytest.mark.integration
def test_full_workflow(tmp_path):
    """Test the full workflow: config -> session -> llm."""
    # 1. Setup session
    session_dir = tmp_path / "sessions"
    path = create(session_dir)

    # 2. Create config with session path
    config = Config(session_path=path)

    # 3. Get file context (creates initial messages)
    # Use the project root which has files
    project_dir = Path(__file__).parent.parent.parent
    msgs = file_context(project_dir, config)
    assert len(msgs) > 0

    # 4. Persist file context to session
    for msg in msgs:
        append(path, msg)

    # 5. User appends a message to session
    human_msg = HumanMessage(content="What is this project about?")
    append(path, human_msg)

    # 6. Reload messages from session
    msgs = list(messages(path))
    assert len(msgs) > 1
    assert msgs[-1].content == "What is this project about?"


@pytest.mark.integration
def test_session_persistence(tmp_path):
    """Test that session persists across loads."""
    session_dir = tmp_path / "sessions"

    # Create session and add messages
    path = create(session_dir)
    append(path, HumanMessage(content="hello", id="msg-1"))
    append(path, AIMessage(content="world", id="msg-2"))

    # Load and verify
    loaded_path = load(session_dir)
    assert loaded_path == path

    msgs = list(messages(path))
    assert len(msgs) == 2
    assert msgs[0].content == "hello"
    assert msgs[1].content == "world"


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
def test_file_context_creates_fresh_when_empty(tmp_path):
    """Test that file_context creates fresh messages when session is empty."""
    session_dir = tmp_path / "sessions"
    path = create(session_dir)

    config = Config(session_path=path)

    # Create a dummy file in tmp_path
    (tmp_path / "test.txt").write_text("hello")

    msgs = file_context(tmp_path, config)
    assert len(msgs) > 0


@pytest.mark.integration
def test_config_without_session():
    """Test that functions work without session config."""
    config = Config()
    assert config.session_path is None

    # add_message works independently
    msgs = [HumanMessage(content="start")]
    msgs = add_message(msgs, "hello")
    assert len(msgs) == 2
    assert msgs[-1].content == "hello"


@pytest.mark.integration
def test_tool_execution_flow():
    """Test complete tool execution flow."""
    # Execute a tool
    tc = {"name": "read", "args": {"path": "pyproject.toml"}, "id": "call-1"}
    result = execute_tool(tc)

    # Verify result
    assert result.name == "read"
    assert result.tool_call_id == "call-1"
    assert "xg-project" in result.content

    # Add to conversation
    msgs = [HumanMessage(content="read the config")]
    msgs = tool_result(msgs, result)
    assert len(msgs) == 2


@pytest.mark.integration
def test_message_types_flow():
    """Test that different message types work together."""
    from langchain_core.messages import ToolMessage

    msgs = []

    # Human asks
    msgs = add_message(msgs, "read the config")
    assert isinstance(msgs[0], HumanMessage)

    # AI responds with tool call
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "read",
            "args": {"path": "pyproject.toml"},
            "id": "call-1",
            "type": "tool_call",
        }],
    )
    msgs.append(ai_msg)

    # Tool result
    tc = {"name": "read", "args": {"path": "pyproject.toml"}, "id": "call-1"}
    tool_msg = execute_tool(tc)
    msgs = tool_result(msgs, tool_msg)

    # Verify sequence
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert isinstance(msgs[2], ToolMessage)
    assert msgs[1].tool_calls[0]["id"] == msgs[2].tool_call_id
