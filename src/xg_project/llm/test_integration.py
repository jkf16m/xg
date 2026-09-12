"""Integration tests for xg_project.llm."""

import tempfile

import pytest
from langchain_core.messages import BaseMessage

from xg_project.llm import (
    add_message,
    execute_tool,
    initial_file_messages,
    stream_turn,
    tool_result,
)

TASK = "Create a file called /tmp/xg_test.txt containing the word hello."


@pytest.mark.integration
def test_initial_context():
    """Verify initial_file_messages returns synthetic read_file calls."""
    messages = initial_file_messages()
    assert len(messages) > 0, "initial context should not be empty"
    for msg in messages:
        assert msg.type in ("ai", "tool"), f"unexpected message type: {msg.type}"


@pytest.mark.integration
def test_stream_turn():
    """Verify stream_turn streams characters and returns an AIMessage."""
    messages: list[BaseMessage] = []
    characters: list[str] = []

    def on_text(ch):
        characters.append(ch)

    messages = add_message(messages, "Say exactly: integration test ok")
    response, messages = stream_turn(messages, on_text=on_text)

    assert response is not None, "stream_turn returned None"
    assert response.content, "response content should not be empty"
    assert len(characters) > 0, "on_text was never called"


@pytest.mark.integration
def test_tool_execution():
    """Verify execute_tool runs a tool and tool_result appends it."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello from test")
        path = f.name

    tc: dict[str, object] = {
        "name": "read_file",
        "args": {"path": path},
        "id": "test-read-1",
        "type": "tool_call",
    }
    tm = execute_tool(tc)
    assert tm.content == "hello from test", f"unexpected content: {tm.content!r}"

    messages: list[BaseMessage] = [tm]
    messages = tool_result(messages, tm)
    assert len(messages) == 2, f"expected 2 messages, got {len(messages)}"


@pytest.mark.integration
def test_full_loop():
    """End-to-end: user message -> stream -> tool call -> tool result -> summary."""
    messages: list[BaseMessage] = initial_file_messages()
    characters: list[str] = []
    messages = add_message(messages, TASK)

    response, messages = stream_turn(messages, on_text=lambda ch: characters.append(ch))
    assert response is not None

    if response.tool_calls:
        for tc in response.tool_calls:
            tm = execute_tool({"name": tc["name"], "args": tc["args"], "id": tc["id"]})
            messages = tool_result(messages, tm)
        response, messages = stream_turn(messages, on_text=lambda ch: characters.append(ch))

    assert response.content, "final response should have content"
