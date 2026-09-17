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


# --- interface approval helpers ---


@pytest.mark.integration
def test_unique_tool_calls_collapses_exact_duplicates():
    """Repeated streamed tool calls collapse to one, preserving first-seen order."""
    from xg_project.interface import _unique_tool_calls

    build = {"name": "cmd", "args": {"command": "make"}, "id": "a"}
    test = {"name": "cmd", "args": {"command": "pytest"}, "id": "b"}

    result = _unique_tool_calls([
        build,
        dict(build, id="c"),          # same call, different id
        test,
        dict(test, id="d"),
        build,
    ])

    assert [call["id"] for call in result] == ["a", "b"]


@pytest.mark.integration
def test_unique_tool_calls_keeps_distinct_args():
    """Calls that differ only in args are all kept."""
    from xg_project.interface import _unique_tool_calls

    calls = [
        {"name": "cmd", "args": {"command": "make"}, "id": "a"},
        {"name": "cmd", "args": {"command": "pytest"}, "id": "b"},
    ]

    assert len(_unique_tool_calls(calls)) == 2


@pytest.mark.integration
def test_unique_tool_calls_keeps_distinct_names():
    """Calls with the same args but different tools are all kept."""
    from xg_project.interface import _unique_tool_calls

    calls = [
        {"name": "read", "args": {"path": "a.py"}, "id": "a"},
        {"name": "write", "args": {"path": "a.py"}, "id": "b"},
    ]

    assert len(_unique_tool_calls(calls)) == 2


@pytest.mark.integration
def test_tool_label_renders_cmd_and_file():
    """Tool labels use the $ prefix for commands and the tool name otherwise."""
    from xg_project.interface import _tool_label

    assert _tool_label({"name": "cmd", "args": {"command": "ls -la"}}).plain == "$ ls -la"
    assert _tool_label({"name": "read", "args": {"path": "a.py"}}).plain == "read a.py"


@pytest.mark.integration
def test_read_byte_raises_eof(monkeypatch):
    """End of stdin raises EOFError instead of returning an empty key and spinning."""
    import xg_project.interface as interface

    class _FakeStdin:
        def fileno(self) -> int:
            return 0

    monkeypatch.setattr(interface.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(interface.os, "read", lambda _fd, _n: b"")

    with pytest.raises(EOFError):
        interface._read_byte()


# --- approval patch preview ---


@pytest.mark.integration
def test_preview_patch_write_new_file(tmp_path):
    """A write preview shows the whole new file as additions."""
    from xg_project.interface import _preview_patch

    target = tmp_path / "new.py"
    patch = _preview_patch("write", {"path": str(target), "content": "print('hi')\n"})

    assert patch is not None
    assert "--- /dev/null" in patch
    assert f"+++ {target}" in patch
    assert "+print('hi')" in patch


@pytest.mark.integration
def test_preview_patch_write_existing_file_is_none(tmp_path):
    """write refuses existing files, so its preview is empty."""
    from xg_project.interface import _preview_patch

    target = tmp_path / "exists.py"
    target.write_text("x\n")

    assert _preview_patch("write", {"path": str(target), "content": "y\n"}) is None


@pytest.mark.integration
def test_preview_patch_edit_shows_old_and_new(tmp_path):
    """An edit preview shows the replaced span."""
    from xg_project.interface import _preview_patch

    target = tmp_path / "a.py"
    target.write_text("alpha\nbeta\n")

    patch = _preview_patch(
        "edit", {"path": str(target), "old_text": "beta", "new_text": "gamma"}
    )

    assert patch is not None
    assert "-beta" in patch
    assert "+gamma" in patch


@pytest.mark.integration
def test_preview_patch_edit_non_unique_is_none(tmp_path):
    """edit requires a unique match, so an ambiguous preview is empty."""
    from xg_project.interface import _preview_patch

    target = tmp_path / "a.py"
    target.write_text("dup\ndup\n")

    assert _preview_patch(
        "edit", {"path": str(target), "old_text": "dup", "new_text": "x"}
    ) is None


@pytest.mark.integration
def test_preview_patch_edit_missing_file_is_none(tmp_path):
    """edit of a missing file has no preview."""
    from xg_project.interface import _preview_patch

    assert _preview_patch(
        "edit", {"path": str(tmp_path / "nope.py"), "old_text": "a", "new_text": "b"}
    ) is None


@pytest.mark.integration
def test_preview_patch_only_applies_to_write_and_edit(tmp_path):
    """Non-mutating tools produce no diff."""
    from xg_project.interface import _preview_patch

    assert _preview_patch("cmd", {"command": "ls"}) is None
    assert _preview_patch("read", {"path": str(tmp_path / "a.py")}) is None


# --- patch formatter ---


@pytest.mark.integration
def test_format_patch_runs_external_command():
    """A working formatter command receives the patch on stdin."""
    from xg_project.interface import _format_patch

    patch = "--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n"

    assert _format_patch(patch, "cat") == patch


@pytest.mark.integration
def test_format_patch_missing_command_falls_back():
    """A command that does not exist falls back to the in-process renderer."""
    from xg_project.interface import _format_patch

    assert _format_patch("+line\n", "definitely-not-a-real-formatter-xyz") is None


@pytest.mark.integration
def test_format_patch_none_means_fallback():
    """No configured formatter falls back to the in-process renderer."""
    from xg_project.interface import _format_patch

    assert _format_patch("+line\n", None) is None


# --- cmd output reaches the conversation ---


@pytest.mark.integration
def test_execute_tool_puts_stdout_in_tool_message(monkeypatch):
    """cmd stdout is carried in the ToolMessage that enters the context."""
    monkeypatch.delenv("XG_PTY", raising=False)

    tm = execute_tool(
        {"name": "cmd", "args": {"command": "echo context-visible"}, "id": "c1"}
    )

    assert tm.tool_call_id == "c1"
    assert "context-visible" in tm.content


@pytest.mark.integration
def test_cmd_under_tty_returns_output_for_context():
    """Under a tty, cmd captures the child output instead of dropping it.

    interact() reads the pty directly and never fills child.before, so this
    covers the regression where the model only saw "(exit code 0)".
    """
    import sys

    import pexpect

    script = (
        "import os; os.environ['XG_PTY'] = '1';"
        "from xg_project.llm._tools import cmd;"
        "print('RESULT=' + repr(cmd.invoke("
        "{'command': 'echo hello-pty; echo boom 1>&2'})))"
    )
    child = pexpect.spawn(sys.executable, ["-c", script], encoding="utf-8", timeout=30)
    child.expect(pexpect.EOF)

    assert "RESULT='hello-pty" in child.before
    assert "boom" in child.before
    assert "(exit code 0)" not in child.before


@pytest.mark.integration
def test_cmd_streams_live_only_under_pty_tty(monkeypatch):
    """cmd output is only suppressed on screen when interact streamed it."""
    from xg_project.interface import _cmd_streams_live

    class _FakeStdin:
        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setenv("XG_PTY", "1")
    monkeypatch.setattr("xg_project.interface.sys.stdin", _FakeStdin(True))
    assert _cmd_streams_live() is True

    monkeypatch.setattr("xg_project.interface.sys.stdin", _FakeStdin(False))
    assert _cmd_streams_live() is False

    monkeypatch.delenv("XG_PTY", raising=False)
    monkeypatch.setattr("xg_project.interface.sys.stdin", _FakeStdin(True))
    assert _cmd_streams_live() is False
