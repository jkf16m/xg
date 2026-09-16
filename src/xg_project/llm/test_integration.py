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
from xg_project.session import configure


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    """Use a temporary database for each test."""
    db_path = tmp_path / "test.db"
    configure(db_path)
    yield
    configure(None)


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
    tc = {"name": "read", "args": {"path": "pyproject.toml"}, "id": "test-1"}
    result = execute_tool(tc)
    assert isinstance(result, ToolMessage)


@pytest.mark.integration
def test_execute_tool_preserves_id():
    """execute_tool() preserves the tool_call_id."""
    tc = {"name": "read", "args": {"path": "pyproject.toml"}, "id": "my-id"}
    result = execute_tool(tc)
    assert result.tool_call_id == "my-id"


@pytest.mark.integration
def test_execute_tool_preserves_name():
    """execute_tool() preserves the tool name."""
    tc = {"name": "read", "args": {"path": "pyproject.toml"}, "id": "test-1"}
    result = execute_tool(tc)
    assert result.name == "read"


@pytest.mark.integration
def test_execute_tool_error_returns_error_status():
    """execute_tool() returns error status on failure."""
    tc = {"name": "read", "args": {"path": "/nonexistent"}, "id": "test-err"}
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


# --- system_prompt() tests ---


@pytest.mark.integration
def test_system_prompt_falls_back_to_default(tmp_path):
    """system_prompt() uses the built-in default without .xg/SYSTEM.md."""
    from xg_project.llm._api import SYSTEM, system_prompt

    assert system_prompt(tmp_path) == SYSTEM


@pytest.mark.integration
def test_system_prompt_reads_system_file(tmp_path):
    """system_prompt() reads .xg/SYSTEM.md when present."""
    from xg_project.llm._api import system_prompt

    system_dir = tmp_path / ".xg"
    system_dir.mkdir()
    (system_dir / "SYSTEM.md").write_text("custom prompt", encoding="utf-8")

    assert system_prompt(tmp_path) == "custom prompt"


@pytest.mark.integration
def test_system_prompt_defaults_to_cwd(tmp_path, monkeypatch):
    """system_prompt() resolves against the current working directory by default."""
    from xg_project.llm._api import SYSTEM, system_prompt

    monkeypatch.chdir(tmp_path)

    assert system_prompt() == SYSTEM


# --- project_files() module discovery tests ---


def _write_config(root, body):
    config_dir = root / ".xg"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(body, encoding="utf-8")


@pytest.mark.integration
def test_project_files_uses_module_allowlist(tmp_path):
    """A module with files keeps only listed paths in its subtree."""
    from xg_project.llm._context import project_files

    module = tmp_path / "pkg"
    _write_config(module, '{"files": ["keep.py"]}')
    (module / "keep.py").write_text("")
    (module / "drop.py").write_text("")

    names = {path.name for path in project_files(tmp_path, use_gitignore=False)}

    assert "keep.py" in names
    assert "drop.py" not in names


@pytest.mark.integration
def test_project_files_module_without_files_keeps_subtree(tmp_path):
    """A module without a files key does not restrict its subtree."""
    from xg_project.llm._context import project_files

    module = tmp_path / "pkg"
    _write_config(module, '{"use_gitignore": false}')
    (module / "keep.py").write_text("")

    names = {path.name for path in project_files(tmp_path, use_gitignore=False)}

    assert "keep.py" in names


@pytest.mark.integration
def test_project_files_empty_allowlist_keeps_nothing(tmp_path):
    """An explicit empty files list keeps no files in the module subtree."""
    from xg_project.llm._context import project_files

    module = tmp_path / "pkg"
    _write_config(module, '{"files": []}')
    (module / "drop.py").write_text("")

    names = {path.name for path in project_files(tmp_path, use_gitignore=False)}

    assert "drop.py" not in names


@pytest.mark.integration
def test_load_context_module_requires_config(tmp_path):
    """A directory is a module only when .xg/config.json exists."""
    from xg_project.llm._context import load_context_module

    assert load_context_module(tmp_path) is None

    _write_config(tmp_path, "{}")

    module = load_context_module(tmp_path)
    assert module is not None
    assert module.files is None
