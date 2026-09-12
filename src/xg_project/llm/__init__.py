"""xg_project.llm — LLM integration for the xg coding agent.

Public API
----------
Context:
    initial_file_messages(root) -> list[BaseMessage]

Conversation:
    add_message(msgs, text) -> list[BaseMessage]
    stream_turn(msgs, on_text) -> (response, msgs)
    run_turn(msgs) -> (response, msgs)

Tools:
    execute_tool(tc) -> ToolMessage
    tool_result(msgs, tm) -> list[BaseMessage]
    TOOLS -> list
"""

from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from xg_project.llm._api import SYSTEM, get_llm
from xg_project.llm._context import project_files
from xg_project.llm._tools import TOOLS


def initial_file_messages(root: Path | None = None) -> list[BaseMessage]:
    """Materialize the launch snapshot as read-tool-call/tool-result messages."""
    root = (root or Path.cwd()).resolve()
    result: list[BaseMessage] = []
    for number, path in enumerate(project_files(root), 1):
        try:
            content = path.read_text()
        except (UnicodeDecodeError, OSError) as exc:
            content = f"[unreadable file: {exc}]"
        tool_id = f"launch-read-{number}"
        result.extend([
            AIMessage(content="", tool_calls=[{
                "name": "read_file", "args": {"path": str(path)}, "id": tool_id, "type": "tool_call"
            }]),
            ToolMessage(content=content, tool_call_id=tool_id, name="read_file"),
        ])
    return result


def add_message(messages: list[BaseMessage], text: str) -> list[BaseMessage]:
    """Append a HumanMessage. No LLM call is made."""
    return messages + [HumanMessage(content=text)]


def stream_turn(
    messages: list[BaseMessage],
    on_text: Callable[[str], None] | None = None,
) -> tuple[AIMessage, list[BaseMessage]]:
    """Stream exactly one LLM turn, calling on_text(char) for each character."""
    llm = get_llm().bind_tools(TOOLS)
    response: AIMessage | None = None
    for chunk in llm.stream([SystemMessage(content=SYSTEM), *messages]):
        content = chunk.content
        if isinstance(content, str):
            for character in content:
                if on_text is not None:
                    on_text(character)
        if response is None:
            response = chunk
        else:
            response = response + chunk

    if response is None:
        response = AIMessage(content="")
    return response, messages + [response]


def run_turn(messages: list[BaseMessage]) -> tuple[AIMessage, list[BaseMessage]]:
    """Make one non-streaming LLM call."""
    llm = get_llm().bind_tools(TOOLS)
    response = llm.invoke([SystemMessage(content=SYSTEM), *messages])
    return response, messages + [response]


def execute_tool(tool_call: dict[str, object]) -> ToolMessage:
    """Execute a single tool call and return the result message."""
    name = str(tool_call["name"])
    args = tool_call["args"]
    tool_id = str(tool_call["id"])
    tool_fn = {t.name: t for t in TOOLS}[name]
    try:
        result = tool_fn.invoke(args)
        return ToolMessage(content=str(result), tool_call_id=tool_id, name=name)
    except Exception as exc:
        return ToolMessage(content=str(exc), tool_call_id=tool_id, name=name, status="error")


def tool_result(messages: list[BaseMessage], tool_message: ToolMessage) -> list[BaseMessage]:
    """Append a ToolMessage to the conversation."""
    return messages + [tool_message]
