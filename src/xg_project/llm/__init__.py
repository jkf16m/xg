"""xg_project.llm — LLM integration for the xg coding agent.

Public API
----------
Conversation:
    add_message(msgs, text) -> list[BaseMessage]
    stream_turn(msgs, config, on_text) -> (response, msgs)
    run_turn(msgs, config) -> (response, msgs)
    ProviderError -> provider-side failure the user can retry

Tools:
    execute_tool(tc) -> ToolMessage
    tool_result(msgs, tm) -> list[BaseMessage]
    TOOLS -> list
"""

from collections.abc import Callable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from openrouter.errors import OpenRouterError

from xg_project.config import Config
from xg_project.llm._api import get_llm, system_prompt
from xg_project.llm._tools import TOOLS
from xg_project.session import append as session_append


class ProviderError(RuntimeError):
    """A provider-side failure (rate limit, outage) that the user can retry."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def add_message(messages: list[BaseMessage], text: str) -> list[BaseMessage]:
    """Append a HumanMessage. No LLM call is made."""
    return messages + [HumanMessage(content=text)]


def _persist(config: Config | None, message: BaseMessage) -> None:
    """Append a message to the session if configured."""
    if config is not None and config.session_path is not None:
        session_append(config.session_path, message)


def stream_turn(
    messages: list[BaseMessage],
    config: Config | None = None,
    on_text: Callable[[str], None] | None = None,
) -> tuple[AIMessage, list[BaseMessage]]:
    """Stream exactly one LLM turn, calling on_text(char) for each character."""
    llm = get_llm().bind_tools(TOOLS)
    response: AIMessage | None = None
    try:
        for chunk in llm.stream([SystemMessage(content=system_prompt()), *messages]):
            content = chunk.content
            if isinstance(content, str):
                for character in content:
                    if on_text is not None:
                        on_text(character)
            if response is None:
                response = chunk
            else:
                response = response + chunk
    except OpenRouterError as exc:
        raise ProviderError(str(exc), exc.status_code) from exc

    if response is None:
        response = AIMessage(content="")
    _persist(config, response)
    return response, messages + [response]


def run_turn(
    messages: list[BaseMessage],
    config: Config | None = None,
) -> tuple[AIMessage, list[BaseMessage]]:
    """Make one non-streaming LLM call."""
    llm = get_llm().bind_tools(TOOLS)
    try:
        response = llm.invoke([SystemMessage(content=system_prompt()), *messages])
    except OpenRouterError as exc:
        raise ProviderError(str(exc), exc.status_code) from exc
    _persist(config, response)
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
