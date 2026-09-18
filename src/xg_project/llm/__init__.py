"""xg_project.llm — model integration for the xg coding agent.

Rebuild status
--------------
The OpenRouter/LangChain implementation was removed on the
``refactor/jev-langgraph`` branch. The agent is being rebuilt on LangGraph,
with TypeSafe Jev (``typesafe_sdk``) for typed decisions.

Nothing here is implemented. The names below are kept so the new graph can be
wired in without re-deciding the module's public surface. Every callable
raises ``NotImplementedError``.

Jev is a *decision* model: ``TypeSafeClient().system_one(state, questions)``
returns ``Noul``/``Choice``/``Score`` answers, not text and not tool calls. It
cannot generate a reply or a tool call on its own. See the branch report.
"""

from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from xg_project.config import Config


class ProviderError(RuntimeError):
    """A provider-side failure (rate limit, outage) that the user can retry."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


TOOLS: list = []
"""Tool definitions bound to the model. Re-implement during the rebuild."""


def add_message(messages: list[BaseMessage], text: str) -> list[BaseMessage]:
    """Append a HumanMessage. No model call is made."""
    raise NotImplementedError("llm.add_message: rebuild pending")


def stream_turn(
    messages: list[BaseMessage],
    config: Config | None = None,
    on_text: Callable[[str], None] | None = None,
) -> tuple[AIMessage, list[BaseMessage]]:
    """Stream exactly one model turn, calling ``on_text(char)`` per character."""
    raise NotImplementedError("llm.stream_turn: rebuild pending")


def run_turn(
    messages: list[BaseMessage],
    config: Config | None = None,
) -> tuple[AIMessage, list[BaseMessage]]:
    """Make one non-streaming model call."""
    raise NotImplementedError("llm.run_turn: rebuild pending")


def execute_tool(tool_call: dict[str, object]) -> ToolMessage:
    """Execute a single tool call and return the result message."""
    raise NotImplementedError("llm.execute_tool: rebuild pending")


def tool_result(messages: list[BaseMessage], tool_message: ToolMessage) -> list[BaseMessage]:
    """Append a ToolMessage to the conversation."""
    raise NotImplementedError("llm.tool_result: rebuild pending")
