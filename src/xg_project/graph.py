"""The default human -> agent LangGraph for xg."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from xg_project.llm import get_llm


class GraphState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    prompt: str


def project_files(root: Path | None = None) -> list[Path]:
    """Return project files in deterministic (mtime, path) order.

    Generated environments are deliberately excluded: loading .git/.venv into
    the initial conversation is both surprising and needlessly enormous.
    """
    root = (root or Path.cwd()).resolve()
    ignored = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
    files = [p for p in root.rglob("*") if p.is_file() and not ignored.intersection(p.relative_to(root).parts)]
    return sorted(files, key=lambda p: (p.stat().st_mtime_ns, p.relative_to(root).as_posix()))


def initial_file_messages(root: Path | None = None) -> list:
    """Materialize the launch snapshot as read-tool-call/tool-result messages."""
    root = (root or Path.cwd()).resolve()
    result = []
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


@tool
def read_file(path: str) -> str:
    """Read a file; its result is appended at the bottom of the context."""
    return Path(path).read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Write a file."""
    Path(path).write_text(content)
    return f"wrote {path} ({len(content)} chars)"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one exact, unique occurrence in a file."""
    text = Path(path).read_text()
    if text.count(old_text) != 1:
        return f"error: old_text must occur exactly once in {path}"
    Path(path).write_text(text.replace(old_text, new_text))
    return f"edited {path}"


@tool
def run_cmd(command: str) -> str:
    """Run a command; interactive commands temporarily take over the PTY."""
    if os.environ.get("XG_PTY") != "1":
        completed = subprocess.run(command, shell=True, capture_output=True, text=True)
        return (completed.stdout + completed.stderr).strip() or f"(exit code {completed.returncode})"

    # pexpect connects the command to the same terminal as the xg child.
    # interact() is important: it forwards the user's keystrokes, so programs
    # such as nvim, less, ssh, and password prompts are actually usable.
    import pexpect

    child = pexpect.spawn(
        "/bin/sh", ["-c", command],
        encoding="utf-8",
        timeout=None,
        dimensions=(24, 80),
    )
    try:
        if sys.stdin.isatty():
            child.interact(escape_character=None)
        else:
            # Non-interactive callers (for example tests or pipes) still get
            # captured PTY output without requiring a terminal stdin.
            child.expect(pexpect.EOF)
        output = child.before or ""
        status = child.exitstatus
    finally:
        child.close(force=True)

    return output.strip() or f"(exit code {status or 0})"


TOOLS = [read_file, write_file, edit_file, run_cmd]
SYSTEM = """You are xg, a coding agent in a 1:1 human-guided loop.
There is exactly one agent turn for each human message. Do not invent a task,
do not stop because of a stop reason, and do not ask for confirmation before
using tools. The launch context contains the current project files in mtime
order. A read_file call appends that file to the bottom of context. Respond
with a concise summary when your turn is complete."""


def agent_node(state: GraphState) -> dict:
    llm = get_llm().bind_tools(TOOLS)
    messages = list(state.get("messages", []))
    # The prompt is added once. Tool turns re-enter this node with a
    # ToolMessage at the end and must not duplicate the human message.
    if not messages or not isinstance(messages[-1], HumanMessage):
        messages.append(HumanMessage(content=state.get("prompt", "")))
    response = llm.invoke([SystemMessage(content=SYSTEM), *messages])
    return {"messages": [response]}


def route_agent(state: GraphState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_agent, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile()


graph = build_graph()
