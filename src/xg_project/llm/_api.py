"""Private: OpenRouter LLM configuration."""

import subprocess

from langchain_openrouter import ChatOpenRouter

MODEL = "@preset/mimo"
SYSTEM = """You are xg, a coding agent in a 1:1 human-guided loop.
There is exactly one agent turn for each human message. Do not invent a task,
do not stop because of a stop reason, and do not ask for confirmation before
using tools. The launch context contains the current project files in mtime
order. A read_file call appends that file to the bottom of context. Respond
with a concise summary when your turn is complete."""


def get_api_key() -> str:
    return subprocess.run(
        ["pass", "show", "pi/openrouter"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def get_llm(**kwargs) -> ChatOpenRouter:
    return ChatOpenRouter(model=MODEL, openrouter_api_key=get_api_key(), **kwargs)
