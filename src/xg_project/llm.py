"""LLM setup for xg-project via OpenRouter."""

import subprocess

from langchain_openrouter import ChatOpenRouter

MODEL = "liquid/lfm-2.5-2.6b:free"


def get_api_key() -> str:
    """Fetch the OpenRouter API key from `pass`."""
    return subprocess.run(
        ["pass", "show", "pi/openrouter"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def get_llm(**kwargs) -> ChatOpenRouter:
    return ChatOpenRouter(model=MODEL, openrouter_api_key=get_api_key(), **kwargs)
