"""Shared test helpers.

Widgets such as ``Static`` keep the Rich *markup* they were given, so a test that
greps the widget's content sees ``[b cyan]◆[/b cyan] command`` rather than what
the user sees. ``plain`` renders the markup away, so assertions can be written
against the visible text instead of against tag placement.

There is also one guard: no test may reach the real password store. Both drivers
resolve their key lazily, and a bare ``Jev()`` or ``Executor()`` would otherwise
shell out to ``pass`` — reading, and possibly printing in a failure diff, a real
key. Tests that exercise key resolution patch the function themselves.
"""

from __future__ import annotations

import pytest
from rich.text import Text


@pytest.fixture(autouse=True)
def no_real_key_stores(monkeypatch):
    """Keep every test off the real ``pass`` store, for both drivers."""
    monkeypatch.setattr("xg_project.jev.api_key_from_pass", lambda entry="jev": None)
    monkeypatch.setattr("xg_project.llm.api_key_from_pass", lambda entry="pi/openrouter": None)


@pytest.fixture
def plain():
    """Turn Rich markup into the text a user would actually read."""

    def _plain(markup: str) -> str:
        return Text.from_markup(markup).plain

    return _plain
