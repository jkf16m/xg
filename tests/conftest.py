"""Shared test helpers.

Widgets such as ``Static`` keep the Rich *markup* they were given, so a test that
greps the widget's content sees ``[b cyan]◆[/b cyan] command`` rather than what
the user sees. ``plain`` renders the markup away, so assertions can be written
against the visible text instead of against tag placement.
"""

from __future__ import annotations

import pytest
from rich.text import Text


@pytest.fixture
def plain():
    """Turn Rich markup into the text a user would actually read."""

    def _plain(markup: str) -> str:
        return Text.from_markup(markup).plain

    return _plain
