"""The TUI, driven headlessly through Textual's pilot.

These assert on what the user can see and do: that Enter sends a prompt, that
``:node`` moves, that Jev's routing moves the marker, and that the input line is
cleared. They deliberately read the widgets rather than app internals where the
widget holds the thing being shown.

Every app here is given a Jev backed by a fixed answer. The tests must not reach
TypeSafe: a suite that routes over the network passes or fails depending on an
API key and a round trip.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, RichLog, Static
from typesafe_sdk import TypeSafeAPIConnectionError

from xg_project.app import GRAPH_ID, LOG_ID, PROMPT_ID, TRAIL_ID, XGApp
from xg_project.jev import Jev
from tests.test_jev import FakeClient, response


@pytest.fixture
def app() -> XGApp:
    """An app whose Jev always routes to ``command``."""
    return XGApp(jev=Jev(client=FakeClient(result=response(choice="command"))))


@pytest.fixture
def lost_app() -> XGApp:
    """An app whose Jev cannot answer, which is an ordinary outcome.

    A `TypeSafeAPIConnectionError` rather than any old exception: only the SDK's
    own failures are turned into a sentence, so that a bug in xg is not quietly
    reported to the user as Jev being unreachable.
    """
    unreachable = TypeSafeAPIConnectionError("unreachable")
    return XGApp(jev=Jev(client=FakeClient(raises=unreachable)))


def graph_text(app: XGApp) -> str:
    return app.query_one(f"#{GRAPH_ID}", Static).content


def trail_text(app: XGApp) -> str:
    return app.query_one(f"#{TRAIL_ID}", Static).content


def log_text(app: XGApp) -> str:
    """Flatten the log's rendered lines back into one searchable string."""
    log = app.query_one(f"#{LOG_ID}", RichLog)
    return "\n".join(str(line) for line in log.lines)


async def submit(app: XGApp, pilot, text: str) -> None:
    """Type ``text`` into the prompt and press Enter the way a user would."""
    prompt = app.query_one(f"#{PROMPT_ID}", Input)
    prompt.focus()
    prompt.value = text
    await pilot.press("enter")
    await pilot.pause()


# -- opening state ---------------------------------------------------------


async def test_opens_at_the_origin(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.session.position == "none"


async def test_the_graph_panel_shows_the_whole_graph(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        drawn = graph_text(app)
        assert "none" in drawn
        assert "command" in drawn


async def test_the_current_node_is_marked(app: XGApp, plain) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        drawn = plain(graph_text(app))
        assert "◆ none" in drawn
        assert "← you are here" in drawn


async def test_the_trail_line_names_the_position(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "you are at" in trail_text(app)
        assert "none" in trail_text(app)


# -- prompts ---------------------------------------------------------------


async def test_enter_sends_a_prompt(app: XGApp) -> None:
    """Enter submits, and what was typed reaches the node."""
    async with app.run_test() as pilot:
        await submit(app, pilot, "make the tests pass")
        assert "make the tests pass" in log_text(app)


async def test_the_input_line_is_cleared_after_submitting(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        assert app.query_one(f"#{PROMPT_ID}", Input).value == ""


async def test_a_prompt_at_the_origin_becomes_the_goal(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "make the tests pass")
        assert app.session.goal == "make the tests pass"
        assert "goal:" in trail_text(app)


# -- routing ---------------------------------------------------------------


async def test_a_prompt_at_a_decision_node_is_routed_by_jev(app: XGApp, plain) -> None:
    """The user states a goal; Jev chooses where it goes, and the marker follows."""
    async with app.run_test() as pilot:
        await submit(app, pilot, "make the tests pass")
        assert app.session.position == "command"
        assert "◆ command" in plain(graph_text(app))


async def test_routing_is_logged_with_its_confidence(app: XGApp) -> None:
    """A confident answer and a coin flip must not look alike in the log."""
    async with app.run_test() as pilot:
        await submit(app, pilot, "make the tests pass")
        assert "confidence" in log_text(app)


async def test_the_decision_node_contribution_is_logged(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "make the tests pass")
        assert "contributed" in log_text(app)


async def test_the_trail_reports_how_much_context_was_built(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "make the tests pass")
        assert "context:" in trail_text(app)
        assert "1 entry" in trail_text(app)


async def test_a_routing_that_fails_leaves_the_user_where_they_were(lost_app: XGApp) -> None:
    """The user can retype; they are not dropped somewhere arbitrary."""
    async with lost_app.run_test() as pilot:
        await submit(lost_app, pilot, "make the tests pass")
        assert lost_app.session.position == "none"
        assert "not routed" in log_text(lost_app)


async def test_a_generative_node_is_not_routed_anywhere(app: XGApp) -> None:
    """The run ends at a generative node, so the position stays there."""
    async with app.run_test() as pilot:
        await submit(app, pilot, ":command")
        await submit(app, pilot, "echo hi")
        assert app.session.position == "command"


async def test_the_opening_screen_says_where_jev_gets_its_key(
    monkeypatch, plain
) -> None:
    """With no key the user is told why routing will not work, not left guessing."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    bare = XGApp()
    async with bare.run_test() as pilot:
        await pilot.pause()
        assert "TYPESAFE_API_KEY" in plain(log_text(bare))


async def test_a_prompt_at_command_is_echoed_by_the_node(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, ":command")
        await submit(app, pilot, "echo hi")
        assert "echo hi" in log_text(app)
        assert app.session.goal is None


async def test_a_prompt_with_markup_characters_is_shown_literally(app: XGApp) -> None:
    """User text is escaped, so brackets do not get eaten as Rich markup."""
    async with app.run_test() as pilot:
        await submit(app, pilot, "[red]not markup[/red]")
        assert "not markup" in log_text(app)


# -- moving ----------------------------------------------------------------


async def test_colon_node_moves_to_that_node(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, ":command")
        assert app.session.position == "command"
        assert app.session.trail == ("none", "command")


async def test_moving_redraws_the_graph_marker(app: XGApp, plain) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, ":command")
        assert "◆ command" in plain(graph_text(app))


async def test_moving_extends_the_trail_line(app: XGApp, plain) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, ":command")
        assert "none → command" in plain(trail_text(app))


async def test_the_user_can_move_back_up(app: XGApp) -> None:
    """Only the user may go up, and the TUI is where the user does it."""
    async with app.run_test() as pilot:
        await submit(app, pilot, ":command")
        await submit(app, pilot, ":none")
        assert app.session.position == "none"


async def test_an_unknown_node_is_refused_and_reported(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, ":nowhere")
        assert app.session.position == "none"
        assert "unknown node" in log_text(app)


async def test_a_bare_colon_is_refused_with_a_hint(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, ":")
        assert app.session.position == "none"
        assert "expected a node name" in log_text(app)


async def test_an_empty_submit_does_nothing(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "   ")
        assert app.session.position == "none"
        assert app.session.goal is None
