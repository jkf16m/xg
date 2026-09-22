"""The TUI, driven headlessly through Textual's pilot.

These assert on what the user can see and do: that Enter sends a prompt, that
``/go`` moves, and that the state panel shows what each node introduced. They
deliberately read the widgets rather than app internals where the widget holds
the thing being shown.

Every app here is given a Jev and an executor backed by fixed answers. The tests
must not reach TypeSafe or OpenRouter: a suite that routes over the network
passes or fails depending on a key and a round trip.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from textual.containers import ScrollableContainer, VerticalScroll
from textual.widgets import Input, LoadingIndicator, Static
from typesafe_sdk import TypeSafeAPIConnectionError

from tests.test_jev import FakeClient
from tests.test_llm import FakeHttp, FakeResponse, command_response
from tests.test_turn import FakeExecutor, FakeJev
from xg_project.app import (
    BUSY_ID,
    GRAPH_ID,
    PROMPT_ID,
    STATE_ID,
    STATE_SCROLL_ID,
    STATUS_ID,
    XGApp,
)
from xg_project.edit import EditOutcome
from xg_project.graph import (
    ADD,
    ANSWER,
    COMMAND,
    EDIT,
    FILTER,
    HERE,
    ORIGIN,
    SELECT_MODULE,
    SORT,
)
from xg_project.jev import Jev
from xg_project.llm import AddProposal, EditProposal, Executor
from xg_project.shell import Outcome


class FakeRunner:
    """Stands in for `run_command`, recording what it was asked to run."""

    def __init__(self, *, outcome: Outcome | None = None) -> None:
        self.outcome = outcome or Outcome(command="", code=0, stdout="hello from the command\n")
        self.calls: list[str] = []

    def __call__(self, command: str) -> Outcome:
        self.calls.append(command)
        return replace(self.outcome, command=command)


class FakeEditor:
    """Stands in for `apply_edit`, recording the proposals it was handed."""

    def __init__(self, *, outcome: EditOutcome | None = None) -> None:
        self.outcome = outcome
        self.calls: list[EditProposal] = []

    def __call__(self, proposal: EditProposal) -> EditOutcome:
        self.calls.append(proposal)
        if self.outcome is not None:
            return self.outcome
        return EditOutcome(path=proposal.path or "", applied=True)


class FakeAdder:
    """Stands in for `apply_add`, recording the proposals it was handed."""

    def __init__(self, *, outcome: EditOutcome | None = None) -> None:
        self.outcome = outcome
        self.calls: list[AddProposal] = []

    def __call__(self, proposal: AddProposal) -> EditOutcome:
        self.calls.append(proposal)
        if self.outcome is not None:
            return self.outcome
        return EditOutcome(path=proposal.path or "", applied=True, created=True)


class BlockingJev(FakeJev):
    """A Jev whose answer is held open until the test releases it.

    A turn that returns instantly cannot be observed mid-flight, so this is what
    the spinner tests use to look at the screen while a request is in the air.
    """

    def __init__(self, **overrides) -> None:
        super().__init__(**overrides)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def decide(self, **kwargs):
        self.started.set()
        await self.release.wait()
        return await super().decide(**kwargs)


class BlockingRunner:
    """A command runner that sits in its thread until the test releases it.

    Plain `threading.Event`s, because the command is carried out with
    ``asyncio.to_thread`` and an event loop primitive cannot be awaited there.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []

    def __call__(self, command: str) -> Outcome:
        self.calls.append(command)
        self.entered.set()
        self.release.wait(timeout=5)
        return Outcome(command=command, code=0, stdout="done\n")


def make_app(**overrides) -> XGApp:
    kwargs = {
        "jev": FakeJev(choice=COMMAND),
        "executor": FakeExecutor(command="echo hi"),
        "runner": FakeRunner(),
        "editor": FakeEditor(),
        "adder": FakeAdder(),
        "root": Path("/nonexistent-xyz"),
    }
    kwargs.update(overrides)
    return XGApp(**kwargs)


@pytest.fixture
def app() -> XGApp:
    """An app whose Jev routes to `_XG_COMMAND` and whose executor proposes `echo hi`."""
    return make_app()


@pytest.fixture
def lost_app() -> XGApp:
    """An app whose Jev cannot answer, which is an ordinary outcome."""
    unreachable = TypeSafeAPIConnectionError("unreachable")
    return make_app(jev=Jev(client=FakeClient(raises=unreachable)))


def graph_text(app: XGApp) -> str:
    return app.query_one(f"#{GRAPH_ID}", Static).content


def marked_node(app: XGApp) -> str:
    """The line the drawing puts the where-you-are marker on, box and all.

    The marker is drawn inside the current node's own box, so the node it is
    about is the name on the line above it. Asserting that a name is somewhere on
    screen would not distinguish the node the run is on from the seven it is not.
    """
    lines = graph_text(app).splitlines()
    marked = next(index for index, line in enumerate(lines) if HERE in line)
    return lines[marked - 1]


def state_text(app: XGApp) -> str:
    return app.query_one(f"#{STATE_ID}", Static).content


def status_text(app: XGApp) -> str:
    return app.query_one(f"#{STATUS_ID}", Static).content


def spinner_visible(app: XGApp) -> bool:
    """Whether the busy row is taking space in the layout."""
    return app.query_one(f"#{BUSY_ID}", LoadingIndicator).display


async def screen_text(app: XGApp) -> str:
    """Everything actually on the display, as one string.

    Read from the compositor rather than from the widgets, because the question a
    widget cannot answer is the one that matters: whether a thing is *visible*. A
    notification, or a pane scrolled past, still exists as a widget while being
    nowhere on the screen.
    """
    strips = app.screen._compositor.render_strips()
    if inspect.isawaitable(strips):
        strips = await strips
    return "\n".join(strip.text for strip in strips)


async def press_enter(app: XGApp, pilot, text: str) -> None:
    """Type ``text`` and press Enter, without waiting for the turn it starts."""
    prompt = app.query_one(f"#{PROMPT_ID}", Input)
    prompt.focus()
    prompt.value = text
    await pilot.press("enter")
    await pilot.pause()


async def submit(app: XGApp, pilot, text: str) -> None:
    """Type ``text``, press Enter, and wait for the turn it starts to finish.

    A turn runs in a worker, so pressing Enter no longer implies the model has
    answered. Waiting here is what keeps the tests about outcomes rather than
    scheduling.
    """
    await press_enter(app, pilot, text)
    await app.workers.wait_for_complete()
    await pilot.pause()


async def drive_to_edit(app: XGApp, pilot) -> None:
    """Submit five prompts to walk ORIGIN -> SELECT_MODULE -> FILTER -> SORT -> EDIT."""
    await submit(app, pilot, "edit a file")
    await submit(app, pilot, "go")
    await submit(app, pilot, "go")
    await submit(app, pilot, "go")
    await submit(app, pilot, "go")


# -- opening state ---------------------------------------------------------


async def test_opens_at_the_origin(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.session.position == ORIGIN


async def test_the_graph_panel_shows_the_whole_graph(app: XGApp, plain) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        drawn = plain(graph_text(app))
        for name in (ORIGIN, ADD, COMMAND, SELECT_MODULE, FILTER, SORT, EDIT, ANSWER):
            assert name in drawn


async def test_the_state_panel_starts_empty(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "no state" in state_text(app)


# -- the two windows -------------------------------------------------------


async def test_the_graph_and_the_state_are_separate_windows(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(f"#{GRAPH_ID}", Static).display
        assert app.query_one(f"#{STATE_SCROLL_ID}", VerticalScroll).display


async def test_only_the_state_pane_is_a_scrolling_container(app: XGApp) -> None:
    """The tree is never the thing that scrolls; the state always is."""
    async with app.run_test() as pilot:
        await pilot.pause()
        graph = app.query_one(f"#{GRAPH_ID}", Static)
        state = app.query_one(f"#{STATE_ID}", Static)
        assert not isinstance(graph, ScrollableContainer)
        assert isinstance(state.parent, ScrollableContainer)


async def test_the_trail_stays_with_the_tree_and_out_of_the_scrolling_state(
    app: XGApp,
) -> None:
    """The trail answers "where am I", so it must not scroll away with the state."""
    async with app.run_test() as pilot:
        await submit(app, pilot, "make the tests pass")
        assert "trail:" in graph_text(app)
        assert "trail:" not in state_text(app)


async def test_a_long_state_scrolls_without_pushing_the_tree_off(app: XGApp) -> None:
    """The failure this guards: a growing state squeezing the tree out of sight."""
    async with app.run_test() as pilot:
        for index in range(80):
            app.session.record(f"node{index}", "a value long enough to take a row")
        app.refresh_state()
        await pilot.pause()

        scroll = app.query_one(f"#{STATE_SCROLL_ID}", VerticalScroll)
        assert scroll.max_scroll_y > 0
        assert app.query_one(f"#{GRAPH_ID}", Static).display
        assert app.query_one(f"#{PROMPT_ID}", Input).display

        app.action_state_down()
        await pilot.pause()
        assert scroll.scroll_offset.y > 0


async def test_pagedown_scrolls_the_state_while_the_input_keeps_focus(app: XGApp) -> None:
    """The input line holds focus, so the binding has to reach past it."""
    async with app.run_test() as pilot:
        for index in range(80):
            app.session.record(f"node{index}", "a value long enough to take a row")
        app.refresh_state()
        await pilot.pause()
        scroll = app.query_one(f"#{STATE_SCROLL_ID}", VerticalScroll)

        await pilot.press("pagedown")
        await pilot.pause()
        assert scroll.scroll_offset.y > 0

        await pilot.press("pageup")
        await pilot.pause()
        assert scroll.scroll_offset.y == 0


# -- prompts ---------------------------------------------------------------


async def test_enter_sends_a_prompt_and_the_origin_records_it(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "make the tests pass")
        assert app.session.state[ORIGIN] == "the user wants: make the tests pass"
        assert "make the tests pass" in state_text(app)


async def test_the_state_panel_is_keyed_by_node(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "make the tests pass")
        assert ORIGIN in state_text(app)


async def test_the_input_line_is_kept_after_submitting(app: XGApp) -> None:
    """Submitting does not clear the line: it can be edited and sent again."""
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        assert app.query_one(f"#{PROMPT_ID}", Input).value == "hello"


async def test_ctrl_l_clears_the_input_line(app: XGApp) -> None:
    """``ctrl+l`` is the one thing that empties the line."""
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert app.query_one(f"#{PROMPT_ID}", Input).value == ""


# -- the spinner -----------------------------------------------------------


async def test_the_spinner_is_hidden_when_nothing_is_running(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        assert spinner_visible(app) is False


async def test_the_spinner_shows_while_a_turn_is_in_flight() -> None:
    """A prompt that awaits Jev draws a wait rather than a frozen screen."""
    app = make_app(jev=BlockingJev(choice=COMMAND))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert spinner_visible(app) is False

        await press_enter(app, pilot, "make the tests pass")
        await app.jev.started.wait()
        assert spinner_visible(app) is True

        app.jev.release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert spinner_visible(app) is False


async def test_a_slow_turn_does_not_freeze_the_app() -> None:
    """The pump stays free while the model thinks, so the user can still act."""
    app = make_app(jev=BlockingJev(choice=COMMAND))
    async with app.run_test() as pilot:
        await press_enter(app, pilot, "make the tests pass")
        await app.jev.started.wait()

        # The turn is still blocked on Jev, yet the move is processed: the app
        # would be unresponsive here if the turn were awaited in the pump.
        await press_enter(app, pilot, f"/go {FILTER}")
        assert app.session.position == FILTER

        app.jev.release.set()
        await app.workers.wait_for_complete()


async def test_the_spinner_is_hidden_when_a_turn_ends_on_a_gate() -> None:
    """Waiting for the user is not waiting for a model, so the spinner stops."""
    app = make_app()
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")
        assert app.gate is not None
        assert spinner_visible(app) is False


async def test_the_spinner_shows_while_an_accepted_command_runs() -> None:
    """An accepted command runs in a thread; a slow one is still a wait."""
    runner = BlockingRunner()
    app = make_app(runner=runner)
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")

        app.action_accept_proposal()
        for _ in range(200):
            if runner.entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert runner.entered.is_set()
        await pilot.pause()
        assert spinner_visible(app) is True

        runner.release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert spinner_visible(app) is False


# -- routing ---------------------------------------------------------------


async def test_a_prompt_at_the_origin_is_routed_by_jev(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "make the tests pass")
        assert app.session.position == COMMAND
        assert COMMAND in marked_node(app)


async def test_the_move_is_reported_on_the_status_line(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "make the tests pass")
        assert COMMAND in status_text(app)


async def test_a_routing_that_fails_is_reported_and_leaves_the_user_where_they_were(
    lost_app: XGApp,
) -> None:
    async with lost_app.run_test() as pilot:
        await submit(lost_app, pilot, "make the tests pass")
        assert lost_app.session.position == ORIGIN
        assert "not routed" in status_text(lost_app)


# -- moving ----------------------------------------------------------------


async def test_a_go_command_moves_to_that_node(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {FILTER}")
        assert app.session.position == FILTER
        assert app.session.trail == (ORIGIN, SELECT_MODULE, FILTER)


async def test_go_dot_dot_moves_to_the_parent(app: XGApp) -> None:
    """``/go ..`` means the parent, the way a shell reads it."""
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {SORT}")
        await submit(app, pilot, "/go ..")
        assert app.session.position == FILTER


async def test_go_dot_dot_at_the_origin_is_refused(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "/go ..")
        assert app.session.position == ORIGIN
        assert "nothing above it" in status_text(app)


async def test_an_unknown_node_is_refused_and_reported(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "/go nowhere")
        assert app.session.position == ORIGIN
        assert "unknown node" in status_text(app)


async def test_the_short_spelling_of_a_command_moves_too(app: XGApp) -> None:
    """``/g`` is ``/go``; the alias has to reach the same move, not a message."""
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/g {FILTER}")
        assert app.session.position == FILTER


async def test_a_bare_slash_is_refused_and_says_where_the_commands_are(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "/")
        assert "nothing after" in status_text(app)
        assert "/help" in status_text(app)


async def test_a_command_that_cannot_be_read_is_refused_with_its_usage(app: XGApp) -> None:
    """``/go`` wants a node, and being told so beats a silent nothing."""
    async with app.run_test() as pilot:
        await submit(app, pilot, "/go")
        assert app.session.position == ORIGIN
        assert "Missing argument" in status_text(app)
        assert "/go <node>" in status_text(app)


async def test_a_command_never_runs_a_turn(app: XGApp) -> None:
    """Commands are the interface's, so no model is asked anything and no gate opens."""
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {FILTER}")
        assert app.gate is None
        assert app.session.state == {}


async def test_a_command_is_left_in_the_prompt_as_it_was_typed(app: XGApp) -> None:
    """Submitting does not clear the line, and a command is submitted like any line."""
    async with app.run_test() as pilot:
        await press_enter(app, pilot, f"/go {FILTER}")
        assert app.query_one(f"#{PROMPT_ID}", Input).value == f"/go {FILTER}"


async def test_help_answers_in_the_interface_and_leaves_the_run_alone() -> None:
    """Reference material, so it is neither state nor a status line.

    Notifications are off in ``run_test`` by default, and this test turns them on:
    what matters is that the help reaches the screen, not that an object was made.
    """
    app = make_app()
    async with app.run_test(notifications=True) as pilot:
        await submit(app, pilot, "/help")
        assert app.session.position == ORIGIN
        assert app.session.state == {}

        shown = await screen_text(app)
        assert "/go <node>" in shown
        assert "(also /g)" in shown
        assert "(also /?)" in shown


async def test_clear_empties_the_prompt_line_and_nothing_else(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, "/clear")
        assert app.query_one(f"#{PROMPT_ID}", Input).value == ""
        assert app.session.position == ORIGIN
        assert app.session.state == {}


async def test_a_command_that_changes_nothing_spares_a_waiting_proposal(app: XGApp) -> None:
    """Asking for the help is not a reason to throw away an undecided proposal."""
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")
        assert app.gate is not None

        await submit(app, pilot, "/help")

        assert app.gate is not None


async def test_going_somewhere_by_command_leaves_a_waiting_proposal_behind(app: XGApp) -> None:
    """Same rule as ``ctrl+u``: the proposal belongs to the node being left."""
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")
        assert app.gate is not None

        await submit(app, pilot, f"/go {FILTER}")

        assert app.gate is None
        assert app.session.position == FILTER
        assert "set aside" in status_text(app)

async def test_the_old_colon_syntax_is_reported_rather_than_sent_as_a_prompt(app: XGApp) -> None:
    """A stale ``:..`` read as prose would be a line handed to a model."""
    async with app.run_test() as pilot:
        await submit(app, pilot, ":..")
        assert app.session.position == ORIGIN
        assert app.session.state == {}
        assert "/go .." in status_text(app)
        assert "start with" in status_text(app)


async def test_the_old_syntax_hint_names_help_when_there_is_nothing_to_carry_over(
    app: XGApp,
) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, ":")
        assert "/help" in status_text(app)


async def test_ctrl_u_moves_up_to_the_parent_node(app: XGApp) -> None:
    """The keyboard's ``/go ..``: same move, no typing."""
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {FILTER}")
        assert app.session.position == FILTER

        await pilot.press("ctrl+u")
        await pilot.pause()

        assert app.session.position == SELECT_MODULE
        assert app.session.trail == (ORIGIN, SELECT_MODULE)


async def test_ctrl_u_leaves_the_prompt_line_alone(app: XGApp) -> None:
    """ctrl+l is the only thing that empties the line, keys included."""
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {FILTER}")
        prompt = app.query_one(f"#{PROMPT_ID}", Input)
        assert prompt.value == f"/go {FILTER}"

        await pilot.press("ctrl+u")
        await pilot.pause()

        assert prompt.value == f"/go {FILTER}"


async def test_ctrl_u_at_the_origin_says_there_is_nothing_above(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await pilot.press("ctrl+u")
        await pilot.pause()
        assert app.session.position == ORIGIN
        assert "nothing above it" in status_text(app)


async def test_ctrl_u_sets_aside_a_waiting_proposal() -> None:
    """Walking away must not leave a proposal a later accept would still act on."""
    runner = FakeRunner()
    app = make_app(runner=runner)
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "where am i")
        assert app.gate is not None

        await pilot.press("ctrl+u")
        await pilot.pause()

        assert app.session.position == ORIGIN
        assert app.gate is None
        assert COMMAND not in app.session.state
        assert runner.calls == []


# -- the command gate ------------------------------------------------------


async def test_a_prompt_at_command_proposes_a_command_and_waits(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")
        assert app.gate is not None
        assert app.gate.proposal.command == "echo hi"
        assert app.runner.calls == []
        assert "nothing has happened yet" in status_text(app)


async def test_a_pending_command_gets_no_patch(app: XGApp) -> None:
    """There is no file changing, so there is nothing to diff."""
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")
        assert "diff --git" not in state_text(app)


async def test_the_state_panel_shows_the_pending_proposal(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")
        assert "pending" in state_text(app)
        assert "echo hi" in state_text(app)


async def test_ctrl_y_accepts_and_runs_the_pending_command(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")
        await pilot.press("ctrl+y")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.runner.calls == ["echo hi"]
        assert app.gate is None
        assert "hello from the command" in status_text(app)


async def test_the_run_outcome_is_shown_with_its_exit_code(app: XGApp) -> None:
    app.runner = FakeRunner(outcome=Outcome(command="", code=3, stderr="nope\n"))
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")
        await pilot.press("ctrl+y")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "nope" in status_text(app)
        assert "exit 3" in status_text(app)


async def test_ctrl_n_rejects_without_running(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert app.runner.calls == []
        assert app.gate is None
        assert "rejected" in status_text(app)
        assert "rejected" in state_text(app)


async def test_a_new_prompt_sets_the_pending_proposal_aside_without_running_it(
    app: XGApp,
) -> None:
    """The old proposal was never accepted, so it must not run behind the user's back."""
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")
        first = app.gate
        assert first is not None
        await submit(app, pilot, "something else")
        assert app.runner.calls == []
        assert first.status == "rejected"
        # The new line is a new proposal at `command`, not the old one carried over.
        assert app.gate is not first


async def test_accepting_with_nothing_pending_does_nothing(app: XGApp) -> None:
    async with app.run_test() as pilot:
        await pilot.press("ctrl+y")
        await pilot.pause()
        assert app.runner.calls == []


async def test_a_proposal_that_fails_is_reported_and_holds_nothing() -> None:
    app = make_app(executor=FakeExecutor(problem="no key for the executor"))
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "echo hi")
        assert app.gate is None
        assert "nothing proposed" in status_text(app)


async def test_the_command_workflow_reaches_the_executor_by_name() -> None:
    """The app drives the real Executor through the node, not a special case."""
    executor = Executor(
        client=FakeHttp(response=FakeResponse(200, command_response(command="pwd")))
    )
    app = make_app(executor=executor)
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/go {COMMAND}")
        await submit(app, pilot, "where am i")
        assert app.gate is not None
        assert app.gate.proposal.command == "pwd"


# -- the edit workflow -----------------------------------------------------


@pytest.fixture
def edit_app(tmp_path: Path) -> XGApp:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    return make_app(
        jev=FakeJev(choice=SELECT_MODULE, scores={"a.py": 0.2, "b.py": 0.9}),
        executor=FakeExecutor(old="y = 2", new="y = 42"),
        editor=None,
        root=tmp_path,
    )


async def test_the_edit_workflow_fills_the_state_node_by_node(edit_app: XGApp) -> None:
    async with edit_app.run_test() as pilot:
        await drive_to_edit(edit_app, pilot)
        assert edit_app.session.position == EDIT
        assert set(edit_app.session.state[FILTER]["files"]) == {"a.py", "b.py"}
        assert list(edit_app.session.state[SORT]["files"]) == ["b.py", "a.py"]


async def test_edit_uses_the_top_ranked_file(edit_app: XGApp) -> None:
    async with edit_app.run_test() as pilot:
        await drive_to_edit(edit_app, pilot)
        assert edit_app.gate is not None
        assert edit_app.gate.proposal.path == "b.py"
        assert edit_app.gate.proposal.old_text == "y = 2"


async def test_accepting_an_edit_writes_the_file(edit_app: XGApp) -> None:
    """The default editor is real: accepting the proposal changes the file on disk."""
    async with edit_app.run_test() as pilot:
        await drive_to_edit(edit_app, pilot)
        await pilot.press("ctrl+y")
        await edit_app.workers.wait_for_complete()
        await pilot.pause()
        assert (edit_app.root / "b.py").read_text(encoding="utf-8") == "y = 42\n"
        assert edit_app.gate is None
        assert "applied" in status_text(edit_app)


async def test_rejecting_an_edit_leaves_the_file_alone(edit_app: XGApp) -> None:
    async with edit_app.run_test() as pilot:
        await drive_to_edit(edit_app, pilot)
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert (edit_app.root / "b.py").read_text(encoding="utf-8") == "y = 2\n"
        assert "rejected" in status_text(edit_app)


async def test_a_pending_edit_is_previewed_as_a_patch(edit_app: XGApp) -> None:
    """What is changing in which file, readable, before anything is written."""
    async with edit_app.run_test() as pilot:
        await drive_to_edit(edit_app, pilot)

        shown = state_text(edit_app)
        assert "diff --git a/b.py b/b.py" in shown
        assert "-y = 2" in shown
        assert "+y = 42" in shown


async def test_the_patch_appears_before_the_edit_is_accepted(edit_app: XGApp) -> None:
    """The preview is for deciding, so it cannot arrive with the outcome."""
    async with edit_app.run_test() as pilot:
        await drive_to_edit(edit_app, pilot)
        assert (edit_app.root / "b.py").read_text(encoding="utf-8") == "y = 2\n"
        assert "+y = 42" in state_text(edit_app)


# -- the add workflow ------------------------------------------------------


@pytest.fixture
def add_app(tmp_path: Path) -> XGApp:
    """A request for a new file, answered by an addition from the origin."""
    return make_app(
        jev=FakeJev(choice=ADD),
        executor=FakeExecutor(path="pkg/greet.py", content='def greet():\n    return "hi"\n'),
        adder=None,
        root=tmp_path,
    )


async def drive_to_add(app: XGApp, pilot) -> None:
    """Move to ADD and run it, leaving its proposal pending."""
    await submit(app, pilot, f"/go {ADD}")
    await submit(app, pilot, "add a module that greets")


async def test_a_new_file_request_reaches_the_add_node(add_app: XGApp) -> None:
    async with add_app.run_test() as pilot:
        await drive_to_add(add_app, pilot)
        assert add_app.session.position == ADD
        assert add_app.gate is not None
        assert add_app.gate.proposal.path == "pkg/greet.py"
        assert ADD in marked_node(add_app)


async def test_a_pending_addition_is_previewed_as_a_creating_patch(add_app: XGApp) -> None:
    """The patch says the file is new, because that is what accepting would do."""
    async with add_app.run_test() as pilot:
        await drive_to_add(add_app, pilot)

        shown = state_text(add_app)
        assert "diff --git a/pkg/greet.py b/pkg/greet.py" in shown
        assert "new file mode 100644" in shown
        assert "--- /dev/null" in shown
        assert "+def greet():" in shown


async def test_the_creating_patch_appears_before_acceptance(add_app: XGApp) -> None:
    async with add_app.run_test() as pilot:
        await drive_to_add(add_app, pilot)
        assert not (add_app.root / "pkg/greet.py").exists()
        assert "new file mode" in state_text(add_app)


async def test_accepting_an_addition_creates_the_file(add_app: XGApp) -> None:
    """The default adder is real: accepting creates the file, nested directory and all."""
    async with add_app.run_test() as pilot:
        await drive_to_add(add_app, pilot)
        await pilot.press("ctrl+y")
        await add_app.workers.wait_for_complete()
        await pilot.pause()

        assert (add_app.root / "pkg/greet.py").read_text(encoding="utf-8") == (
            'def greet():\n    return "hi"\n'
        )
        assert add_app.gate is None
        assert "created" in status_text(add_app)


async def test_rejecting_an_addition_writes_nothing(add_app: XGApp) -> None:
    async with add_app.run_test() as pilot:
        await drive_to_add(add_app, pilot)
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert not (add_app.root / "pkg/greet.py").exists()
        assert "rejected" in status_text(add_app)


async def test_an_addition_is_refused_when_the_file_already_exists(tmp_path: Path) -> None:
    """The one check a preview cannot make, since the file may appear while it waits."""
    (tmp_path / "taken.py").write_text("already here\n", encoding="utf-8")
    app = make_app(
        jev=FakeJev(choice=ADD),
        executor=FakeExecutor(path="taken.py", content="something else\n"),
        adder=None,
        root=tmp_path,
    )
    async with app.run_test() as pilot:
        await drive_to_add(app, pilot)
        await pilot.press("ctrl+y")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert (tmp_path / "taken.py").read_text(encoding="utf-8") == "already here\n"
        assert "already exists" in status_text(app)


async def test_accepting_an_addition_goes_through_the_injected_adder(tmp_path: Path) -> None:
    """An addition is dispatched to the adder, not to the editor or the runner."""
    adder = FakeAdder()
    app = make_app(
        jev=FakeJev(choice=ADD),
        executor=FakeExecutor(path="new.py", content="x = 1\n"),
        adder=adder,
        root=tmp_path,
    )
    async with app.run_test() as pilot:
        await drive_to_add(app, pilot)
        await pilot.press("ctrl+y")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert [proposal.path for proposal in adder.calls] == ["new.py"]
        assert not (tmp_path / "new.py").exists()


# -- the answer workflow ---------------------------------------------------


@pytest.fixture
def answer_app(tmp_path: Path) -> XGApp:
    """A project question: the origin reads the project, then sort answers it."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    return make_app(
        jev=FakeJev(
            choices={ORIGIN: SELECT_MODULE, SORT: ANSWER},
            kept={"a.py"},
            scores={"a.py": 0.9},
        ),
        executor=FakeExecutor(answer="xg is a small workflow runner"),
        root=tmp_path,
    )


async def test_a_question_reads_the_project_and_is_answered(answer_app: XGApp) -> None:
    """The whole point of the fork: a question does not become an edit."""
    async with answer_app.run_test() as pilot:
        await submit(answer_app, pilot, "what is this project?")
        await submit(answer_app, pilot, "go")
        await submit(answer_app, pilot, "go")
        await submit(answer_app, pilot, "go")
        await submit(answer_app, pilot, "go")
        assert answer_app.session.position == ANSWER
        assert answer_app.gate is None
        assert "small workflow runner" in state_text(answer_app)


async def test_the_answer_is_not_gated(answer_app: XGApp) -> None:
    """Reading proposes nothing, so there is nothing to accept or reject."""
    async with answer_app.run_test() as pilot:
        await submit(answer_app, pilot, "what is this project?")
        await submit(answer_app, pilot, "go")
        await submit(answer_app, pilot, "go")
        await submit(answer_app, pilot, "go")
        await submit(answer_app, pilot, "go")
        assert answer_app.gate is None
        assert "nothing has happened yet" not in status_text(answer_app)
