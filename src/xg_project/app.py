"""The xg TUI.

Two things are on screen at once, and each answers one question.

- **the graph** — the whole node tree, drawn from the origin, with the current
  node marked. This is the answer to "where can I go".
- **the state** — what each node introduced, keyed by that node's name. This is
  the answer to "where am I and what do I know", and it is the whole of the
  application's state: there is no transcript of actions to read past.

A single status line sits under the state for the one thing that is not state: a
proposal waiting to be accepted or rejected, or the sentence explaining why the
last action did not happen. It is replaced, not appended to, so nothing scrolls.
Under it is a spinner, shown only while a turn or an accepted action is in
flight, so a slow model reads as waiting rather than as a frozen screen.

The input line does two different things depending on its first character. A
``/command`` — ``/go FILTER`` to move, ``/help`` for the list — is answered by the
interface itself, through :mod:`xg_project.commands`. Anything else is a prompt,
submitted with Enter, and goes through :func:`take_turn`. The input is left as
typed after submitting; ``ctrl+l`` clears it, and so does ``/clear``.

The handler is ``async`` because a turn awaits Jev at a branching node and the
executor at a leaf. It does not await that turn itself: Textual awaits an async
message handler inline in the message pump, so awaiting a model call there would
freeze the interface until the model answered. The turn runs in a **worker**
instead, which leaves the pump free to keep drawing and handling keys, and the
same is true of an accepted proposal. Awaiting leaves the event loop free either
way, so the interface stays responsive while a request is in flight.

A **gated** leaf — ``_XG_COMMAND`` or ``_XG_EDIT`` — does not act on its own: the
proposal is shown and held, and the user accepts it with ``ctrl+y`` or rejects it
with ``ctrl+n``. Nothing runs and nothing is written until they accept.
``_XG_ANSWER`` is a leaf too, but it proposes nothing and changes nothing, so it
simply introduces its text into the state.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, LoadingIndicator, Static

from xg_project.commands import (
    PREFIX,
    Clear,
    Go,
    Help,
    Invalid,
    help_text,
    is_command,
    parse,
)
from xg_project.edit import EditOutcome, apply_edit
from xg_project.graph import Actor, Registry, default_graph
from xg_project.jev import Jev
from xg_project.llm import (
    ENV_API_KEY,
    PASS_ENTRY,
    CommandProposal,
    EditProposal,
    Executor,
)
from xg_project.session import Move, Session
from xg_project.shell import Outcome, run_command
from xg_project.turn import ACCEPTED, REJECTED, Gate, take_turn

PROMPT_ID = "prompt"
GRAPH_ID = "graph"
STATE_ID = "state"
STATE_SCROLL_ID = "state-scroll"
STATUS_ID = "status"
BUSY_ID = "busy"

MAX_OUTPUT_LINES = 24
"""How much of an accepted command's output the status line will show."""


class XGApp(App[None]):
    """The terminal interface to one xg session."""

    TITLE = "xg"

    CSS = """
    Screen {
        overflow-y: hidden;
    }
    #graph {
        height: auto;
        max-height: 60%;
        overflow: hidden;
        padding: 1 2 0 2;
    }
    #state-scroll {
        height: 1fr;
        min-height: 3;
        padding: 0 2 1 2;
        border-top: solid $primary;
    }
    #state {
        height: auto;
    }
    #status {
        height: auto;
        padding: 0 2;
        border-top: solid $primary;
    }
    #prompt {
        margin: 0 1;
    }
    #busy {
        height: 1;
        padding: 0 2;
        content-align: left middle;
        display: none;
    }
    """
    """The layout: two windows for the graph and the state, then the controls.

    The graph pane carries the trail as well, because both answer "where is this
    run" and neither may scroll out of sight. Its height is capped so a large
    graph cannot push the state pane — or the input line below it — off the
    screen; the screen itself does not scroll, so nothing can be pushed anywhere.
    The state pane is the only thing that scrolls, because it is the only thing
    that grows without bound.
    """

    BINDINGS = [
        Binding("escape", "focus_prompt", "prompt", show=False),
        Binding("ctrl+l", "clear_prompt", "clear input"),
        Binding("ctrl+u", "go_up", "up", priority=True),
        Binding("pageup", "state_up", "state ▲", priority=True),
        Binding("pagedown", "state_down", "state ▼", priority=True),
        Binding("ctrl+y", "accept_proposal", "accept", show=False),
        Binding("ctrl+n", "reject_proposal", "reject", show=False),
        Binding("ctrl+q", "quit", "quit"),
    ]

    def __init__(
        self,
        graph: Registry | None = None,
        jev: Jev | None = None,
        executor: Executor | None = None,
        runner: Callable[[str], Outcome] | None = None,
        editor: Callable[[EditProposal], EditOutcome] | None = None,
        root: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.graph = graph if graph is not None else default_graph()
        self.session = Session(self.graph)
        # Injectable so the tests can drive a turn to completion without a key or
        # a network round trip. Each driver builds its client lazily, on first
        # use, so xg opens and works with nothing configured.
        self.jev = jev if jev is not None else Jev()
        self.executor = executor if executor is not None else Executor()
        # The command runner and the edit applier are injectable so a test can
        # accept a proposal without starting a process or writing a file. Both are
        # run in a thread so a slow one does not block the interface.
        self.runner = runner if runner is not None else run_command
        self.editor = editor if editor is not None else self._apply_edit
        self.root = Path(root) if root is not None else Path.cwd()
        self.gate: Gate | None = None
        """The proposal awaiting accept or reject, or ``None`` when nothing is."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(id=GRAPH_ID)
        with VerticalScroll(id=STATE_SCROLL_ID):
            yield Static(id=STATE_ID)
        yield Static(id=STATUS_ID)
        yield LoadingIndicator(id=BUSY_ID)
        yield Input(placeholder="prompt, /go <node> to move, /help for commands", id=PROMPT_ID)
        yield Footer()

    # -- lifecycle ---------------------------------------------------------

    def on_mount(self) -> None:
        """Draw the opening state and put the cursor where the user will type."""
        hints = []
        if not os.environ.get("TYPESAFE_API_KEY", "").strip():
            hints.append("no TYPESAFE_API_KEY, so routing is off")
        if not os.environ.get(ENV_API_KEY, "").strip():
            hints.append(f"the executor reads `pass show {PASS_ENTRY}` when needed")
        self.set_status("[dim]" + " · ".join(hints) + "[/dim]" if hints else "")
        self.set_busy(False)
        self.refresh_state()
        self.action_focus_prompt()

    async def on_unmount(self) -> None:
        """Close Jev's and the executor's connection pools, if any were opened.

        Textual awaits an async ``on_unmount`` during shutdown, so the pools are
        closed before the loop stops rather than left to process teardown.
        """
        await self.jev.aclose()
        await self.executor.aclose()

    def action_focus_prompt(self) -> None:
        """Put focus back on the input line."""
        self.query_one(f"#{PROMPT_ID}", Input).focus()

    def action_clear_prompt(self) -> None:
        """Empty the input line and keep focus there.

        Submitting leaves the line as typed, so this is the only thing that
        empties it.
        """
        self.query_one(f"#{PROMPT_ID}", Input).value = ""
        self.action_focus_prompt()

    def action_go_up(self) -> None:
        """Move to the node above this one: the keyboard's version of ``/go ..``.

        The input line is left exactly as it was typed. Navigating does not clear
        it any more than submitting does — ``ctrl+l`` is still the only binding
        that empties it — which is why this takes ``ctrl+u`` as a priority
        binding: the input line would otherwise spend it deleting to the start of
        the line.
        """
        self.go_to("..")

    def go_to(self, node: str) -> None:
        """Move to a node, setting aside a proposal that was waiting where we were.

        Leaving a node behind sets aside the proposal waiting there, for the same
        reason submitting a line does: the proposal is not what the user is doing
        now, and accepting it afterwards would act on a node the run has already
        walked away from. A move that goes nowhere sets nothing aside, since
        nothing has been left behind.
        """
        moved = self.do_move(node)
        if moved is None:
            return
        if self._set_aside_pending():
            self.set_status(
                f"[dim]{moved.frm}[/dim] → [b]{moved.to}[/b] "
                "[yellow]· the pending proposal was set aside[/yellow]"
            )
        self.refresh_state()

    def action_state_up(self) -> None:
        """Scroll the state pane up a page.

        Bound to the keyboard rather than left to the mouse wheel, because the
        input line holds focus and the state is the one pane that moves. It is a
        priority binding because the input line declares its own page-up and
        page-down for scrolling its content, and a one-line prompt has nothing to
        scroll: without the priority, the key would be spent there.
        """
        self.query_one(f"#{STATE_SCROLL_ID}", VerticalScroll).scroll_page_up(animate=False)

    def action_state_down(self) -> None:
        """Scroll the state pane down a page."""
        self.query_one(f"#{STATE_SCROLL_ID}", VerticalScroll).scroll_page_down(animate=False)

    # -- rendering ---------------------------------------------------------

    def set_status(self, markup: str) -> None:
        """Replace the status line. It is never appended to."""
        self.query_one(f"#{STATUS_ID}", Static).update(markup)

    def set_busy(self, busy: bool) -> None:
        """Show or hide the spinner that marks a model or an action in flight.

        A turn can spend many seconds awaiting Jev or the executor, so the wait
        is drawn rather than left as a frozen screen. The row is hidden instead
        of emptied, so it takes no space when nothing is running.
        """
        self.query_one(f"#{BUSY_ID}", LoadingIndicator).display = busy

    def refresh_state(self) -> None:
        """Redraw both panes from the session.

        The graph pane gets the tree, the node the run is on, and the trail; the
        state pane gets the state and nothing else. Keeping the trail with the
        tree is what stops the "where am I" half of the display from scrolling
        away — the state is the only pane with a scrollbar, so anything that
        belongs to the other half has to live there.
        """
        session = self.session
        node = session.node
        self.query_one(f"#{GRAPH_ID}", Static).update(
            "\n".join(
                [
                    session.graph.render_tree(session.position),
                    f"[b]you are at[/b] [b cyan]{escape(node.name)}[/b cyan] "
                    f"[dim]· {escape(node.summary)}[/dim]",
                    f"[dim]trail:[/dim] {escape(' → '.join(session.trail))}",
                ]
            )
        )
        self.query_one(f"#{STATE_ID}", Static).update(self._state_lines())

    def _state_lines(self) -> str:
        """The state, one entry per node, with the node the run is on marked."""
        lines: list[str] = []
        if not self.session.state:
            lines.append("[dim]no state has been introduced yet[/dim]")
        for name, value in self.session.state.items():
            here = " [cyan]←[/cyan]" if name == self.session.position else ""
            lines.append(f"[b]{escape(name)}[/b]{here}  {self.describe(value)}")
        return "\n".join(lines)

    def describe(self, value: object) -> str:
        """One state entry as a line, dispatched on what the value is.

        Any value that knows how to render itself wins; the rest are read
        generically, so a node a user adds is legible without teaching the TUI
        about it.
        """
        preview = getattr(value, "preview", None)
        if callable(preview):
            text = escape(str(preview()))
            detail = getattr(value, "detail", None)
            if callable(detail):
                extra = str(detail())
                if extra:
                    text += "\n" + self._patch_markup(extra)
            return text
        if isinstance(value, str):
            return escape(value)
        if isinstance(value, Mapping):
            return self._describe_mapping(value)
        explain = getattr(value, "explain", None)
        if callable(explain):
            return escape(str(explain()))
        return escape(str(value))

    def _describe_mapping(self, value: Mapping) -> str:
        parts: list[str] = []
        if "module" in value:
            return self._describe_module(value)
        files = value.get("files")
        scores = value.get("scores")
        read = value.get("read")
        if isinstance(files, Mapping):
            if isinstance(read, int) and read > len(files):
                # Filter read one set and the model kept a smaller one; showing
                # both halves is the difference between "2 files" and "2 of 11".
                parts.append(f"{len(files)} of {read} files")
            else:
                parts.append(f"{len(files)} files")
            if files and isinstance(scores, Mapping):
                parts.append(f"top: {escape(str(next(iter(files))))}")
        dropped = value.get("dropped")
        if isinstance(dropped, (list, tuple)) and dropped:
            parts.append(f"[yellow]{len(dropped)} did not fit[/yellow]")
        problem = value.get("problem")
        if problem:
            parts.append(f"[yellow]{escape(str(problem))}[/yellow]")
        if parts:
            return " · ".join(parts)
        return escape(str(dict(value)))

    def _describe_module(self, value: Mapping) -> str:
        """The module selection step: which module, and how much it admits.

        Three states that look similar and are not: no module bounded the read,
        a module was chosen, and a module was chosen but admits nothing.
        """
        parts: list[str] = []
        module = value.get("module")
        if module:
            parts.append(f"{escape(str(module))} ({value.get('name') or 'unnamed'})")
            includes = value.get("includes")
            count = len(includes) if isinstance(includes, (list, tuple)) else 0
            parts.append(f"{count} files exposed" if count else "[yellow]exposes nothing[/yellow]")
        else:
            parts.append("no module · whole project in scope")
        missing = value.get("missing")
        if isinstance(missing, (list, tuple)) and missing:
            parts.append(f"[yellow]{len(missing)} declaration(s) resolved to nothing[/yellow]")
        problems = value.get("problems")
        if isinstance(problems, (list, tuple)) and problems:
            parts.append(f"[yellow]{len(problems)} unusable manifest(s)[/yellow]")
        problem = value.get("problem")
        if problem:
            parts.append(f"[yellow]{escape(str(problem))}[/yellow]")
        return " · ".join(parts)

    def _patch_markup(self, patch: str) -> str:
        """A patch for the state pane, indented and coloured by its own markers.

        The colour is decided by the first character of each line, which is the
        patch format's own signal: additions and removals read first, the headers
        and hunk markers stay dim, and context is left alone so the change is
        what the eye lands on. Each line is escaped on its own because the markup
        closes at the end of the line.
        """
        lines: list[str] = []
        for line in patch.splitlines():
            body = escape(line)
            if line.startswith(("+++", "---")):
                lines.append(f"  [b]{body}[/b]")
            elif line.startswith("+"):
                lines.append(f"  [green]{body}[/green]")
            elif line.startswith("-"):
                lines.append(f"  [red]{body}[/red]")
            elif line.startswith("@@"):
                lines.append(f"  [cyan]{body}[/cyan]")
            else:
                lines.append(f"  [dim]{body}[/dim]")
        return "\n".join(lines)

    # -- input -------------------------------------------------------------

    def _set_aside_pending(self) -> bool:
        """Drop a proposal the user has moved on from, and say whether there was one.

        Nothing runs: the proposal was never accepted. Marking it rejected rather
        than forgetting it is what keeps it from vanishing silently.
        """
        if self.gate is None:
            return False
        self.gate.status = REJECTED
        self.gate = None
        return True

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Route one line of input: a ``/command``, or a prompt.

        The input line is deliberately left as the user typed it. Only
        ``ctrl+l`` — and ``/clear`` — empties it, so a line that was routed badly
        can be edited and sent again without being retyped.
        """
        text = event.value.strip()
        if not text:
            return
        if is_command(text):
            self.run_command(text)
            return
        if text.startswith(":"):
            # The syntax this replaced. Saying so costs one line and saves a
            # surprise: read as a prompt, a stale ``:..`` would be sent to a model
            # as free text that it might act on.
            stale = text[1:].strip()
            instead = f"{PREFIX}go {stale}" if stale else f"{PREFIX}help"
            self.set_status(
                f"[yellow]commands start with '{PREFIX}' now. Try: {escape(instead)}[/yellow]"
            )
            return
        # A new line is a new intention, whether it turns out to be a move or a
        # prompt: the proposal was never accepted, so nothing runs.
        if self._set_aside_pending():
            self.set_status("[yellow]the pending proposal was set aside[/yellow]")
        # A turn waits on the network, and Textual awaits an async handler inline
        # in the message pump. Awaiting it here would freeze the app — no keys, no
        # moves — until the model answered. A worker runs it as its own task, so
        # the pump stays free and the spinner keeps animating.
        self.run_worker(self.do_prompt(text), group="turn", exclusive=True)

    def run_command(self, line: str) -> None:
        """Do what a command line asks for, or say why it cannot be done.

        Nothing here reaches the workflow: a command is answered by the interface,
        so no turn runs and no model is asked anything. Whether a command sets
        aside a waiting proposal is the command's own business — ``/go`` leaves a
        node behind and so does, while ``/clear`` and ``/help`` change nothing and
        so leave the proposal exactly where it was.
        """
        try:
            intent = parse(line)
        except Invalid as problem:
            self.set_status(f"[red]{escape(str(problem))}[/red]")
            return
        match intent:
            case Go(node=node):
                self.go_to(node)
            case Clear():
                self.action_clear_prompt()
            case Help():
                # A notification rather than the status line: this is several
                # lines of reference, and neither of the two windows is the place
                # for something that is not what the run is doing.
                self.notify(help_text(), title=f"{PREFIX}help", markup=False, timeout=15)

    def do_move(self, name: str) -> Move | None:
        """Move to a named node, reporting a refusal rather than raising.

        ``..`` does not name a node; it resolves to the parent of the current
        one, and is the manual way up that mirrors a shell's ``cd ..``.

        Returns the move when one happened and ``None`` when none did — a
        refusal, an empty name, or ``..`` at the origin — so a caller can tell
        whether the run went anywhere without reading the status line back.
        """
        if not name:
            self.set_status(f"[red]expected a node name. Try: {PREFIX}go <node>[/red]")
            return None

        if name == "..":
            parent = self.graph.parent(self.session.position)
            if parent is None:
                self.set_status(
                    f"[red]already at the origin {escape(self.session.position)}: "
                    "nothing above it[/red]"
                )
                return None
            name = parent

        moved = self.session.move(name, Actor.USER)
        if not moved.ok:
            self.set_status(f"[red]cannot move to {escape(name)}: {moved.reason}[/red]")
            return None
        self.set_status(f"[dim]{moved.frm}[/dim] → [b]{moved.to}[/b]")
        return moved

    async def do_prompt(self, prompt: str) -> None:
        """Run a prompt at the current node, and report where it went.

        A proposing node hands back a gate instead of acting, and that is shown
        as the pending proposal. Everything else is reported on the status line:
        a move, a routing failure, or a node that introduced state but went
        nowhere.
        """
        self.set_busy(True)
        try:
            turn = await take_turn(
                self.session,
                prompt,
                jev=self.jev,
                executor=self.executor,
                root=self.root,
            )
        finally:
            self.set_busy(False)

        try:
            if turn.gate is not None:
                self.show_gate(turn.gate)
                return

            if turn.routing is not None and not turn.routing.ok:
                self.set_status(f"[yellow]not routed:[/yellow] {escape(turn.routing.explain())}")
                return

            if turn.moved is not None and turn.moved.ok:
                destination = f"[dim]{turn.moved.frm}[/dim] → [b]{turn.moved.to}[/b]"
                if turn.routing is not None:
                    confidence = escape(turn.routing.explain())
                    destination += f" [dim]· {confidence}[/dim]"
                self.set_status(destination)
                return

            if turn.moved is not None:  # pragma: no cover - options() pre-filters these
                self.set_status(f"[red]move refused:[/red] {escape(turn.moved.reason or '')}")
                return

            self.set_status(f"[dim]{escape(turn.node)} introduced state[/dim]")
        finally:
            self.refresh_state()

    # -- the gate ----------------------------------------------------------

    def show_gate(self, gate: Gate) -> None:
        """Show what the executor proposed, and hold it for accept or reject.

        A proposal with a problem is reported and nothing is held: there is no
        action to decide about.
        """
        proposal = gate.proposal
        if not proposal.ok:
            self.gate = None
            self.set_status(f"[yellow]nothing proposed:[/yellow] {escape(proposal.problem or '')}")
            return

        self.gate = gate
        self.set_status(
            "[yellow]nothing has happened yet.[/yellow] [dim]ctrl+y[/dim] accept "
            "[dim]·[/dim] [dim]ctrl+n[/dim] reject"
        )

    def action_accept_proposal(self) -> None:
        """Kick off the pending proposal, in a worker so the interface stays live.

        The gate is cleared here, synchronously, rather than when the worker
        starts: a second ``ctrl+y`` must not find a proposal that is already
        being carried out.
        """
        gate = self.gate
        if gate is None:
            return
        self.gate = None
        self.run_worker(self._accept_proposal(gate), group="action", exclusive=True)

    async def _accept_proposal(self, gate: Gate) -> None:
        """Carry out the proposal: a command or an edit, in a thread."""
        proposal = gate.proposal
        if isinstance(proposal, EditProposal):
            action, argument = self.editor, proposal
        elif isinstance(proposal, CommandProposal):
            action, argument = self.runner, proposal.command or ""
        else:  # pragma: no cover - every proposing node yields one of the two
            self.set_status("[red]the pending proposal cannot be carried out[/red]")
            return

        # A command or an edit runs in a thread; either can take a while, so the
        # wait is drawn the same way a model call is.
        self.set_busy(True)
        try:
            outcome: object = await asyncio.to_thread(action, argument)
        finally:
            self.set_busy(False)

        gate.status = ACCEPTED
        gate.outcome = outcome
        self.set_status(self.describe_outcome(outcome))
        self.refresh_state()

    def action_reject_proposal(self) -> None:
        """Drop the pending proposal without carrying it out."""
        gate = self.gate
        if gate is None:
            return
        self.gate = None
        gate.status = REJECTED
        self.set_status("[yellow]rejected[/yellow]")
        self.refresh_state()

    def _apply_edit(self, proposal: EditProposal) -> EditOutcome:
        """The default editor: write the edit into the file, relative to the root."""
        return apply_edit(proposal, root=self.root)

    def describe_outcome(self, outcome: object) -> str:
        """The status line for an accepted proposal's result."""
        if isinstance(outcome, Outcome):
            return self._describe_run(outcome)
        explain = getattr(outcome, "explain", None)
        if callable(explain):
            return escape(str(explain()))
        return escape(str(outcome))

    def _describe_run(self, outcome: Outcome) -> str:
        """A command's output, labelled, and bounded so it cannot flood the view."""
        if outcome.problem is not None:
            return f"[red]{escape(outcome.problem)}[/red]"

        lines: list[str] = []
        for line in _lines(outcome.stdout):
            lines.append(f"[dim]│[/dim] {escape(line)}")
        for line in _lines(outcome.stderr):
            lines.append(f"[red]│[/red] {escape(line)}")
        if len(lines) > MAX_OUTPUT_LINES:
            hidden = len(lines) - MAX_OUTPUT_LINES
            lines = lines[:MAX_OUTPUT_LINES] + [f"[dim]… {hidden} more lines[/dim]"]
        colour = "green" if outcome.code == 0 else "red"
        lines.append(f"[{colour}]exit {outcome.code}[/{colour}]")
        return "\n".join(lines)


def _lines(text: str) -> list[str]:
    """The lines of captured output, ignoring an empty result and a trailing newline."""
    if not text:
        return []
    return text.rstrip("\n").split("\n")


def main() -> None:
    """Run the TUI."""
    XGApp().run()


if __name__ == "__main__":
    main()
