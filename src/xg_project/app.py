"""The xg TUI.

Three things are on screen at once, and each answers one question.

- **the graph** — the whole node tree, drawn from the origin, with the current
  node marked. This is the answer to "where can I go".
- **the trail** — where the user is, how they got there, the goal stated at the
  origin, and how much context has been accumulated. This is the answer to
  "where am I".
- **the log** — what happened, in order. This is the answer to "what did that
  do".

The input line does two different things depending on its first character. A
prompt is submitted with Enter and goes through :func:`take_turn`. A ``:name`` is
a move to that node, which is how the user reaches a specific node directly
rather than being routed there.

The handler is ``async`` because a turn at a decision node awaits Jev. Awaiting
leaves the event loop free, so the interface stays responsive while the request
is in flight; a synchronous client here would freeze the screen for the length of
a round trip.
"""

from __future__ import annotations

import os

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, RichLog, Static
from rich.markup import escape

from xg_project.graph import Actor, NodeKind, Registry, default_graph
from xg_project.jev import Jev
from xg_project.session import Session
from xg_project.turn import take_turn

PROMPT_ID = "prompt"
GRAPH_ID = "graph"
TRAIL_ID = "trail"
LOG_ID = "log"


class XGApp(App[None]):
    """The terminal interface to one xg session."""

    TITLE = "xg"

    CSS = """
    #graph {
        height: auto;
        padding: 1 2 0 2;
    }
    #trail {
        height: auto;
        padding: 0 2 1 2;
        border-bottom: solid $primary;
    }
    #log {
        height: 1fr;
        margin: 1 1 0 1;
    }
    #prompt {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "focus_prompt", "prompt", show=False),
        Binding("ctrl+q", "quit", "quit"),
    ]

    def __init__(self, graph: Registry | None = None, jev: Jev | None = None) -> None:
        super().__init__()
        self.graph = graph if graph is not None else default_graph()
        self.session = Session(self.graph)
        # Injectable so the tests can drive a turn to completion without a key or
        # a network round trip. The default builds its client lazily, on the
        # first decision, so xg opens and works with nothing configured.
        self.jev = jev if jev is not None else Jev()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(id=GRAPH_ID)
        yield Static(id=TRAIL_ID)
        yield RichLog(id=LOG_ID, markup=True, wrap=True, max_lines=2000)
        yield Input(placeholder="prompt, or :node to move", id=PROMPT_ID)
        yield Footer()

    # -- lifecycle ---------------------------------------------------------

    def on_mount(self) -> None:
        """Draw the opening state and put the cursor where the user will type."""
        self.log_line("[dim]xg — deterministic agents[/dim]")
        if os.environ.get("TYPESAFE_API_KEY", "").strip():
            self.log_line("[dim]Jev is configured. Routing happens at decision nodes.[/dim]")
        else:
            self.log_line(
                "[yellow]TYPESAFE_API_KEY is not set, so Jev cannot route.[/yellow] "
                "[dim]You can still move between nodes by hand with :node.[/dim]"
            )
        self.log_line("")
        self.describe(self.session.node.name)
        self.refresh_state()
        self.action_focus_prompt()

    async def on_unmount(self) -> None:
        """Close Jev's connection pool, if one was opened.

        Textual awaits an async ``on_unmount`` during shutdown, so the pool is
        closed before the loop stops rather than left to process teardown.
        """
        await self.jev.aclose()

    def action_focus_prompt(self) -> None:
        """Put focus back on the input line."""
        self.query_one(f"#{PROMPT_ID}", Input).focus()

    # -- rendering ---------------------------------------------------------

    def log_line(self, markup: str) -> None:
        """Append one line to the log."""
        self.query_one(f"#{LOG_ID}", RichLog).write(markup)

    def refresh_state(self) -> None:
        """Redraw the graph and the trail from the session's current position."""
        session = self.session
        self.query_one(f"#{GRAPH_ID}", Static).update(
            session.graph.render_tree(session.position)
        )

        node = session.node
        kind = (
            "decision — Jev routes from here, and this adds to the context"
            if node.kind is NodeKind.DECISION
            else "generative — an LLM answers here, from the context"
        )
        trail = " → ".join(session.trail)
        lines = [
            f"[b]you are at[/b] [b cyan]{node.name}[/b cyan] "
            f"[dim]· level {node.level} · {kind}[/dim]",
            f"[dim]trail:[/dim] {escape(trail)}",
        ]
        if session.goal is not None:
            lines.append(f"[dim]goal:[/dim] {escape(session.goal)}")
        if session.context:
            lines.append(
                f"[dim]context:[/dim] {len(session.context)} "
                f"{'entry' if len(session.context) == 1 else 'entries'}"
            )
        self.query_one(f"#{TRAIL_ID}", Static).update("\n".join(lines))

    def describe(self, name: str) -> None:
        """Log a one-line description of a node, read from its declaration."""
        node = self.graph.get(name)
        self.log_line(f"[b cyan]{node.name}[/b cyan] [dim]·[/dim] {node.summary}")

    # -- input -------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Route one line of input: a ``:node`` is a move, anything else a prompt."""
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith(":"):
            self.do_move(text[1:].strip())
        else:
            await self.do_prompt(text)
        self.refresh_state()

    def do_move(self, name: str) -> None:
        """Move to a named node, reporting a refusal rather than raising."""
        if not name:
            self.log_line("[red]expected a node name after ':'[/red]")
            return

        moved = self.session.move(name, Actor.USER)
        if not moved.ok:
            self.log_line(f"[red]cannot move to {escape(name)}: {moved.reason}[/red]")
            return

        self.log_line("")
        self.log_line(f"[dim]{moved.frm}[/dim] → [b]{moved.to}[/b]")
        self.describe(moved.to)

    async def do_prompt(self, prompt: str) -> None:
        """Send a prompt to the current node, follow through on its kind, and log it.

        A generative node answers, and that ends the turn. A decision node
        contributes to the context window and hands the choice of the next node to
        Jev, so what gets logged is a destination and the confidence behind it —
        including when Jev could not answer, which leaves the session where it
        was and says why.
        """
        turn = await take_turn(self.session, self.jev, prompt)
        run = turn.run

        self.log_line("")
        self.log_line(f"[dim]you →[/dim] {escape(run.prompt)}")

        if run.kind is NodeKind.GENERATIVE:
            self.log_line(f"[b cyan]{run.node}[/b cyan] → {escape(run.result)}")
            self.log_line(
                "[dim]the node returned this text; the executor that reads it and "
                "acts is the last step and is not wired.[/dim]"
            )
            return

        self.log_line(f"[b cyan]{run.node}[/b cyan] [dim]contributed:[/dim] {escape(run.result)}")
        routing = turn.routing
        if routing is None:  # pragma: no cover - a decision node always routes
            return
        if not routing.ok:
            self.log_line(f"[yellow]not routed:[/yellow] {escape(routing.explain())}")
            return

        moved = turn.moved
        self.log_line(
            f"[dim]jev →[/dim] [b]{escape(routing.explain())}[/b]"
            + (f" [dim]via {escape(routing.model)}[/dim]" if routing.model else "")
        )
        if moved is not None and moved.ok:
            self.log_line(f"[dim]{moved.frm}[/dim] → [b]{moved.to}[/b]")
            self.describe(moved.to)
        elif moved is not None:  # pragma: no cover - options() pre-filters these
            self.log_line(f"[red]move refused:[/red] {escape(moved.reason or '')}")


def main() -> None:
    """Run the TUI."""
    XGApp().run()


if __name__ == "__main__":
    main()
