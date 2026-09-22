"""The command line: ``/go FILTER``, and the rest of what the interface is asked.

A line typed at the prompt is one of two things, and its first character is what
tells them apart. Without a ``/`` it is a **prompt**: free text, handed to the
workflow, which may end in a proposal to change a file. With a ``/`` it is a
**command**: it asks the interface for something — go here, empty the line, list
what there is — and the interface answers it itself, with no model in the loop and
nothing left to accept.

That is why commands live apart from ``app``, and why they do not *act*. A command
parses into a small frozen **intent** and returns it; the interface decides what
touching itself means. So no command here can reach the network, write a file, or
await anything, and the whole syntax can be tested without a terminal.

Splitting the line and checking the arguments is click's job. Click's other life is
writing command-line programs, so what it brings is what a hand-rolled splitter
gets wrong: a missing or extra argument is refused with a sentence a person can
read, the help is generated from the same declaration that does the enforcement,
and the set of commands exists in one place to be listed. It is driven in-process
through ``standalone_mode=False``, the documented way to embed it — without that,
click answers a bad line by printing to the terminal and calling ``sys.exit``,
which inside a full-screen interface would scribble over the display and take the
app down with it.

Making ``/`` the prefix means a prompt cannot begin with one. That is a real cost,
paid deliberately: the alternative is guessing, and a mistyped command guessed to
be a prompt is a line sent to a model that may propose an edit. A line that begins
with ``/`` and names no command is always reported as a bad command, never quietly
re-read as prose.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

import click

PREFIX = "/"
"""What makes a line a command instead of a prompt."""

PROGRAM = "xg"
"""What click calls this program in the messages it writes."""


@dataclass(frozen=True)
class Go:
    """Go to a node. ``..`` means the node above this one, as in a shell."""

    node: str


@dataclass(frozen=True)
class Clear:
    """Empty the prompt line. The same thing ``ctrl+l`` does."""


@dataclass(frozen=True)
class Help:
    """List the commands."""


Intent = Go | Clear | Help
"""What the interface can be asked to do, as a value rather than as a call."""


class Invalid(Exception):
    """A line that is a command but not one that can be obeyed.

    Raised for a name no command answers to and for arguments that do not fit.
    Deliberately not click's ``UsageError``: nothing above this module should have
    to know that click is the parser, and the message is already shaped as the one
    line the status line has room for.
    """


# -- the commands ---------------------------------------------------------
#
# Each one returns an intent. None of them takes the app, the session, or a
# client, and none of them is ``async``: that a command cannot act on the world is
# the point rather than an accident, so there is nothing here to await.


@click.command(name="go", add_help_option=False)
@click.argument("node")
def _go(node: str) -> Go:
    """Go to a node, or to the node above this one with `/go ..`."""
    return Go(node=node)


@click.command(name="clear", add_help_option=False)
def _clear() -> Clear:
    """Empty the prompt line, the same as ctrl+l."""
    return Clear()


@click.command(name="help", add_help_option=False)
def _help() -> Help:
    """List the commands and their short names."""
    return Help()
COMMANDS: tuple[click.Command, ...] = (_go, _clear, _help)
"""Every command, in the order they are listed to the user.

Each docstring is a single sentence on purpose. Click shortens a command's help
for a listing by cutting its first paragraph at the first full stop, so a second
sentence would be read as the long help, shown nowhere, and silently dropped from
the one-line summary.
"""

ALIASES: dict[str, str] = {"g": "go", "c": "clear", "?": "help"}
"""Short spellings, mapped to the name of the command each one stands for.

Kept beside the commands rather than on each command, because click has no alias
attribute to put on one: its own documentation's answer to "how do I make an
alias" is to override the group, which is what `_Group` below does with this
table. An alias is a way of spelling a command rather than a second command, so it
is resolved on the way in and never appears as a command of its own.
"""


class _Group(click.Group):
    """A click group that also answers to the short spellings in `ALIASES`."""

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Resolve a name click does not know to the command an alias names.

        Looked up after click's own resolution and only when that found nothing, so
        a real command always wins over an alias that happens to collide with it.
        """
        found = super().get_command(ctx, cmd_name)
        if found is not None:
            return found
        return self.commands.get(ALIASES.get(cmd_name, ""))


_CLI = _Group(
    name=PROGRAM,
    commands={command.name: command for command in COMMANDS},
    # A group handed no arguments prints its help. For a command-line program that
    # is a kindness; here it would write to stdout, underneath the interface.
    no_args_is_help=False,
    add_help_option=False,
)


def is_command(line: str) -> bool:
    """Whether a line asks the interface for something rather than the workflow."""
    return line.startswith(PREFIX)


def _usage(command: click.Command) -> str:
    """How a command line is spelled, for a message with one line to spare.

    Built here rather than asked of click, whose own usage is written for a shell:
    it names the program, adds an ``[OPTIONS]`` there are none of, and wraps. What
    is wanted beside the reason a line was refused is short enough to read at a
    glance.
    """
    parts = [f"{PREFIX}{command.name}"]
    parts += [
        f"<{parameter.name}>" for parameter in command.params if isinstance(parameter, click.Argument)
    ]
    return " ".join(parts)


def _refusal(problem: click.UsageError) -> Invalid:
    """Turn click's refusal into one line that names what to type instead.

    The fault is click's message verbatim, since it already names the argument or
    the option at fault; only its punctuation is made to fit the sentence the
    usage is joined onto, which is not always there — click ends some of these
    messages and leaves others open.
    """
    message = problem.format_message()
    if not message.endswith((".", "!", "?")):
        message += "."
    # A name nothing answers to leaves no command whose usage could be shown, and
    # click's context there belongs to the group rather than to a command.
    if isinstance(problem, click.NoSuchCommand):
        return Invalid(f"{message} Try: {PREFIX}help")
    command = problem.ctx.command if problem.ctx is not None else None
    if command is None or command is _CLI:
        return Invalid(message)
    return Invalid(f"{message} Try: {_usage(command)}")


def parse(line: str) -> Intent:
    """Work out what a command line asks for.

    ``line`` includes the leading ``/``. Raises `Invalid` if it names no command,
    if it names one whose arguments do not fit, or if there is nothing after the
    slash.

    That this much is the *whole* of what happens here is the invariant: commands
    return intents rather than acting, so obeying one is the same event as reading
    it. The words are split with `shlex`, which is what lets a node name be quoted
    and turns an unbalanced quote into a refusal rather than into a strange
    argument; click then does the part worth having it for — finding the command,
    checking the arguments against how it was declared, and calling it.
    """
    try:
        words = shlex.split(line[len(PREFIX) :])
    except ValueError as problem:
        raise Invalid(f"cannot read that line: {problem}") from problem

    if not words:
        raise Invalid(f"nothing after '{PREFIX}'. Try: {PREFIX}help")

    try:
        return _CLI.main(
            words,
            prog_name=PROGRAM,
            # Never a shell asking for tab completion: this line was typed into a
            # full-screen app. Left alone, click looks for the environment
            # variable that means "completion request" and exits the process.
            complete_var="",
            # False returns the command's value instead of ending the process,
            # which is the reason click can be driven from in here at all.
            standalone_mode=False,
        )
    except click.UsageError as problem:
        raise _refusal(problem) from problem


def help_text() -> str:
    """Every command with its short spellings, as lines to show the user.

    Built from the same commands click dispatches to, so a command cannot be
    registered and left out of the help, and a command that is renamed is renamed
    in both places at once.
    """
    lines = []
    for command in COMMANDS:
        also = [f"{PREFIX}{alias}" for alias, target in ALIASES.items() if target == command.name]
        spelled = f" (also {', '.join(also)})" if also else ""
        lines.append(f"{_usage(command)}{spelled} — {command.get_short_help_str(limit=100)}")
    return "\n".join(lines)
