"""The command line: which lines are commands, and what each one asks for.

No app, no session, no client. A command parses into an intent and stops, and that
is the invariant these tests exist to hold: if a command ever grows the ability to
act on the world, it will be because one of these stopped being a plain function
call.
"""

from __future__ import annotations

import pytest

from xg_project.commands import (
    ALIASES,
    COMMANDS,
    PREFIX,
    Clear,
    Go,
    Help,
    Invalid,
    help_text,
    is_command,
    parse,
)

# -- command or prompt -----------------------------------------------------


def test_a_line_starting_with_a_slash_is_a_command() -> None:
    assert is_command("/go FILTER")


def test_a_line_that_does_not_is_a_prompt() -> None:
    assert not is_command("go FILTER")
    assert not is_command(":..")
    assert not is_command("")


def test_the_slash_must_come_first() -> None:
    """A prompt is free text, so a slash inside one cannot turn it into a command."""
    assert not is_command("what does src/a.py do?")
    assert not is_command(" a / b")


# -- what each command asks for --------------------------------------------


def test_go_asks_to_go_to_a_node() -> None:
    assert parse("/go FILTER") == Go(node="FILTER")


def test_go_names_the_parent_as_a_dot_dot() -> None:
    """``..`` is not a node; parsing does not care which names are nodes."""
    assert parse("/go ..") == Go(node="..")


def test_clear_asks_to_empty_the_line() -> None:
    assert parse("/clear") == Clear()


def test_help_asks_for_the_list() -> None:
    assert parse("/help") == Help()


# -- short spellings -------------------------------------------------------


@pytest.mark.parametrize(("short", "long"), sorted(ALIASES.items()))
def test_a_short_spelling_reaches_the_command_it_stands_for(short: str, long: str) -> None:
    """Not merely accepted: the same command, arguments and all."""
    arguments = " FILTER" if long == "go" else ""
    assert parse(f"{PREFIX}{short}{arguments}") == parse(f"{PREFIX}{long}{arguments}")


def test_every_alias_names_a_real_command() -> None:
    """A misspelt alias would silently be a command that does not exist."""
    assert set(ALIASES.values()) <= {command.name for command in COMMANDS}


def test_an_alias_is_not_also_a_command() -> None:
    """A spelling is not a second command, or everything would be listed twice."""
    names = {command.name for command in COMMANDS}
    assert not names & set(ALIASES)


# -- refusing a line -------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/nope", "No such command 'nope'. Try: /help"),
        ("/gof FILTER", "Did you mean 'go'?"),
        ("/go", "Missing argument 'NODE'. Try: /go <node>"),
        ("/go a b", "Got unexpected extra argument (b). Try: /go <node>"),
        ("/go --x", "No such option '--x'. Try: /go <node>"),
        ("/", "nothing after '/'. Try: /help"),
        ("/clear now", "Got unexpected extra argument (now). Try: /clear"),
    ],
)
def test_a_bad_command_line_is_refused_with_a_usable_message(line: str, expected: str) -> None:
    with pytest.raises(Invalid) as raised:
        parse(line)
    assert expected in str(raised.value)


def test_a_refusal_fits_on_the_status_line() -> None:
    """One line is all there is, so a message that wraps is one half shown."""
    for line in ("/nope", "/gof FILTER", "/go", "/go a b", "/"):
        with pytest.raises(Invalid) as raised:
            parse(line)
        message = str(raised.value)
        assert "\n" not in message
        assert message == message.strip()


def test_an_unclosed_quote_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(Invalid):
        parse('/go "unfinished')


def test_a_longer_name_that_starts_with_a_command_is_not_that_command() -> None:
    with pytest.raises(Invalid):
        parse("/gofoo")


# -- arguments -------------------------------------------------------------


def test_a_node_name_may_be_quoted() -> None:
    """Splitting with shlex is what makes this one argument instead of two."""
    assert parse('/go "a node with spaces"') == Go(node="a node with spaces")


def test_a_node_name_is_taken_as_typed() -> None:
    assert parse("/go   FILTER  ") == Go(node="FILTER")


# -- the help --------------------------------------------------------------


def test_the_help_lists_every_command() -> None:
    listed = help_text()
    for command in COMMANDS:
        assert f"{PREFIX}{command.name}" in listed


def test_the_help_shows_each_short_spelling_beside_its_command() -> None:
    listed = help_text()
    for short in ALIASES:
        assert f"(also {PREFIX}{short})" in listed


def test_the_help_says_what_each_command_does() -> None:
    listed = help_text()
    for command in COMMANDS:
        assert command.get_short_help_str(limit=100) in listed


def test_the_help_is_one_line_per_command() -> None:
    assert len(help_text().splitlines()) == len(COMMANDS)
