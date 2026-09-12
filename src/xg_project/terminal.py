"""Terminal utilities used by the xg harness."""


def type_into_pty(child, text: str, *, submit: bool = False) -> None:
    """Type text into a pexpect PTY, optionally followed by Enter.

    Keeping this separate lets future command completion/autocomplete code use
    the same PTY input primitive without coupling it to the interface loop.
    """
    child.send(text)
    if submit:
        child.sendline("")
