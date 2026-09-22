"""Running the command a proposal names.

This is the only place in xg that starts a process, and it exists because the
``command`` node's whole purpose is to run a shell command. It is kept apart from
:mod:`xg_project.llm`, which proposes a command, so that "ask a model" and "run a
process" are two modules with two failure modes rather than one blurry one.

The safety model is the *gate*, not this function. A proposal is shown to the
user and this runs only after they accept it. ``shell=True`` is deliberate: the
model proposes a shell command line, the same string a user would type at a
prompt, and re-splitting it into an argv would silently change its meaning for
pipes, redirection, and quoting. There is no allowlist here because the user's
acceptance is the allowlist.

Failure is a value, like everywhere else: a timeout and a missing binary come
back as an :class:`Outcome` with ``problem`` set rather than raising.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

DEFAULT_TIMEOUT = 120.0
"""Seconds a command may run before it is killed.

Long enough for a build or a test run, short enough that a command that waits on
input does not wedge the session.
"""


@dataclass(frozen=True)
class Outcome:
    """What running a command produced, or why it could not be run.

    ``code`` is the process exit status. It is ``None`` only when the process
    never ran to a status — a timeout or a spawn failure — which is exactly when
    ``problem`` is set.
    """

    command: str
    code: int | None = None
    stdout: str = ""
    stderr: str = ""
    problem: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the command ran and exited zero."""
        return self.problem is None and self.code == 0

    def explain(self) -> str:
        """A one-line account of the run, for the log."""
        if self.problem is not None:
            return self.problem
        return f"exit {self.code}"


def run_command(command: str, *, timeout: float = DEFAULT_TIMEOUT, cwd: str | None = None) -> Outcome:
    """Run ``command`` in a shell and capture its output.

    The command's own environment is inherited. ``cwd`` is where it runs; the
    default is the process's, which for the TUI is the project xg was started in.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return Outcome(command=command, problem=f"the command did not finish within {timeout:.0f}s")
    except OSError as error:
        return Outcome(command=command, problem=f"could not run the command: {error}")

    return Outcome(
        command=command,
        code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
