"""Running an accepted command: the process seam, and how each failure reads.

Nothing here starts a real process. `subprocess.run` is stubbed so the tests
assert on how a result is turned into an `Outcome`, which is the only logic this
module has.
"""

from __future__ import annotations

import subprocess

from xg_project.shell import DEFAULT_TIMEOUT, Outcome, run_command


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["sh", "-c", "x"], returncode, stdout, stderr)


def test_a_successful_run_carries_stdout_and_a_zero_code(monkeypatch) -> None:
    monkeypatch.setattr(
        "xg_project.shell.subprocess.run", lambda cmd, **kwargs: completed(stdout="hi\n")
    )
    outcome = run_command("echo hi")
    assert outcome.ok
    assert outcome.command == "echo hi"
    assert outcome.code == 0
    assert outcome.stdout == "hi\n"
    assert outcome.problem is None


def test_a_nonzero_exit_is_not_ok_but_is_not_a_problem(monkeypatch) -> None:
    """The command ran; it just failed. That is a result, not an xg fault."""
    monkeypatch.setattr(
        "xg_project.shell.subprocess.run", lambda cmd, **kwargs: completed(stderr="boom\n", returncode=2)
    )
    outcome = run_command("false")
    assert not outcome.ok
    assert outcome.problem is None
    assert outcome.code == 2
    assert outcome.stderr == "boom\n"
    assert outcome.explain() == "exit 2"


def test_the_command_runs_in_a_shell_so_pipes_keep_their_meaning(monkeypatch) -> None:
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen.update(kwargs)
        return completed()

    monkeypatch.setattr("xg_project.shell.subprocess.run", fake_run)
    run_command("echo hi | wc -l")
    assert seen["cmd"] == "echo hi | wc -l"
    assert seen["shell"] is True


def test_a_timeout_is_a_problem_not_a_raise(monkeypatch) -> None:
    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, DEFAULT_TIMEOUT)

    monkeypatch.setattr("xg_project.shell.subprocess.run", hang)
    outcome = run_command("sleep 999")
    assert not outcome.ok
    assert outcome.code is None
    assert "did not finish" in outcome.problem


def test_a_missing_binary_is_a_problem_not_a_raise(monkeypatch) -> None:
    def boom(cmd, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr("xg_project.shell.subprocess.run", boom)
    outcome = run_command("nope")
    assert not outcome.ok
    assert "could not run" in outcome.problem


def test_explain_reads_the_problem_when_there_is_one() -> None:
    assert Outcome(command="x", problem="boom").explain() == "boom"
    assert Outcome(command="x", code=0).explain() == "exit 0"
