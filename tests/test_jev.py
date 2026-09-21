"""The Jev driver: the question it asks, and what it does with each answer.

Nothing here touches the network. The SDK's own response types are constructed
directly, so the tests exercise the real decoding surface — `.choices`,
`.choice`, `.confidence` — rather than a stand-in that could drift from it.
"""

from __future__ import annotations

import subprocess

import httpx2
import pytest
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    NoulAnswer,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeError,
    Usage,
)

from xg_project.jev import NEXT_NODE, Jev, api_key_from_pass, build_state

OPTIONS = {"command": "run a self-contained instruction", "plan": "write a plan"}


def response(*, choice: str, confidence: float = 0.9, model: str = "jev-latest") -> SystemOneResponse:
    """A real `SystemOneResponse` with one choice answer named ``next_node``."""
    answer = ChoiceAnswer(choice=choice, confidence=confidence, probabilities={choice: confidence})
    return SystemOneResponse(model=model, usage=Usage(), answers={NEXT_NODE: answer})


class FakeClient:
    """Stands in for `AsyncTypeSafeClient`, recording what it was asked."""

    def __init__(self, *, result=None, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[dict] = []
        self.closed = False

    async def system_one(self, *, state, questions):
        self.calls.append({"state": state, "questions": questions})
        if self.raises is not None:
            raise self.raises
        return self.result

    async def aclose(self) -> None:
        self.closed = True


async def decide(jv: Jev, **overrides) -> object:
    kwargs = {
        "current": "none",
        "context": ["the user wants: do the thing"],
        "prompt": "do the thing",
        "options": OPTIONS,
    }
    return await jv.decide(**{**kwargs, **overrides})


@pytest.fixture(autouse=True)
def no_password_store(monkeypatch):
    """Keep tests off the real `pass` store.

    Without this, any test that builds a bare ``Jev()`` would shell out to `pass`
    and could read — and then print in a failure diff — a real key. Tests that
    exercise key resolution patch this themselves.
    """
    monkeypatch.setattr("xg_project.jev.api_key_from_pass", lambda entry="jev": None)


# -- the happy path --------------------------------------------------------


async def test_a_routing_carries_the_chosen_node_and_its_confidence() -> None:
    routing = await decide(Jev(client=FakeClient(result=response(choice="command", confidence=0.91))))
    assert routing.ok
    assert routing.node == "command"
    assert routing.confidence == pytest.approx(0.91)
    assert routing.model == "jev-latest"
    assert routing.problem is None


async def test_the_question_offers_exactly_the_criteria() -> None:
    """The option set Jev is asked about is the one the graph handed over."""
    client = FakeClient(result=response(choice="command"))
    await decide(Jev(client=client))

    questions = client.calls[0]["questions"]
    assert set(questions) == {NEXT_NODE}
    question = questions[NEXT_NODE]
    assert isinstance(question, Choice)
    assert question.criteria == OPTIONS
    assert question.instructions


async def test_the_state_names_the_current_node_the_context_and_the_prompt() -> None:
    client = FakeClient(result=response(choice="command"))
    await decide(Jev(client=client))
    state = client.calls[0]["state"]
    joined = "\n".join(state)

    assert "currently at node: none" in joined
    assert "the user wants: do the thing" in joined
    assert "the user says: do the thing" in joined


def test_build_state_puts_the_goal_first_and_does_not_repeat_the_prompt() -> None:
    """Ordering is the meaning: the goal comes first, later entries were added by
    nodes further down. A prompt already in the context is not repeated."""
    state = build_state(current="none", context=["the user wants: x"], prompt="x")
    assert state[0] == "currently at node: none"
    assert state[1] == "the user wants: x"
    assert state[-1] == "the user says: x"


def test_build_state_appends_a_prompt_the_context_does_not_have() -> None:
    state = build_state(current="none", context=["the user wants: x"], prompt="then y")
    assert state[-1] == "the user says: then y"


# -- refusals, none of which are exceptions ---------------------------------


async def test_no_options_is_answered_without_a_request() -> None:
    """A leaf has no children, so there is nothing to ask and nothing to spend."""
    client = FakeClient(result=response(choice="command"))
    routing = await decide(Jev(client=client), options={})
    assert not routing.ok
    assert "no nodes to route to" in routing.problem
    assert client.calls == []


async def test_a_label_outside_the_offered_set_is_refused() -> None:
    """Accepting it would hide a schema problem behind a normal-looking move."""
    routing = await decide(Jev(client=FakeClient(result=response(choice="invented"))))
    assert not routing.ok
    assert "not offered" in routing.problem
    assert "invented" in routing.problem


async def test_an_answer_that_is_not_a_choice_is_refused() -> None:
    """Asking for a `Choice` and getting a `Noul` back means the request was wrong."""
    only_noul = SystemOneResponse(
        model="jev-latest",
        usage=Usage(),
        answers={NEXT_NODE: NoulAnswer(noul=0.5)},
    )
    routing = await decide(Jev(client=FakeClient(result=only_noul)))
    assert not routing.ok
    assert "no answer to the routing question" in routing.problem


async def test_a_missing_answer_is_refused() -> None:
    empty = SystemOneResponse(model="jev-latest", usage=Usage(), answers={})
    routing = await decide(Jev(client=FakeClient(result=empty)))
    assert not routing.ok


async def test_an_api_error_becomes_a_problem_not_a_raise() -> None:
    error = TypeSafeAPIError(500, {"error": "boom"}, httpx2.Headers())
    routing = await decide(Jev(client=FakeClient(raises=error)))
    assert not routing.ok
    assert "rejected the request" in routing.problem


async def test_a_connection_error_becomes_a_problem_not_a_raise() -> None:
    routing = await decide(Jev(client=FakeClient(raises=TypeSafeAPIConnectionError("timed out"))))
    assert not routing.ok
    assert "could not reach Jev" in routing.problem


async def test_a_missing_api_key_becomes_a_problem_not_a_raise(monkeypatch) -> None:
    """xg opens and is usable with no key configured; routing is what fails."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    routing = await decide(Jev())
    assert not routing.ok
    assert "not configured" in routing.problem


# -- lifecycle -------------------------------------------------------------


async def test_close_closes_a_client_it_built(monkeypatch) -> None:
    """A client this object built is a client this object closes.

    The SDK client is stubbed rather than keyed: construction succeeding is the
    thing under test, and a real client would reach the network.
    """
    built: list = []

    class StubClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.closed = False
            built.append(self)

        async def system_one(self, **kwargs):
            return response(choice="command")

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr("xg_project.jev.AsyncTypeSafeClient", StubClient)

    jv = Jev(model="jev-latest")
    assert (await decide(jv)).ok
    assert len(built) == 1
    assert built[0].kwargs == {"api_key": None, "model": "jev-latest"}

    await jv.aclose()
    assert built[0].closed is True
    assert jv._client is None


async def test_one_client_serves_every_decision(monkeypatch) -> None:
    """The pool is built once and reused, not rebuilt per decision."""
    built: list = []

    class StubClient:
        def __init__(self, **kwargs) -> None:
            built.append(self)

        async def system_one(self, **kwargs):
            return response(choice="command")

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr("xg_project.jev.AsyncTypeSafeClient", StubClient)

    jv = Jev()
    await decide(jv)
    await decide(jv)
    assert len(built) == 1


async def test_close_leaves_an_injected_client_alone() -> None:
    """Passing a client in transfers ownership of closing it to the caller."""
    client = FakeClient(result=response(choice="command"))
    jv = Jev(client=client)
    await decide(jv)
    await jv.aclose()
    assert client.closed is False


def test_available_is_false_until_a_decision_needs_a_client() -> None:
    assert Jev().available is False


def test_explain_reads_as_a_sentence_for_the_log() -> None:
    from xg_project.jev import Routing

    assert "0.91" in Routing(node="command", confidence=0.91).explain()
    assert "no confidence" in Routing(node="command").explain()
    assert Routing(problem="boom").explain() == "boom"


# -- the key: environment first, then `pass` -------------------------------


def recording_client(built: list):
    """A stub SDK client that records the keyword arguments it was built with."""

    class StubClient:
        def __init__(self, **kwargs) -> None:
            built.append(kwargs)

        async def system_one(self, **kwargs):
            return response(choice="command")

        async def aclose(self) -> None:
            pass

    return StubClient


async def test_the_environment_key_wins_over_the_password_store(monkeypatch) -> None:
    """An exported key is never second-guessed by shelling out to `pass`."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    monkeypatch.setattr(
        "xg_project.jev.api_key_from_pass", lambda entry="jev": pytest.fail("pass consulted")
    )
    built: list = []
    monkeypatch.setattr("xg_project.jev.AsyncTypeSafeClient", recording_client(built))

    assert (await decide(Jev())).ok
    assert built[0]["api_key"] == "env-key"


async def test_the_key_falls_back_to_the_password_store(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("xg_project.jev.api_key_from_pass", lambda entry="jev": "pass-key")
    built: list = []
    monkeypatch.setattr("xg_project.jev.AsyncTypeSafeClient", recording_client(built))

    assert (await decide(Jev())).ok
    assert built[0]["api_key"] == "pass-key"


def completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["pass", "show", "jev"], returncode, stdout, "")


def test_pass_key_is_read_from_the_entry(monkeypatch) -> None:
    calls: list = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return completed("apikey_secret\n")

    monkeypatch.setattr("xg_project.jev.subprocess.run", fake_run)
    assert api_key_from_pass() == "apikey_secret"
    assert calls == [["pass", "show", "jev"]]


def test_a_nonzero_pass_exit_is_no_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "xg_project.jev.subprocess.run", lambda cmd, **kwargs: completed(returncode=1)
    )
    assert api_key_from_pass() is None


def test_an_empty_pass_entry_is_no_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "xg_project.jev.subprocess.run", lambda cmd, **kwargs: completed("   \n")
    )
    assert api_key_from_pass() is None


def test_a_missing_pass_binary_is_no_key(monkeypatch) -> None:
    def boom(cmd, **kwargs):
        raise FileNotFoundError("pass")

    monkeypatch.setattr("xg_project.jev.subprocess.run", boom)
    assert api_key_from_pass() is None


def test_a_hanging_pass_is_no_key(monkeypatch) -> None:
    """A gpg prompt must not hang a routing decision; the timeout is a no-key."""

    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 10)

    monkeypatch.setattr("xg_project.jev.subprocess.run", hang)
    assert api_key_from_pass() is None


# -- selecting the files worth reading -------------------------------------


FILES = {"src/parse.py": "def parse(): ...", "docs/readme.md": "# readme"}


def noul_response(scores: dict[str, float]) -> SystemOneResponse:
    """A real response carrying one Noul answer per file path."""
    return SystemOneResponse(
        model="jev-latest",
        usage=Usage(),
        answers={path: NoulAnswer(noul=score) for path, score in scores.items()},
    )


async def select_files(client: object, **overrides):
    kwargs = {"files": FILES, "prompt": "where is the parser?"}
    return await Jev(client=client).select(**{**kwargs, **overrides})


async def test_one_noul_question_is_asked_per_file_named_by_its_path() -> None:
    from typesafe_sdk import Noul

    client = FakeClient(result=noul_response({"src/parse.py": 0.9, "docs/readme.md": 0.2}))
    await select_files(client)

    questions = client.calls[0]["questions"]
    assert set(questions) == set(FILES)
    assert all(isinstance(question, Noul) for question in questions.values())


async def test_the_selection_state_is_the_file_map_itself() -> None:
    """Path -> direct content, so the question name is the key into the state."""
    client = FakeClient(result=noul_response({"src/parse.py": 0.9, "docs/readme.md": 0.2}))
    await select_files(client)
    assert client.calls[0]["state"] == FILES


async def test_files_above_the_threshold_are_kept_and_the_rest_dropped() -> None:
    client = FakeClient(result=noul_response({"src/parse.py": 0.9, "docs/readme.md": 0.4}))
    selection = await select_files(client)
    assert selection.ok
    assert set(selection.files) == {"src/parse.py"}
    assert selection.files["src/parse.py"] == FILES["src/parse.py"]


async def test_the_threshold_is_exclusive() -> None:
    """More than 0.85: a file exactly at 0.85 is dropped."""
    client = FakeClient(result=noul_response({"src/parse.py": 0.85, "docs/readme.md": 0.86}))
    selection = await select_files(client)
    assert set(selection.files) == {"docs/readme.md"}


async def test_every_file_is_scored_even_when_not_answered() -> None:
    client = FakeClient(result=noul_response({"src/parse.py": 0.9}))
    selection = await select_files(client)
    assert set(selection.files) == {"src/parse.py"}
    assert selection.scores == {"src/parse.py": 0.9, "docs/readme.md": 0.0}


async def test_the_threshold_can_be_lowered() -> None:
    client = FakeClient(result=noul_response({"src/parse.py": 0.5, "docs/readme.md": 0.4}))
    selection = await select_files(client, threshold=0.45)
    assert set(selection.files) == {"src/parse.py"}
    assert selection.threshold == 0.45


async def test_a_selection_failure_is_a_problem_not_a_raise() -> None:
    selection = await select_files(FakeClient(raises=TypeSafeAPIConnectionError("timed out")))
    assert not selection.ok
    assert "could not reach Jev" in selection.problem


async def test_no_files_is_a_problem_and_costs_no_request() -> None:
    client = FakeClient(result=noul_response({}))
    selection = await select_files(client, files={})
    assert not selection.ok
    assert client.calls == []


def test_explain_reports_the_kept_count_against_the_whole() -> None:
    from xg_project.jev import Selection

    selection = Selection(files={"a": ""}, scores={"a": 0.9, "b": 0.1})
    assert "kept 1 of 2" in selection.explain()
    assert Selection(problem="boom").explain() == "boom"
