"""Offline tests for the Jev pipeline planner.

The loop is exercised with a fake client and real ``typesafe_sdk`` answer
structs, so the response-to-``Plan`` mapping and every guard are tested without
a network call. The live path is covered by ``scripts/jev_planner_test.py``.
"""

import pytest
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer

from xg_project.planner import (
    NEW_FILE,
    Plan,
    Step,
    StepKind,
    build_questions,
    build_state,
    plan,
    step_criteria,
)

FILES = {"src/a.py": "print('a')\n", "src/b.py": "print('b')\n"}


class _FakeResponse:
    def __init__(self, choices, nouls, model="jev-test"):
        self.choices = choices
        self.nouls = nouls
        self.model = model


class _FakeClient:
    """Returns one scripted response per round and records the requests."""

    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.calls = []

    def system_one(self, *, state, questions, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        return self._rounds.pop(0)


def _round(done=0.0, node="edit:src/a.py", confidence=0.9, model="jev-test"):
    return _FakeResponse(
        choices={
            "next": ChoiceAnswer(
                choice=node,
                confidence=confidence,
                probabilities={node: confidence, "answer": 1 - confidence},
            ),
        },
        nouls={"done": NoulAnswer(noul=done)},
        model=model,
    )


def _plan(rounds, **kwargs):
    kwargs.setdefault("max_steps", len(rounds) + 1)
    return plan("refactor the parser", files=FILES, client=_FakeClient(rounds), **kwargs)


def test_builds_ordered_steps_until_done():
    result = _plan(
        [
            _round(node="read:src/a.py"),
            _round(node="edit:src/b.py"),
            _round(done=0.9),
        ]
    )

    assert isinstance(result, Plan)
    assert result.steps == (Step(StepKind.READ, "src/a.py"), Step(StepKind.EDIT, "src/b.py"))
    assert result.stopped == "done"
    assert result.rounds == 3
    assert result.model == "jev-test"


def test_stops_as_soon_as_done_fires():
    result = _plan([_round(done=0.8)])

    assert result.is_empty
    assert result.stopped == "done"
    assert result.rounds == 1


def test_state_shows_the_plan_built_so_far():
    client = _FakeClient([_round(node="read:src/a.py"), _round(done=0.9)])
    plan("refactor", files=FILES, client=client, max_steps=4)

    assert client.calls[0]["state"]["plan"] == []
    assert client.calls[1]["state"]["plan"] == [
        {"index": 0, "step": "read", "target": "src/a.py"}
    ]


def test_results_are_included_when_supplied():
    client = _FakeClient([_round(done=0.9)])
    plan("refactor", files=FILES, client=client, max_steps=4, results={0: "line 3 changed"})

    assert client.calls[0]["state"]["results"] == {"0": "line 3 changed"}


def test_new_file_target_is_left_unnamed():
    result = _plan([_round(node=f"write:{NEW_FILE}"), _round(done=0.9)])

    assert result.steps == (Step(StepKind.WRITE, ""),)
    assert result.steps[0].new_file


def test_answer_node_has_no_target():
    result = _plan([_round(node="answer"), _round(done=0.9)])

    assert result.steps == (Step(StepKind.ANSWER, ""),)
    assert not result.steps[0].new_file


def test_unexecutable_node_stops():
    result = _plan([_round(node="deploy")])

    assert result.is_empty
    assert "deploy" in result.stopped


def test_uncertain_node_stops():
    result = _plan([_round(confidence=0.2)])

    assert result.is_empty
    assert "uncertain" in result.stopped


def test_unknown_target_stops():
    result = _plan([_round(node="edit:src/ghost.py")])

    assert result.is_empty
    assert "ghost.py" in result.stopped


def test_write_to_a_named_path_stops():
    result = _plan([_round(node="write:src/new.py")])

    assert result.is_empty
    assert "new_file" in result.stopped


def test_node_without_a_target_stops():
    result = _plan([_round(node="edit:")])

    assert result.is_empty
    assert "not a known file" in result.stopped


def test_repeated_node_stops_instead_of_looping():
    result = _plan([_round(), _round()])

    assert result.steps == (Step(StepKind.EDIT, "src/a.py"),)
    assert "already in the plan" in result.stopped


def test_budget_stops_the_loop():
    result = _plan(
        [_round(node="edit:src/a.py"), _round(node="read:src/b.py")], max_steps=2
    )

    assert len(result.steps) == 2
    assert result.rounds == 2
    assert result.stopped == "budget"


def test_questions_use_the_right_primitives():
    questions = build_questions(FILES)

    assert isinstance(questions["done"], Noul)
    assert isinstance(questions["next"], Choice)
    assert set(questions["next"].criteria) == {
        "edit:src/a.py",
        "read:src/a.py",
        "edit:src/b.py",
        "read:src/b.py",
        f"write:{NEW_FILE}",
        "answer",
    }


def test_step_criteria_describe_every_option():
    criteria = step_criteria(FILES)

    assert len(criteria) == 2 * len(FILES) + 2
    assert all(isinstance(value, dict) for value in criteria.values())


def test_planned_nodes_are_removed_from_the_options():
    criteria = step_criteria(FILES, [Step(StepKind.EDIT, "src/a.py")])

    assert "edit:src/a.py" not in criteria
    assert "read:src/a.py" in criteria  # a read and an edit are distinct nodes
    assert len(criteria) == 2 * len(FILES) + 1


def test_loop_stops_when_no_nodes_remain():
    nodes = [
        "edit:src/a.py",
        "read:src/a.py",
        "edit:src/b.py",
        "read:src/b.py",
        f"write:{NEW_FILE}",
        "answer",
    ]
    result = _plan([_round(node=node) for node in nodes], max_steps=10)

    assert len(result.steps) == len(nodes)
    assert result.stopped == "no unplanned node remains"
    assert result.rounds == len(nodes)


def test_build_state_is_json_shaped():
    state = build_state("refactor", FILES, [Step(StepKind.EDIT, "src/a.py")])

    assert state["request"] == "refactor"
    assert state["files"] == FILES
    assert state["plan"] == [{"index": 0, "step": "edit", "target": "src/a.py"}]


def test_plan_serializes():
    result = _plan([_round(node=f"write:{NEW_FILE}"), _round(done=0.9)])

    assert result.to_dict() == {
        "request": "refactor the parser",
        "model": "jev-test",
        "stopped": "done",
        "rounds": 2,
        "steps": [{"step": "write", "target": None}],
    }


def test_empty_request_rejected():
    with pytest.raises(ValueError):
        plan("   ", files=FILES, client=_FakeClient([]))
