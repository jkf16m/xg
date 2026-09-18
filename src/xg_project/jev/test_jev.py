"""Offline tests for the Jev request classifier.

These use a fake client and real ``typesafe_sdk`` answer structs, so the
response-to-``Classification`` mapping is exercised without a network call.
The live path is covered by ``test_integration.py``.
"""

import pytest
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer

from xg_project.jev import (
    COMPLEXITY_LEVELS,
    CONFIDENCE_FLOOR,
    CONFIDENCE_HIGH,
    REQUEST_KIND_CRITERIA,
    Classification,
    ConfidenceBand,
    RequestKind,
    band,
    build_questions,
    classify,
)


class _FakeResponse:
    def __init__(self, choices, scores, nouls, model="jev-test"):
        self.choices = choices
        self.scores = scores
        self.nouls = nouls
        self.model = model


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def system_one(self, *, state, questions, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        return self._response


def _response(
    kind="edit_feature",
    kind_confidence=0.9,
    complexity=2.0,
    complexity_confidence=0.7,
    changes_code=0.95,
    destructive=0.1,
):
    return _FakeResponse(
        choices={
            "request_kind": ChoiceAnswer(
                choice=kind,
                confidence=kind_confidence,
                probabilities={kind: kind_confidence, "other": 1 - kind_confidence},
            ),
        },
        scores={
            "complexity": ScoreAnswer(
                score=complexity,
                confidence=complexity_confidence,
                legend={i: level for i, level in enumerate(COMPLEXITY_LEVELS)},
                probabilities={
                    i: 1 / len(COMPLEXITY_LEVELS)
                    for i in range(len(COMPLEXITY_LEVELS))
                },
            )
        },
        nouls={
            "changes_code": NoulAnswer(noul=changes_code),
            "is_destructive": NoulAnswer(noul=destructive),
        },
    )


def test_taxonomy_criteria_cover_every_option():
    assert set(REQUEST_KIND_CRITERIA) == {kind.value for kind in RequestKind}
    assert "other" in REQUEST_KIND_CRITERIA
    assert 2 <= len(COMPLEXITY_LEVELS) <= 10


def test_build_questions_uses_the_right_primitives():
    questions = build_questions()
    assert set(questions) == {
        "request_kind",
        "complexity",
        "changes_code",
        "is_destructive",
    }
    assert isinstance(questions["request_kind"], Choice)
    assert isinstance(questions["complexity"], Score)
    assert isinstance(questions["changes_code"], Noul)
    assert isinstance(questions["is_destructive"], Noul)
    assert list(questions["complexity"].criteria) == list(COMPLEXITY_LEVELS)


def test_classify_maps_typed_answers():
    client = _FakeClient(_response())
    result = classify("move the parser into its own module", client=client)

    assert isinstance(result, Classification)
    assert result.kind is RequestKind.EDIT_FEATURE
    assert result.request_kind.label == "edit_feature"
    assert result.request_kind.confidence == pytest.approx(0.9)
    assert result.complexity.score == pytest.approx(2.0)
    assert result.complexity.normalized == pytest.approx(
        2.0 / (len(COMPLEXITY_LEVELS) - 1)
    )
    assert result.changes_code.yes
    assert not result.is_destructive.yes
    assert result.is_change
    assert result.band is ConfidenceBand.HIGH
    assert not result.needs_clarification
    assert result.model == "jev-test"


def test_classify_asks_all_questions_in_one_call():
    client = _FakeClient(_response())
    classify("explain how sessions work", client=client)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["state"] == {"request": "explain how sessions work"}
    assert set(call["questions"]) == {
        "request_kind",
        "complexity",
        "changes_code",
        "is_destructive",
    }


def test_low_confidence_needs_clarification():
    client = _FakeClient(_response(kind_confidence=0.3))
    result = classify("do the thing", client=client)

    assert result.needs_clarification
    assert result.band is ConfidenceBand.LOW


def test_unknown_label_falls_back_to_other():
    client = _FakeClient(_response(kind="something_new"))
    result = classify("???", client=client)

    assert result.request_kind.label == "something_new"
    assert result.kind is RequestKind.OTHER


def test_question_kind_is_not_a_change():
    client = _FakeClient(_response(kind="question", changes_code=0.1))
    result = classify("what does config.resolve do?", client=client)

    assert result.kind is RequestKind.QUESTION
    assert not result.is_change


def test_context_is_attached_to_state():
    client = _FakeClient(_response())
    classify("edit it", context={"files": ["a.py"]}, client=client)

    assert client.calls[0]["state"] == {
        "request": "edit it",
        "context": {"files": ["a.py"]},
    }


def test_empty_request_rejected():
    with pytest.raises(ValueError):
        classify("   ", client=_FakeClient(_response()))


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (1.0, ConfidenceBand.HIGH),
        (CONFIDENCE_HIGH, ConfidenceBand.HIGH),
        (CONFIDENCE_FLOOR, ConfidenceBand.MEDIUM),
        (0.6, ConfidenceBand.MEDIUM),
        (CONFIDENCE_FLOOR - 0.01, ConfidenceBand.LOW),
        (0.0, ConfidenceBand.LOW),
    ],
)
def test_band_thresholds(confidence, expected):
    assert band(confidence) is expected
