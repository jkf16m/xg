"""Live tests for the Jev request classifier.

Requires ``TYPESAFE_API_KEY``; skipped without it:

    uv run pytest -m integration -v src/xg_project/jev/test_integration.py
"""

import os

import pytest

from xg_project.jev import RequestKind, classify

pytestmark = pytest.mark.integration

requires_key = pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY"),
    reason="TYPESAFE_API_KEY is not set",
)


@requires_key
def test_classifies_a_feature_edit():
    result = classify("rename the stream_turn function to start_turn everywhere")
    assert result.kind in {RequestKind.REFACTOR, RequestKind.EDIT_FEATURE}
    assert result.changes_code.yes


@requires_key
def test_classifies_a_question_as_no_change():
    result = classify("what does the .xgignore file do?")
    assert result.kind is RequestKind.QUESTION
    assert not result.is_change


@requires_key
def test_exposes_probability_distribution():
    result = classify("add a --json flag to the classifier CLI")
    assert result.request_kind.probabilities
    assert sum(result.request_kind.probabilities.values()) == pytest.approx(1.0, abs=0.01)
