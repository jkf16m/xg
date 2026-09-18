"""xg_project.jev — the TypeSafe Jev decision layer.

Jev is a *decision* model: it returns typed ``Noul``/``Choice``/``Score``
answers, never text and never tool calls. It answers every question about one
``state`` in a single parallel pass. See https://docs.typesafe.ai.

The pipeline here classifies an incoming request so the router can choose a
path before any generative model runs. Later pipelines (file selection, task
gating) follow the same shape: build a state, ask typed questions, gate on
confidence.

Public API
----------
    classify(request, *, context=None, client=None, model=None) -> Classification
    build_questions() -> dict[str, Choice | Noul | Score]
    build_state(request, context=None) -> dict[str, object]

Types:
    Classification, ChoiceOutcome, ScoreOutcome, NoulOutcome
    RequestKind, Scope, ConfidenceBand
    JevError -> a TypeSafe-side failure the caller can retry
"""

from xg_project.jev._classify import (
    ChoiceOutcome,
    Classification,
    JevError,
    NoulOutcome,
    ScoreOutcome,
    State,
    build_questions,
    build_state,
    classify,
)
from xg_project.jev._client import build_client, get_api_key
from xg_project.jev._taxonomy import (
    COMPLEXITY_LEVELS,
    CONFIDENCE_FLOOR,
    CONFIDENCE_HIGH,
    REQUEST_KIND_CRITERIA,
    SCOPE_CRITERIA,
    ConfidenceBand,
    RequestKind,
    Scope,
    band,
)

__all__ = [
    "COMPLEXITY_LEVELS",
    "CONFIDENCE_FLOOR",
    "CONFIDENCE_HIGH",
    "REQUEST_KIND_CRITERIA",
    "SCOPE_CRITERIA",
    "ChoiceOutcome",
    "Classification",
    "ConfidenceBand",
    "JevError",
    "NoulOutcome",
    "RequestKind",
    "Scope",
    "ScoreOutcome",
    "State",
    "band",
    "build_client",
    "build_questions",
    "build_state",
    "classify",
    "get_api_key",
]
