"""Request classification: the first Jev pipeline.

One ``system_one`` call evaluates the request against every question the
router needs, in parallel (see https://docs.typesafe.ai/primitives). The result
is a typed :class:`Classification`, not prose.

Jev never generates the reply or the tool calls. This module decides what kind
of request arrived, what the work will produce (``operation``), how hard it is,
whether the repository must change, and whether it is risky, so the router can
choose a path before any generative model runs. A request the model cannot
understand is marked ``confused`` and rerouted to the user rather than guessed
at. How far-reaching the request is (*scope*) is not decided here; the
``local_research`` step decides it by finding the relevant files.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient, TypeSafeError

from xg_project.jev._client import build_client
from xg_project.jev._taxonomy import (
    COMPLEXITY_LEVELS,
    CONFIDENCE_FLOOR,
    OPERATION_CRITERIA,
    REQUEST_KIND_CRITERIA,
    ConfidenceBand,
    Operation,
    RequestKind,
    band,
)

State = str | Mapping[str, object] | list[object]
"""Jev's ``state``: text, an object of named fields, or a sequence of records."""


class JevError(RuntimeError):
    """A TypeSafe-side failure (auth, rate limit, outage) the caller can retry."""


@dataclass(frozen=True)
class ChoiceOutcome:
    """A ``Choice`` answer: the chosen label, its confidence, the full spread."""

    label: str
    confidence: float
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class ScoreOutcome:
    """A ``Score`` answer, normalized onto 0..1 across the rubric levels."""

    score: float
    confidence: float
    levels: int

    @property
    def normalized(self) -> float:
        """The score divided by the top level, so scales of any length compare."""
        top = self.levels - 1
        return self.score / top if top > 0 else 0.0


@dataclass(frozen=True)
class NoulOutcome:
    """A ``Noul`` answer: the probability the answer is yes."""

    probability: float

    @property
    def yes(self) -> bool:
        return self.probability >= 0.5


@dataclass(frozen=True)
class Classification:
    """Everything the router learned about one request in a single call."""

    request_kind: ChoiceOutcome
    operation: ChoiceOutcome
    complexity: ScoreOutcome
    changes_code: NoulOutcome
    is_destructive: NoulOutcome
    model: str

    @property
    def kind(self) -> RequestKind:
        """The primary kind, coerced to the enum (``OTHER`` on unknown labels)."""
        try:
            return RequestKind(self.request_kind.label)
        except ValueError:
            return RequestKind.OTHER

    @property
    def operation_kind(self) -> Operation | None:
        """What the work will produce, or ``None`` when the label is unknown.

        ``None`` is deliberately not coerced to a default: a caller that has to
        act on the operation should reroute rather than assume one.
        """
        try:
            return Operation(self.operation.label)
        except ValueError:
            return None

    @property
    def band(self) -> ConfidenceBand:
        return band(self.request_kind.confidence)

    @property
    def needs_clarification(self) -> bool:
        """Whether the model is too unsure for the router to act on."""
        return self.request_kind.confidence < CONFIDENCE_FLOOR

    @property
    def needs_reroute(self) -> bool:
        """Whether to send this back to the user instead of acting on it.

        True when the model is unsure of the kind, or when the kind is one the
        graph has no path for: ``confused`` because the request cannot be acted
        on, ``other`` because nothing here handles it. Guessing would be worse
        than asking.
        """
        return self.needs_clarification or self.kind in {
            RequestKind.CONFUSED,
            RequestKind.OTHER,
        }

    @property
    def is_change(self) -> bool:
        """Whether fulfilling the request is expected to modify the repository."""
        return self.changes_code.yes and self.kind is not RequestKind.QUESTION

    def to_dict(self) -> dict[str, object]:
        """A JSON-serializable view, for logging and the CLI."""
        return {
            "model": self.model,
            "request_kind": self.request_kind.label,
            "kind_confidence": self.request_kind.confidence,
            "kind_band": self.band.value,
            "kind_probabilities": dict(self.request_kind.probabilities),
            "operation": self.operation.label,
            "operation_confidence": self.operation.confidence,
            "complexity": self.complexity.score,
            "complexity_normalized": self.complexity.normalized,
            "complexity_confidence": self.complexity.confidence,
            "changes_code": self.changes_code.probability,
            "is_destructive": self.is_destructive.probability,
            "needs_clarification": self.needs_clarification,
            "needs_reroute": self.needs_reroute,
            "is_change": self.is_change,
        }


def build_questions() -> dict[str, Choice | Noul | Score]:
    """The full question set asked in one ``system_one`` call."""
    return {
        "request_kind": Choice(
            instructions=(
                "What is the user asking for? Pick the single best description of "
                "the request's primary goal. If the request both asks a question "
                "and asks for a change, pick the change."
            ),
            criteria=REQUEST_KIND_CRITERIA,
        ),
        "operation": Choice(
            instructions=(
                "What will fulfilling this request produce? Pick what the work "
                "creates, not what the user is talking about."
            ),
            criteria=OPERATION_CRITERIA,
        ),
        "complexity": Score(
            instructions="How complex is fulfilling this request overall?",
            criteria=list(COMPLEXITY_LEVELS),
        ),
        "changes_code": Noul(
            instructions=(
                "Does fulfilling this request require modifying files in the "
                "repository?"
            ),
            criteria={
                "true": "At least one file must be created, edited, or deleted.",
                "false": (
                    "The request can be fully served by reading files, answering "
                    "in text, or running a non-mutating command."
                ),
            },
        ),
        "is_destructive": Noul(
            instructions=(
                "Could fulfilling this request delete or overwrite data, rewrite "
                "history, or run an irreversible command?"
            ),
            criteria={
                "true": "Data loss, history rewrite, or an irreversible action is plausible.",
                "false": "The request is read-only, additive, or easily reversible.",
            },
        ),
    }


def build_state(request: str, context: Mapping[str, object] | None = None) -> dict[str, object]:
    """Compose the ``state`` object the questions are evaluated against."""
    state: dict[str, object] = {"request": request}
    if context:
        state["context"] = dict(context)
    return state


def classify(
    request: str,
    *,
    context: Mapping[str, object] | None = None,
    client: TypeSafeClient | None = None,
    model: str | None = None,
) -> Classification:
    """Classify one request with Jev.

    ``client`` lets a caller reuse a connection; when omitted, one is built
    (from ``TYPESAFE_API_KEY`` or ``pass show jev``) and closed. ``model``
    overrides the SDK default (``jev-latest``) and applies only to a client
    this call builds.
    """
    if not request.strip():
        raise ValueError("request must not be empty")

    state = build_state(request, context)
    if client is not None:
        return _run(client, state, model)

    try:
        with build_client(model) as owned:
            return _run(owned, state, model)
    except TypeSafeError as exc:
        raise JevError(str(exc)) from exc


def _run(
    client: TypeSafeClient, state: dict[str, object], model: str | None
) -> Classification:
    """Issue the single call and map the typed answers."""
    try:
        response = client.system_one(
            state=state, questions=build_questions(), model=model
        )
    except TypeSafeError as exc:
        raise JevError(str(exc)) from exc

    kind = response.choices["request_kind"]
    operation = response.choices["operation"]
    complexity = response.scores["complexity"]
    return Classification(
        request_kind=ChoiceOutcome(
            label=kind.choice,
            confidence=kind.confidence,
            probabilities=dict(kind.probabilities),
        ),
        operation=ChoiceOutcome(
            label=operation.choice,
            confidence=operation.confidence,
            probabilities=dict(operation.probabilities),
        ),
        complexity=ScoreOutcome(
            score=complexity.score,
            confidence=complexity.confidence,
            levels=len(COMPLEXITY_LEVELS),
        ),
        changes_code=NoulOutcome(response.nouls["changes_code"].noul),
        is_destructive=NoulOutcome(response.nouls["is_destructive"].noul),
        model=response.model,
    )
