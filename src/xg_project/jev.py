"""The Jev driver: the one place that asks TypeSafe where a request goes.

Jev is a *classifier*, not a generator. It is asked one `Choice` question —
"which node should handle this?" — and it answers with a label. Everything else
in xg is bookkeeping around that answer.

Three things about this module are deliberate.

**It is async, and the client is held open.** xg's TUI runs on an event loop, and
a synchronous HTTP call inside a message handler would freeze the interface for
the length of a round trip. The async client is created once and reused, so the
connection pool outlives a single decision.

**Failure is a value, not an exception.** A missing API key, a timeout, an
unreachable API: none of these are faults in xg, and none of them should end the
session. They come back as a :class:`Routing` with ``problem`` set, which the TUI
prints as a sentence. The one exception is a bug in xg itself.

**The option set is built from the graph.** :meth:`Jev.decide` is handed the
criteria; it does not know what nodes exist. That is what keeps the drawn graph
and the asked question the same object.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    NoulCriteria,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeError,
)

NEXT_NODE = "next_node"
"""The question name. It appears in the answer map and in request logs, so it is
a constant rather than a literal repeated at each call site."""

ENV_API_KEY = "TYPESAFE_API_KEY"
"""The environment variable the SDK reads on its own. Checked first, so a shell
that exports a key is never second-guessed by the password store."""

PASS_ENTRY = "jev"
"""The `pass` entry holding the key, as ``pass show jev`` on the command line."""

PASS_TIMEOUT = 30.0
"""Seconds to wait for `pass`.

A cold gpg-agent pops a pinentry dialog and waits for a human, which takes
longer than a network call and longer than a person expects a key lookup to
take. A routing decision must not hang the session forever, so there is a
bound — but it is set for a person typing a passphrase, not for a warm cache.
"""


def api_key_from_pass(entry: str = PASS_ENTRY) -> str | None:
    """Read the TypeSafe key from the `pass` store, or ``None`` on any problem.

    ``pass show`` prints the secret alone when the entry is a plain string, which
    is the shape ``jev`` has, so the whole of stdout after stripping is the key.

    Every failure returns ``None`` rather than raising: a missing `pass` binary, a
    locked store, a missing entry, and a timeout are all "no key here", and the
    caller already has a path for that — the request comes back as a
    :class:`Routing` with ``problem`` set. Distinguishing them would only move the
    same sentence somewhere else.

    The call is synchronous, so it blocks the event loop while `pass` runs. It
    happens once, on the first decision, and it is a local process rather than a
    network round trip, which is why it is not pushed to a thread.
    """
    try:
        result = subprocess.run(
            ["pass", "show", entry],
            capture_output=True,
            text=True,
            check=False,
            timeout=PASS_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    key = result.stdout.strip()
    return key or None

INSTRUCTIONS = """\
You are choosing the node in a graph of workflows that should handle a request.

The request is the `state` above. Its first entry is the user's goal, stated at
the origin; later entries are what earlier nodes contributed. The node the
request is currently sitting at is named in the state as well — it is never one
of the options, because every option is a move further into the graph.

Each option is a node, and its text says what that node does — the effect it has
on a request. It does not tell you when to pick it; that is your judgement.
Choose the option whose described effect is the one the request needs.

If more than one option looks plausible, choose the one that is the closest
*precondition* for the request: the node that would have to run first for the
others to make sense. If the request does not match any option's description,
choose the option that is the least wrong and answer with low confidence.
"""


SELECT_INSTRUCTIONS = """\
The user asked: {request}

The state above is a JSON object whose keys are file paths and whose values are
the full contents of those files. Answer with the probability that the file at
`{path}` is one of the files the user's request is about.

Judge only that one file, and judge it for belonging to the set the request is
about rather than for being the most important member of that set. Every file
that would contribute to fulfilling the request belongs to it, whether the
request would change it, quote it, or only be understood through it. A file is
unrelated only when it has nothing to do with the request at all; being
secondary to another file that also belongs, or mattering less than it, is not a
reason to leave it out.

This question gathers and does not rank. A later question ranks what is gathered
here, so admitting a file that belongs but is not central costs only context,
while leaving out a file that belongs loses it for the rest of the run.
"""

SELECT_CRITERIA = NoulCriteria(
    true="the file is one of the files the request is about",
    false="the file has nothing to do with the request",
)
"""The two ends of the question FILTER asks, in terms of belonging rather than
of usefulness.

A file is dropped only by falling below the threshold, so the shape of the false
end is what decides whether a secondary-but-related file survives: "would not
help fulfil the request" puts every supporting file near a no, while "has
nothing to do with the request" puts them near a yes. FILTER gathers the set the
request is about; SORT ranks it. Asking FILTER to rank as well makes it drop the
supporting files that a request to explain or change something is usually made
of.
"""

IMPORTANCE_INSTRUCTIONS = """\
The user asked: {request}

The state above is a JSON object whose keys are file paths and whose values are
the full contents of those files. Answer with the probability that the file at
`{path}` is one of the files that must be read to fulfil the request, judged
against the others.

Judge only that one file. A file the request cannot be fulfilled without scores
near one; a file that merely lives in the same project, or only mentions a
related name, scores near zero. This is a ranking, so reserve high probabilities
for the few files that carry the request.
"""

IMPORTANCE_CRITERIA = NoulCriteria(
    true="the file is central to fulfilling the request",
    false="the file is peripheral or irrelevant to the request",
)

MODULE_INSTRUCTIONS = """\
The user asked: {request}

The state above is a JSON object whose keys are context module paths and whose
values are the modules' own declarations of what they expose and import. Answer
with the probability that the module at `{path}` owns the context this request
should be answered from.

Judge only that one module, and judge it on its own declaration. A module whose
description says it is about what the request is about scores high. A module
that merely lives near the subject, or that would be convenient to have, scores
low — the module chosen is the only one whose files will be read, so a wrong
choice hides the right files entirely.
"""

MODULE_CRITERIA = NoulCriteria(
    true="the request should be answered from this module's context",
    false="the request belongs to a different module's context",
)

RELEVANCE_THRESHOLD = 0.5
"""The probability above which a file counts as relevant.

The Noul primitive returns the probability that a yes/no question is true, and
its documentation recommends 0.5 when a yes and a no are equally easy to act on,
saying to lower it when missing a true yes is expensive. That is the asymmetry
here: dropping a file the request needed makes the answer or the edit wrong,
while keeping one file too many costs only a little context. An earlier 0.85 was
raised for the context cost and turned out above what the model ever produced,
so nothing was ever selected.
"""

RESIZE_STATUSES = frozenset({400, 413})
"""HTTP statuses meaning the request itself was refused, so fewer files may fit.

400 is the API's bad-request answer and 413 is payload-too-large; both are worth
asking again with a smaller state rather than reporting. Anything else — a 429, a
5xx, a connection error — is not about the size of the request, and is either
retried by the SDK or reported as the problem it is.
"""

SHRINK_RATIO = 0.8
"""What fraction of a refused payload to keep before asking again.

Dropping one file at a time would re-ask once per file on a very large tree.
Cutting to four fifths converges in a handful of attempts while removing as few
files as possible, because the largest are dropped first.
"""


@dataclass(frozen=True)
class Selection:
    """Which files Jev judged relevant to a request, or why it could not say.

    ``files`` is the kept subset, ``scores`` is every file's probability, kept or
    not. The scores are retained because a dropped file at 0.84 and a dropped
    file at 0.02 look identical if only the survivors are returned, and that
difference is what a threshold is chosen against.
    """

    files: Mapping[str, str] = field(default_factory=dict)
    scores: Mapping[str, float] = field(default_factory=dict)
    model: str | None = None
    threshold: float = RELEVANCE_THRESHOLD
    dropped: tuple[str, ...] = ()
    """Files left out because even the shrunk request would not fit.

    They keep their zero score and so are not kept: nothing judged them, and
    passing them on would put content into the context that nothing vouched for.
    They are named rather than merely counted so the state can show what was
    left unread.
    """
    requests: int = 0
    """How many requests the selection took, including any refused for size."""
    problem: str | None = None

    @property
    def ok(self) -> bool:
        """Whether Jev produced a selection."""
        return self.problem is None

    def explain(self) -> str:
        """A one-line account of the selection, for the log."""
        if self.problem is not None:
            return self.problem
        line = f"kept {len(self.files)} of {len(self.scores)} files above {self.threshold:.2f}"
        if self.dropped:
            line += f" ({len(self.dropped)} did not fit)"
        return line


def build_state_from(files: Mapping[str, str]) -> dict[str, str]:
    """The state parameter for a relevance question: path -> content.

    A named function rather than a ``dict(files)`` inline, because this is the
    one place the wire shape of the selection state is written down. The keys
    are the question names as well, which is what lets one request carry one
    question per file.
    """
    return dict(files)


def _shrink(items: Sequence[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[str]]:
    """Drop the largest files until the payload is a fifth smaller.

    Largest first because a byte of budget freed is a byte freed: removing one
    big file keeps more of the small ones than removing them one by one. The last
    file is never dropped, since a request with no state has nothing to answer.
    """
    remaining = list(items)
    total = sum(len(path) + len(content) for path, content in remaining)
    target = total * SHRINK_RATIO
    dropped: list[str] = []
    while len(remaining) > 1 and total > target:
        biggest = max(remaining, key=lambda item: len(item[0]) + len(item[1]))
        remaining.remove(biggest)
        dropped.append(biggest[0])
        total -= len(biggest[0]) + len(biggest[1])
    return remaining, dropped


@dataclass(frozen=True)
class Routing:
    """Where Jev sent a request, or why it could not.

    ``node`` is ``None`` exactly when ``problem`` is set, so a caller can branch
    on either; both are checked in tests because a caller could get it wrong.
    ``probabilities`` is kept because it is the cheap explanation for a decision:
    a confident answer and a coin flip look identical if only the winner is kept.
    """

    node: str | None = None
    confidence: float = 0.0
    probabilities: Mapping[str, float] = field(default_factory=dict)
    model: str | None = None
    problem: str | None = None

    @property
    def ok(self) -> bool:
        """Whether Jev produced a destination."""
        return self.node is not None

    def explain(self) -> str:
        """A one-line account of the decision, for the log."""
        if self.problem is not None:
            return self.problem
        if self.confidence <= 0:
            return f"{self.node} (no confidence reported)"
        return f"{self.node} (confidence {self.confidence:.2f})"


def build_state(*, current: str, context: Sequence[str], prompt: str) -> list[str]:
    """Assemble what Jev is shown.

    A flat list of labelled lines rather than a JSON object, because the ordering
    is the meaning: the goal comes first and each later entry was added by a node
    further down the graph. A list of strings preserves that and reads the same
    way in a log.
    """
    state = [f"currently at node: {current}"]
    state.extend(context)
    if not context or context[-1] != prompt:
        state.append(f"the user says: {prompt}")
    return state


class Jev:
    """A reusable connection to TypeSafe, plus the one question xg asks it."""

    def __init__(self, *, client: object | None = None, model: str | None = None) -> None:
        """Wrap a client, or arrange to build one on first use.

        ``client`` is injectable so tests can exercise routing without a network
        call and without an API key. Passing one also transfers ownership of
        closing it to the caller, which is why :meth:`aclose` only closes a
        client this object built.
        """
        self._client = client
        self._model = model
        self._owned = client is None

    @property
    def available(self) -> bool:
        """Whether a client has been built yet. Building one is lazy."""
        return self._client is not None

    def _connection(self):
        """Return the client, building it on first use.

        The key is resolved here rather than at construction: the environment
        variable first, then `pass show jev`. Construction is deferred until a
        decision is actually needed, so xg starts, draws its graph, and lets the
        user move around with no key configured. If neither source has a key the
        SDK raises its own error, which :meth:`decide` turns into a sentence.
        """
        if self._client is None:
            api_key = os.environ.get(ENV_API_KEY) or api_key_from_pass()
            self._client = AsyncTypeSafeClient(api_key=api_key, model=self._model)
        return self._client

    async def decide(
        self,
        *,
        current: str,
        context: Sequence[str],
        prompt: str,
        options: Mapping[str, str],
    ) -> Routing:
        """Ask which node should handle ``prompt``, and return Jev's answer.

        An empty option set is answered without a request: a leaf has no children,
        so there is nothing to choose and no reason to spend a round trip proving
        it.
        """
        if not options:
            return Routing(problem=f"{current!r} has no nodes to route to")

        try:
            client = self._connection()
            response = await client.system_one(
                state=build_state(current=current, context=context, prompt=prompt),
                questions={
                    NEXT_NODE: Choice(instructions=INSTRUCTIONS, criteria=dict(options))
                },
            )
        except TypeSafeAPIConnectionError as error:
            return Routing(problem=f"could not reach Jev: {error}")
        except TypeSafeAPIError as error:
            return Routing(problem=f"Jev rejected the request: {error}")
        except TypeSafeError as error:
            # Missing key, malformed question: xg's side of the exchange.
            return Routing(problem=f"Jev is not configured: {error}")

        answer = response.choices.get(NEXT_NODE)
        if answer is None:
            return Routing(problem="Jev returned no answer to the routing question")

        if answer.choice not in options:
            # A label outside the offered set is not something to guess at; the
            # movement policy would refuse it anyway, and silently accepting it
            # would hide a model or schema problem behind a normal-looking move.
            offered = ", ".join(sorted(options))
            return Routing(
                problem=f"Jev chose {answer.choice!r}, which was not offered ({offered})"
            )

        return Routing(
            node=answer.choice,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
            model=response.model,
        )

    async def select(
        self,
        *,
        files: Mapping[str, str],
        prompt: str,
        threshold: float = RELEVANCE_THRESHOLD,
        instructions: str | None = None,
        criteria: NoulCriteria | None = None,
    ) -> Selection:
        """Ask one question per file, all of them in one request where it fits.

        The whole file set goes into a **single** request. Jev answers every
        question against the same state and evaluates the questions independently,
        so one request has the model judge the files against each other. Splitting
        the set across requests makes a file's score depend on which files shared
        its request, which is a judgement nobody asked for.

        If the request is refused because it is too large, the largest files are
        dropped and it is asked again, until it fits or only one file is left. The
        files left out are reported in ``dropped`` rather than scored as if they
        had been judged and found wanting.

        ``instructions`` is a template with two placeholders, ``{request}`` and
        ``{path}``, and it is formatted once per file. The path is not optional:
        a question's *id* is not sent to the model, so a request of questions that
        differ only in their keys is a request the model cannot make sense of.
        Naming the file in its own instructions is what makes one question per
        file mean anything. A caller wanting a different judgement — FILTER keeps
        what is relevant, SORT ranks what matters most — supplies its own
        instructions and criteria, and shares this request handling.

        A file Jev did not answer for scores zero and is dropped. That is the safe
        direction: its relevance is unknown, and passing it on would put content
        into the context that nothing vouched for.
        """
        if not files:
            return Selection(problem="there are no files to consider", threshold=threshold)

        question_instructions = instructions if instructions is not None else SELECT_INSTRUCTIONS
        question_criteria = criteria if criteria is not None else SELECT_CRITERIA

        try:
            client = self._connection()
        except TypeSafeError as error:
            return Selection(problem=f"Jev is not configured: {error}", threshold=threshold)

        scores: dict[str, float] = {path: 0.0 for path in files}
        remaining = list(files.items())
        dropped: list[str] = []
        model: str | None = None
        requests = 0

        while True:
            requests += 1
            questions = {
                path: Noul(
                    instructions=question_instructions.format(request=prompt, path=path),
                    criteria=question_criteria,
                )
                for path, _ in remaining
            }
            try:
                response = await client.system_one(
                    state=build_state_from(dict(remaining)),
                    questions=questions,
                )
            except TypeSafeAPIConnectionError as error:
                return Selection(problem=f"could not reach Jev: {error}", threshold=threshold)
            except TypeSafeAPIError as error:
                if error.status not in RESIZE_STATUSES:
                    return Selection(
                        problem=f"Jev rejected the request: {error}", threshold=threshold
                    )
                remaining, just_dropped = _shrink(remaining)
                if not just_dropped:
                    return Selection(
                        problem=f"Jev refused even one file: {error}", threshold=threshold
                    )
                dropped.extend(just_dropped)
                continue
            except TypeSafeError as error:
                return Selection(problem=f"Jev is not configured: {error}", threshold=threshold)

            model = response.model
            for path, _ in remaining:
                answer = response.nouls.get(path)
                scores[path] = answer.noul if answer is not None else 0.0
            break

        kept = {
            path: content
            for path, content in files.items()
            if scores.get(path, 0.0) > threshold
        }
        return Selection(
            files=kept,
            scores=scores,
            model=model,
            threshold=threshold,
            dropped=tuple(dropped),
            requests=requests,
        )

    async def aclose(self) -> None:
        """Close the connection if this object built it."""
        if self._client is not None and self._owned:
            await self._client.aclose()
            self._client = None
