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
The user asked: `query`

The state above is a JSON object with two keys. `query` is the user's request.
`files` is an array of file contents, and this question names one entry of it.

This question is about `files[{index}]`, the file at:

{path}

Answer with the probability that it is one of the files the user's request is
about.

Judge only that one file, and judge it for belonging to the set the request is
about rather than for being the most important member of it. Every file that
would contribute to fulfilling the request belongs: the request may change it,
quote it, or only be understood through it. A file is unrelated only when it has
nothing to do with the request at all; being secondary to another file that also
belongs, or mattering less than it, is not a reason to call it unrelated.

The probabilities returned here are also the order the files are used in, so
spend the range and let them separate. Do not return the same value for several
files because all of them belong.
"""

SELECT_CRITERIA = NoulCriteria(
    true="`files[{index}]` is one of the files the request is about",
    false="`files[{index}]` has nothing to do with the request",
)
"""The two ends of the question FILTER asks, in terms of belonging rather than
of usefulness, and pointed at the file by its key in the state.

Backticked references are a documented TypeSafe feature, not a prompt trick: a
question that names the slice of the state it is about removes a hop of
inference, and makes a wrong answer debuggable — you can see which entry the
question claimed to be about. Naming the entry in the criteria as well as the
instructions is the same move twice, since the criteria are what define the two
ends of the probability.

A file is dropped only by falling below the threshold, so the shape of the false
end is what decides whether a secondary-but-related file survives: "would not
help fulfil the request" puts every supporting file near a no, while "has
nothing to do with the request" puts them near a yes.
"""

MODULE_INSTRUCTIONS = """\
The user asked: `query`

The state above is a JSON object with two keys. `query` is the user's request.
`files` is an array, and each entry is one context module's own declaration of
what it exposes and imports. This question names one entry of it.

Answer with the probability that `files[{index}]` — the module at {path} — owns
the context this request should be answered from.

Judge only that one module, and judge it on its own declaration. A module whose
description says it is about what the request is about scores high. A module
that merely lives near the subject, or that would be convenient to have, scores
low — the module chosen is the only one whose files will be read, so a wrong
choice hides the right files entirely.
"""

MODULE_CRITERIA = NoulCriteria(
    true="`files[{index}]` is the module the request should be answered from",
    false="`files[{index}]` is a different module's context",
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

BATCH_CHARS = 100_000
"""How much file content one request may carry before the next batch starts.

TypeSafe documents two budgets that apply at the same time: 64k tokens for the
whole request, state plus every question combined, and 32k for the state plus the
single longest question. A file tree respects neither, so the reader cannot send
one request per run and hope that it fits.

It sends as many files as fit, then sends the next batch as its own request, and
merges the scores. Every file is put to the model, in some batch — which is the
point. The previous rule dropped the largest files until the payload shrank, so
on this project the six biggest files, 60% of the tree by volume and the six most
likely to hold the answer, were never asked about at all and were then recorded
as having scored zero.

Chars rather than tokens because the split has to be decided before the request
is built. The number comes from this project's own tree: 254,695 characters over
23 files was refused, and 101,191 characters over 17 files was accepted, so
100,000 sits inside the measured gap and below the documented 32k-token ceiling.
"""

TOO_BIG_STATUSES = frozenset({400, 413})
"""HTTP statuses meaning the payload did not fit, so the batch is split again.

400 is the API's bad-request answer — the one it gives for a state past the token
budget — and 413 is payload-too-large. Both are answered by halving the batch and
asking again, which is convergence that cannot throw a file away. Anything else —
a 429, a 5xx, a connection error — is not about the size of the request, and is
either retried by the SDK or reported as the problem it is.
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
    """Files the model did not answer for, or that were too large to send alone.

    They keep their zero score and so are not kept: nothing judged them, and
    passing them on would put content into the context that nothing vouched for.
    They are named rather than merely counted so the state can show what was left
    unread.

    Batching means this should be empty. Every file goes to the model in some
    batch, and a batch refused for its size is split rather than abandoned. It
    stays as a report because the alternative — scoring a file that was never
    asked about as zero — is indistinguishable from the model having rejected it,
    and that ambiguity is what the batching replaced.
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
            line += f" ({len(self.dropped)} went unanswered)"
        return line


def build_state_from(files: Sequence[tuple[str, str]], *, query: str) -> dict[str, object]:
    """The state for a selection question: the request, and an array of file contents.

    Two named keys rather than one top-level key per file. The old shape used the
    file paths themselves as the state's keys, which made the backticked
    reference a question is meant to use ambiguous: a path carries dots and
    slashes, and a dot is a step in a path expression. Here the request is
    ``query``, the contents are the array ``files``, and a question points at
    exactly one entry of it by index.

    This is the shape TypeSafe's own passage-classification cookbook uses —
    ``{"query": …, "passages": […]}}`` with ``Does `passages[i]` help answer
    `query`?`` — and it is the closest documented analogue to selecting files.
    """
    return {"query": query, "files": [content for _, content in files]}


def _batches(
    items: Sequence[tuple[str, str]], *, budget: int = BATCH_CHARS
) -> list[list[tuple[str, str]]]:
    """Split the files into as few batches as the request budget allows.

    A batch is filled in the order given and closed when the next file would take
    it past the budget. A single file larger than the whole budget still gets a
    batch of its own rather than being dropped: it is as likely as any file to be
    the answer, and one refusal to report beats a silent absence.
    """
    batches: list[list[tuple[str, str]]] = []
    batch: list[tuple[str, str]] = []
    size = 0
    for path, content in items:
        weight = len(path) + len(content)
        if batch and size + weight > budget:
            batches.append(batch)
            batch, size = [], 0
        batch.append((path, content))
        size += weight
    if batch:
        batches.append(batch)
    return batches


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

        ``instructions`` and ``criteria`` are templates with two placeholders,
        ``{index}`` and ``{path}``, formatted once per file. Both matter: a
        question's *id* is not sent to the model, so a request whose questions
        differ only in their keys is a request the model cannot make sense of,
        and a backticked reference into the state is what tells it which entry to
        judge. Naming the file in the instructions and again in the criteria is
        the same move at both ends of the probability.

        The files go out in batches that fit the request budget — every one of
        them, with the scores merged afterwards. A batch refused for being too
        large is halved and asked again, so nothing is discarded for its size. A
        file the model did not answer for is named in ``dropped`` and scores
        zero; with batching that should not happen, which is why it is reported
        rather than passed over.
        """
        if not files:
            return Selection(problem="there are no files to consider", threshold=threshold)

        question_instructions = instructions if instructions is not None else SELECT_INSTRUCTIONS
        question_criteria = criteria if criteria is not None else SELECT_CRITERIA

        try:
            client = self._connection()
        except TypeSafeError as error:
            return Selection(problem=f"Jev is not configured: {error}", threshold=threshold)

        scores: dict[str, float] = {}
        unanswered: list[str] = []
        model: str | None = None
        requests = 0
        problems: list[str] = []
        pending = _batches(list(files.items()))

        while pending:
            batch = pending.pop(0)
            requests += 1
            questions = {
                path: Noul(
                    instructions=question_instructions.format(index=index, path=path),
                    criteria=NoulCriteria(
                        true=question_criteria["true"].format(index=index),
                        false=question_criteria["false"].format(index=index),
                    ),
                )
                for index, (path, _) in enumerate(batch)
            }
            try:
                response = await client.system_one(
                    state=build_state_from(batch, query=prompt),
                    questions=questions,
                )
            except TypeSafeAPIConnectionError as error:
                return Selection(problem=f"could not reach Jev: {error}", threshold=threshold)
            except TypeSafeAPIError as error:
                if error.status not in TOO_BIG_STATUSES:
                    return Selection(
                        problem=f"Jev rejected the request: {error}", threshold=threshold
                    )
                if len(batch) == 1:
                    unanswered.append(batch[0][0])
                    problems.append(f"{batch[0][0]} was too large to send")
                    continue
                middle = len(batch) // 2
                pending[:0] = [batch[:middle], batch[middle:]]
                continue
            except TypeSafeError as error:
                return Selection(problem=f"Jev is not configured: {error}", threshold=threshold)

            model = response.model
            for path, _ in batch:
                answer = response.nouls.get(path)
                if answer is None:
                    unanswered.append(path)
                else:
                    scores[path] = answer.noul

        kept = {
            path: content
            for path, content in files.items()
            if scores.get(path, 0.0) > threshold
        }
        return Selection(
            files=kept,
            scores={path: scores.get(path, 0.0) for path in files},
            model=model,
            threshold=threshold,
            dropped=tuple(unanswered),
            requests=requests,
            problem="; ".join(problems) or None,
        )

    async def aclose(self) -> None:
        """Close the connection if this object built it."""
        if self._client is not None and self._owned:
            await self._client.aclose()
            self._client = None
