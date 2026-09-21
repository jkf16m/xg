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
    TypeSafeAPIError,
    TypeSafeAPIConnectionError,
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

PASS_TIMEOUT = 10.0
"""Seconds to wait for `pass`. It can block on a gpg passphrase prompt, and a
routing decision must not hang the session waiting for one."""


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
You are routing one request to the single node in a graph of workflows that
should handle it next.

The request is the `state` above. Its first entry is the user's goal, stated at
the origin; later entries are what earlier nodes contributed. The node the
request is currently sitting at is named in the state as well — it is never one
of the options, because every option is a move further into the graph.

Each option is a node. The label is the node's name and the text describes what
that node does and when to pick it. Choose the option whose description the
request actually matches.

If more than one option looks plausible, choose the one that is the closest
*precondition* for the request: the node that would have to run first for the
others to make sense. If the request does not match any option's description,
choose the option that is the least wrong and answer with low confidence.
"""


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

    async def aclose(self) -> None:
        """Close the connection if this object built it."""
        if self._client is not None and self._owned:
            await self._client.aclose()
            self._client = None
