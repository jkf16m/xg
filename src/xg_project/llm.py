"""The executor: the one place that asks an LLM for a structured action.

``Jev`` decides *where* a request goes; the executor decides *what to do* once it
arrives at a leaf. It asks for exactly one thing each time — a shell command or a
file edit, as a **forced tool call** so the answer is structured, or a prose
answer to a question, which needs no tool at all.

Three things are deliberate, and they mirror :mod:`xg_project.jev`.

**It is async, and the client is held open.** The TUI runs on an event loop and
a synchronous HTTP call inside a message handler would freeze it for the length
of a round trip. One client is created and reused, and :meth:`Executor.aclose`
closes it.

**Failure is a value, not an exception.** No key, a timeout, a model that
answers with prose instead of the forced tool: none of these end the session.
They come back as a proposal with ``problem`` set, which the TUI prints as a
sentence. A bug in xg is the only thing that raises.

**The answer is a proposal, never an action.** This module does not run a
command or write a file; it returns what it was asked for as a value. Acting on
it is the TUI's job, and only after the user accepts.

The provider is hardcoded for now: OpenRouter, at ``@preset/deepseek``, keyed by
``pass show pi/openrouter``. Those are constants here rather than configuration
because a single caller uses them today; when a second provider arrives, this is
the seam it plugs into.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import httpx

from xg_project.diff import unified_patch
from xg_project.jev import api_key_from_pass

BASE_URL = "https://openrouter.ai/api/v1"
"""OpenRouter's OpenAI-compatible API root. The endpoint is appended to it."""

CHAT_COMPLETIONS = "/chat/completions"

MODEL = "@preset/deepseek"
"""An OpenRouter preset, referenced as a model. The preset chooses the real
underlying model; naming it here keeps the choice out of xg."""

ENV_API_KEY = "OPENROUTER_API_KEY"
"""The environment variable checked first. An exported key is never
second-guessed by shelling out to ``pass``."""

PASS_ENTRY = "pi/openrouter"
"""The ``pass`` entry holding the key, as ``pass show pi/openrouter``."""

RUN_COMMAND = "run_command"
EDIT_FILE = "edit_file"
READ_FILE = "read_file"
"""The tool the reads in a transcript are attributed to.

The reads are hardcoded, never requested: the workflow chose the files before the
request was built, and handing that choice back to the model would let it read
something the workflow did not select. The tool is still declared, because a
transcript whose calls name an undeclared tool is a request some backends refuse.
"""

REQUEST_TIMEOUT = 120.0
"""Seconds to wait for the model.

A model that thinks before it answers is normal; the bound is there so a stalled
request becomes a sentence instead of a hung screen.
"""


@dataclass(frozen=True)
class Proposal:
    """What one forced tool call produced, or why it could not be produced.

    A proposal is a value the user decides about. ``problem`` is set exactly when
    the model did not deliver a usable action, so ``ok`` is the single check a
    caller needs before offering it up.
    """

    rationale: str | None = None
    model: str | None = None
    problem: str | None = None

    @property
    def ok(self) -> bool:
        """Whether there is an action to decide about."""
        return self.problem is None

    @property
    def tool(self) -> str:
        """The tool this proposal came from. Overridden by each subclass."""
        return "proposal"

    def preview(self) -> str:
        """A one-line rendering of the proposed action, for the state panel."""
        return self.problem or "no action"

    def detail(self) -> str:
        """Extra lines to show below ``preview``, for a state panel.

        Empty for a proposal that has nothing more to say, which is every
        proposal but an edit. It is separate from ``preview`` deliberately: the
        preview is also what Jev is shown when it routes, and a routing decision
        must not be made to read through a patch.
        """
        return ""


@dataclass(frozen=True)
class CommandProposal(Proposal):
    """A shell command the executor proposed, awaiting acceptance."""

    command: str | None = None

    @property
    def ok(self) -> bool:
        return self.problem is None and bool(self.command)

    @property
    def tool(self) -> str:
        return RUN_COMMAND

    def preview(self) -> str:
        return self.problem or self.command or "no command"


@dataclass(frozen=True)
class EditProposal(Proposal):
    """A file edit the executor proposed, awaiting acceptance.

    ``path`` is set by the node, not the model: EDIT edits the file SORT ranked
    first, and handing the choice of path back to the model would let it edit a
    file the workflow never chose.
    """

    path: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    content: str | None = None
    """The file as the executor was shown it.

    Kept so the change can be previewed as a patch. A patch needs the lines
    around the swap, which two fragments cannot supply on their own.
    """

    @property
    def ok(self) -> bool:
        return self.problem is None and bool(self.old_text is not None and self.new_text is not None)

    @property
    def tool(self) -> str:
        return EDIT_FILE

    def preview(self) -> str:
        if self.problem is not None:
            return self.problem
        return f"edit {self.path}: replace {_count(self.old_text)} → {_count(self.new_text)}"

    def detail(self) -> str:
        """The proposed change as a git patch, or ``""`` when there is none to draw.

        Nothing is drawn when the replacement is not uniquely determined, because
        that is exactly when ``apply_edit`` refuses: a patch would show a change
        that will never happen. The one-line preview still says what was
        proposed, and accepting reports the refusal.
        """
        if self.content is None or self.old_text is None or self.new_text is None:
            return ""
        if self.content.count(self.old_text) != 1:
            return ""
        return unified_patch(
            path=self.path or "file",
            before=self.content,
            after=self.content.replace(self.old_text, self.new_text, 1),
        )


def _count(text: str | None) -> str:
    if text is None:
        return "nothing"
    lines = text.count("\n") + 1
    return f"{lines} line" + ("" if lines == 1 else "s")


@dataclass(frozen=True)
class Answer:
    """A prose answer the executor produced from the selected files.

    Deliberately not a `Proposal`: an answer is not an action, so there is
    nothing for the user to accept or reject and no gate around it. It is still a
    value — a request that failed to produce one is a sentence in ``problem``,
    never an exception.
    """

    text: str = ""
    model: str | None = None
    problem: str | None = None

    @property
    def ok(self) -> bool:
        """Whether there is an answer to read."""
        return self.problem is None

    def explain(self) -> str:
        """The answer, or the sentence saying why there is not one."""
        return self.problem or self.text

    def __str__(self) -> str:
        return self.explain()


COMMAND_INSTRUCTIONS = """\
The user wants one shell command run. Propose exactly that command by calling
the `run_command` tool.

You are proposing, not running: nothing happens until the user accepts. Make the
command self-contained and complete. Do not wrap it in a code fence and do not
explain it in prose — the tool call is the whole answer. Put any explanation in
the tool's optional `rationale`.
"""

EDIT_INSTRUCTIONS = """\
The user wants one project file edited. Propose the edit by calling the
`edit_file` tool.

You have read the file already: its contents came back from the `read_file` call
above. You are proposing, not editing — nothing is written until the user accepts.
`old_text` must be an exact substring of that file, copied verbatim, including
indentation — it is what gets replaced. `new_text` is what replaces it. Choose the
smallest replacement that does the job. Do not reprint the whole file. Put any
explanation in the tool's optional `rationale`.
"""

ANSWER_INSTRUCTIONS = """\
The user asked about the project. Answer the question in prose, from the files you
have read.

The files came back from the `read_file` calls above. They were selected because
they were judged relevant, and they are all the context there is. If they do not
contain the answer, say what is missing rather than inventing it. Answer only; do
not propose changes and do not call tools.
"""


def _command_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": RUN_COMMAND,
            "description": (
                "Propose one shell command that fulfils the user's instruction. "
                "The command is not run until the user accepts it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command to run.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "One sentence on what the command does, for the user "
                            "deciding whether to run it."
                        ),
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }


def _edit_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": EDIT_FILE,
            "description": (
                "Propose one edit to the file shown: a literal old_text and the "
                "new_text that replaces it. Nothing is written until the user "
                "accepts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_text": {
                        "type": "string",
                        "description": "An exact substring of the file, copied verbatim.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The text that replaces old_text.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One sentence on what the edit changes.",
                    },
                },
                "required": ["old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    }


def _read_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": READ_FILE,
            "description": (
                "Read one project file and return its contents verbatim. The "
                "files are already read; this declares the call that brought "
                "them back."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file's path, exactly as the request gave it.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


def forced_choice(name: str) -> dict[str, object]:
    """The ``tool_choice`` that makes the model call exactly ``name``.

    Named rather than ``"required"``, so the model cannot answer with a different
    tool and cannot decline.
    """
    return {"type": "function", "function": {"name": name}}


def build_read_transcript(files: Mapping[str, str]) -> list[dict[str, object]]:
    """The reading of every file, hardcoded as the turn a model would have taken.

    One ``read_file`` call per file, each with its own result, rather than the
    whole set pasted into a single message. The model is then handed the shape it
    was trained on — a call and what came back — so the contents arrive as
    evidence it gathered rather than as part of its instructions, and each file
    stays attributed to the call that fetched it.

    The reads have already happened. The workflow decided which files matter, so
    there is no turn for the model to take here; the transcript records the turn
    it did not need to make.

    The result is the file's text exactly as it is on disk: no line numbers, no
    elision, nothing added to make it easier to read. ``old_text`` has to be
    copied out of it character for character, and a single added number would make
    that impossible.
    """
    messages: list[dict[str, object]] = []
    for index, (path, content) in enumerate(files.items()):
        call_id = f"read_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": READ_FILE,
                            "arguments": json.dumps({"path": path}),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    return messages


def build_chat_body(
    *,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[dict[str, object]],
    tool_choice: str | Mapping[str, object],
) -> dict[str, object]:
    """The chat-completions body: the transcript, the tools, and which to call.

    ``tool_choice`` names a function rather than saying ``"required"``, so the
    model must call *that* tool. ``provider.require_parameters`` is what makes
    that reliable: the preset load-balances across backends and at least one
    rejects a named ``tool_choice`` with a 400, while OpenRouter treats
    ``tool_choice`` as a soft preference by default. Requiring parameter support
    keeps the request on a backend that honours it.

    ``parallel_tool_calls`` is deliberately not sent: the same provider rejects
    it outright, and naming the one tool already keeps the answer single.
    """
    return {
        "model": MODEL,
        "messages": list(messages),
        "provider": {"require_parameters": True},
        "tools": list(tools),
        "tool_choice": tool_choice,
    }


def build_command_messages(*, prompt: str, context: Sequence[str] = ()) -> list[dict[str, str]]:
    """Messages for a command: the goal's context, then the prompt itself."""
    return [
        {"role": "system", "content": COMMAND_INSTRUCTIONS},
        {"role": "user", "content": _user_text(prompt=prompt, context=context)},
    ]


def build_edit_messages(
    *, path: str, content: str, prompt: str, context: Sequence[str] = ()
) -> list[dict[str, object]]:
    """Messages for an edit: the request, then the hardcoded read of the file.

    The whole file is read rather than an excerpt because ``old_text`` has to be
    copied from it verbatim; a model that cannot see the indentation cannot
    reproduce it. Reading it through a tool call rather than pasting it into the
    prompt keeps the file as evidence the model gathered, and keeps the request
    the same place the model's own instruction sits.
    """
    return [
        {"role": "system", "content": EDIT_INSTRUCTIONS},
        {"role": "user", "content": _user_text(prompt=prompt, context=context)},
        *build_read_transcript({path: content}),
    ]


def build_answer_messages(
    *, files: Mapping[str, str], prompt: str, context: Sequence[str] = ()
) -> list[dict[str, object]]:
    """Messages for an answer: the request, then a hardcoded read of every file.

    Every selected file is read rather than a snippet of each, because which file
    holds the answer is exactly what the selection could not know. Each read is
    its own call, so the model can attribute what it reads to the path it asked
    for.
    """
    lines = [_user_text(prompt=prompt, context=context)]
    if not files:
        lines += ["", "No files were selected as relevant, so there is nothing to read."]
    return [
        {"role": "system", "content": ANSWER_INSTRUCTIONS},
        {"role": "user", "content": "\n".join(lines)},
        *build_read_transcript(files),
    ]


def _user_text(*, prompt: str, context: Sequence[str]) -> str:
    lines = list(context)
    if not lines or lines[-1] != prompt:
        lines.append(prompt)
    return "\n".join(lines)


def build_command_body(*, prompt: str, context: Sequence[str] = ()) -> dict[str, object]:
    """The command request body. Kept as a named function for tests and logs."""
    return build_chat_body(
        messages=build_command_messages(prompt=prompt, context=context),
        tools=[_command_tool()],
        tool_choice=forced_choice(RUN_COMMAND),
    )


def build_edit_body(
    *, path: str, content: str, prompt: str, context: Sequence[str] = ()
) -> dict[str, object]:
    """The edit request body. Kept as a named function for tests and logs.

    ``read_file`` is declared beside ``edit_file`` because the transcript's reads
    name it; ``edit_file`` is the one forced, so the reads stay history and the
    edit is what comes next.
    """
    return build_chat_body(
        messages=build_edit_messages(path=path, content=content, prompt=prompt, context=context),
        tools=[_read_tool(), _edit_tool()],
        tool_choice=forced_choice(EDIT_FILE),
    )


def build_answer_body(
    *, files: Mapping[str, str], prompt: str, context: Sequence[str] = ()
) -> dict[str, object]:
    """The answer request body: a transcript of reads, and prose as the answer.

    ``tool_choice`` is ``"none"`` rather than a forced function. The reads are
    already in the transcript, so there is no tool left to call, and saying so is
    what keeps the model from spending the turn on another ``read_file`` instead
    of the answer it was asked for. ``read_file`` is still declared, because a
    transcript whose calls name an undeclared tool is a request some backends
    refuse.
    """
    return build_chat_body(
        messages=build_answer_messages(files=files, prompt=prompt, context=context),
        tools=[_read_tool()],
        tool_choice="none",
    )


def _http_problem(status: int, data: object) -> str:
    """Turn an error body into a sentence, keeping the provider's own message."""
    message = ""
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif error:
            message = str(error)
    suffix = f": {message}" if message else ""
    return f"OpenRouter rejected the request (HTTP {status}){suffix}"


def _tool_arguments(data: object, tool_name: str) -> tuple[dict | None, str | None, str | None]:
    """Read the forced tool call out of a response.

    Returns ``(arguments, problem, model)``. A response with no tool call, the
    wrong tool, or arguments that are not JSON is a problem rather than a guess:
    accepting anything less would put a made-up action in front of the user as if
    the model had proposed it.
    """
    if not isinstance(data, dict):
        return None, "OpenRouter returned no JSON object", None
    model = data.get("model") if isinstance(data.get("model"), str) else None

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "OpenRouter returned no choices", model
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None, "OpenRouter returned no message", model

    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None, f"the executor answered without calling {tool_name}", model
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if not isinstance(function, dict):
        return None, "the executor's tool call had no function", model
    if function.get("name") != tool_name:
        return None, f"the executor called {function.get('name')!r}, not {tool_name!r}", model

    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as error:
        return None, f"the executor's arguments were not JSON: {error}", model
    if not isinstance(arguments, dict):
        return None, "the executor's arguments were not an object", model
    return arguments, None, model


def _reason(arguments: dict) -> str | None:
    rationale = arguments.get("rationale")
    return rationale if isinstance(rationale, str) and rationale.strip() else None


def parse_command_proposal(data: object) -> CommandProposal:
    """Read a `run_command` tool call into a command proposal."""
    arguments, problem, model = _tool_arguments(data, RUN_COMMAND)
    if problem is not None:
        return CommandProposal(problem=problem, model=model)
    assert arguments is not None  # guaranteed when problem is None
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return CommandProposal(problem="the executor proposed no command", model=model)
    return CommandProposal(command=command.strip(), rationale=_reason(arguments), model=model)


def parse_edit_proposal(data: object, *, path: str) -> EditProposal:
    """Read an `edit_file` tool call into an edit proposal for ``path``."""
    arguments, problem, model = _tool_arguments(data, EDIT_FILE)
    if problem is not None:
        return EditProposal(path=path, problem=problem, model=model)
    assert arguments is not None  # guaranteed when problem is None
    old_text = arguments.get("old_text")
    new_text = arguments.get("new_text")
    if not isinstance(old_text, str) or not old_text:
        return EditProposal(path=path, problem="the executor proposed no old_text", model=model)
    if not isinstance(new_text, str):
        return EditProposal(path=path, problem="the executor proposed no new_text", model=model)
    return EditProposal(
        path=path,
        old_text=old_text,
        new_text=new_text,
        rationale=_reason(arguments),
        model=model,
    )


def parse_answer(data: object) -> Answer:
    """Read a prose completion into an answer, or say why there is none."""
    if not isinstance(data, dict):
        return Answer(problem="OpenRouter returned no JSON object")
    model = data.get("model") if isinstance(data.get("model"), str) else None

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return Answer(problem="OpenRouter returned no choices", model=model)
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return Answer(problem="OpenRouter returned no message", model=model)

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return Answer(problem="the executor returned no answer", model=model)
    return Answer(text=content.strip(), model=model)


def _safe_json(response) -> object:
    """The response body as JSON, or ``None`` when it is not JSON.

    Error bodies are not guaranteed to be JSON, and a parsing failure there must
    not mask the status code that is the real explanation.
    """
    try:
        return response.json()
    except ValueError:
        return None


class Executor:
    """A reusable connection to OpenRouter, plus the actions xg asks it for."""

    def __init__(self, *, client: object | None = None, model: str | None = None) -> None:
        """Wrap a client, or arrange to build one on first use.

        ``client`` is injectable so tests can exercise the request without a
        network call and without a key. Passing one also transfers ownership of
        closing it to the caller, which is why :meth:`aclose` only closes a client
        this object built.
        """
        self._client = client
        self._model = model
        self._owned = client is None

    @property
    def available(self) -> bool:
        """Whether a client has been built yet. Building one is lazy."""
        return self._client is not None

    def _connection(self):
        """Return the client, building it on first use, or ``None`` without a key.

        The key is resolved here rather than at construction, so xg opens with
        nothing configured. ``None`` is a value the caller reports as a sentence,
        not an exception, because working without a key is a supported state.
        """
        if self._client is None:
            api_key = os.environ.get(ENV_API_KEY) or api_key_from_pass(PASS_ENTRY)
            if not api_key:
                return None
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                headers={"Authorization": f"Bearer {api_key}", "X-Title": "xg"},
                timeout=REQUEST_TIMEOUT,
            )
        return self._client

    async def _request(self, body: dict[str, object]) -> tuple[object | None, str | None]:
        """Post one body and return its JSON, or the sentence explaining why not."""
        client = self._connection()
        if client is None:
            return None, (
                f"no key for the executor: set {ENV_API_KEY} or `pass show {PASS_ENTRY}`"
            )
        if self._model is not None:
            body = {**body, "model": self._model}

        try:
            response = await client.post(CHAT_COMPLETIONS, json=body)
        except httpx.HTTPError as error:
            return None, f"could not reach the executor: {error}"

        if response.status_code >= 400:
            return None, _http_problem(response.status_code, _safe_json(response))
        return _safe_json(response), None

    async def propose_command(
        self, *, prompt: str, context: Sequence[str] = ()
    ) -> CommandProposal:
        """Ask for one command to run, and return it as a proposal."""
        body = build_command_body(prompt=prompt, context=context)
        data, problem = await self._request(body)
        if problem is not None:
            return CommandProposal(problem=problem)
        return parse_command_proposal(data)

    async def propose_edit(
        self, *, path: str, content: str, prompt: str, context: Sequence[str] = ()
    ) -> EditProposal:
        """Ask for one edit to ``content``, and return it as a proposal for ``path``."""
        body = build_edit_body(path=path, content=content, prompt=prompt, context=context)
        data, problem = await self._request(body)
        if problem is not None:
            return EditProposal(path=path, problem=problem)
        # The content the model read is carried on the proposal so its change can
        # be previewed as a patch against exactly what it was shown.
        return replace(parse_edit_proposal(data, path=path), content=content)

    async def answer(
        self, *, files: Mapping[str, str], prompt: str, context: Sequence[str] = ()
    ) -> Answer:
        """Ask for a prose answer from ``files``, and return it as a value."""
        body = build_answer_body(files=files, prompt=prompt, context=context)
        data, problem = await self._request(body)
        if problem is not None:
            return Answer(problem=problem)
        return parse_answer(data)

    async def aclose(self) -> None:
        """Close the connection if this object built it."""
        if self._client is not None and self._owned:
            await self._client.aclose()
            self._client = None
