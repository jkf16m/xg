"""The executor: the requests it sends, and what it does with each answer.

Nothing here touches the network. The fake HTTP client records the body it was
handed and returns a body shaped exactly as OpenRouter's chat-completions
response is, so the tests exercise the real parsing surface — ``choices``,
``message.tool_calls``, the JSON ``arguments`` string — rather than a stand-in
that could drift from it.
"""

from __future__ import annotations

import json

import httpx

from xg_project.llm import (
    ADD_FILE,
    CHAT_COMPLETIONS,
    EDIT_FILE,
    ENV_API_KEY,
    MODEL,
    READ_FILE,
    RUN_COMMAND,
    AddProposal,
    Answer,
    CommandProposal,
    EditProposal,
    Executor,
    build_add_body,
    build_answer_body,
    build_command_body,
    build_edit_body,
    parse_add_proposal,
    parse_command_proposal,
    parse_edit_proposal,
)


def tool(body: dict, name: str) -> dict:
    """The declared tool called ``name``, from a request body."""
    return next(entry for entry in body["tools"] if entry["function"]["name"] == name)


class FakeResponse:
    """The httpx response surface ``Executor`` reads: status and JSON."""

    def __init__(self, status_code: int = 200, payload: object = None, *, body: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self._body = body

    def json(self):
        if self._body is not None:
            raise ValueError("not JSON")
        return self._payload


class FakeHttp:
    """Stands in for ``httpx.AsyncClient``, recording each request."""

    def __init__(self, *, response: FakeResponse | None = None, raises: Exception | None = None):
        self.response = response if response is not None else FakeResponse(200, command_response())
        self.raises = raises
        self.calls: list[dict] = []
        self.closed = False

    async def post(self, path: str, json=None):
        self.calls.append({"path": path, "json": json})
        if self.raises is not None:
            raise self.raises
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def tool_response(name: str, arguments: dict, *, model: str = "deepseek/deepseek-v4.1-flash") -> dict:
    """An OpenRouter chat-completions body carrying one forced tool call."""
    return {
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            }
        ],
    }


def command_response(command: str = "echo hi", rationale: str | None = "prints hi") -> dict:
    arguments = {"command": command}
    if rationale is not None:
        arguments["rationale"] = rationale
    return tool_response(RUN_COMMAND, arguments)


def answer_response(text: str = "it is a project", model: str = "deepseek/deepseek-v4.1-flash") -> dict:
    """An OpenRouter body carrying a plain prose completion, with no tool call."""
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }


def edit_response(
    old_text: str = "x = 1", new_text: str = "x = 2", rationale: str | None = "bump the value"
) -> dict:
    arguments = {"old_text": old_text, "new_text": new_text}
    if rationale is not None:
        arguments["rationale"] = rationale
    return tool_response(EDIT_FILE, arguments)


def add_response(
    path: str = "pkg/greet.py",
    content: str = 'def greet():\n    return "hi"\n',
    rationale: str | None = "a new module",
) -> dict:
    arguments = {"path": path, "content": content}
    if rationale is not None:
        arguments["rationale"] = rationale
    return tool_response(ADD_FILE, arguments)


async def propose_command(executor: Executor, **overrides) -> CommandProposal:
    kwargs = {"prompt": "list the files"}
    return await executor.propose_command(**{**kwargs, **overrides})


async def propose_edit(executor: Executor, **overrides) -> EditProposal:
    kwargs = {"path": "src/a.py", "content": "x = 1\n", "prompt": "make it two"}
    return await executor.propose_edit(**{**kwargs, **overrides})


async def propose_add(executor: Executor, **overrides) -> AddProposal:
    kwargs = {"prompt": "add a module for greeting"}
    return await executor.propose_add(**{**kwargs, **overrides})


async def ask(executor: Executor, **overrides) -> Answer:
    kwargs = {"files": {"a.py": "x = 1\n"}, "prompt": "what is this?"}
    return await executor.answer(**{**kwargs, **overrides})


# -- the command request ---------------------------------------------------


def test_the_command_body_forces_exactly_the_command_tool() -> None:
    """Forced by name, not merely required, so the model cannot choose another."""
    body = build_command_body(prompt="list the files")
    assert body["tool_choice"] == {"type": "function", "function": {"name": RUN_COMMAND}}
    assert [tool["function"]["name"] for tool in body["tools"]] == [RUN_COMMAND]


def test_parallel_tool_calls_is_not_sent() -> None:
    """The provider behind the preset rejects the parameter outright (HTTP 400)."""
    assert "parallel_tool_calls" not in build_command_body(prompt="x")


def test_only_providers_that_support_the_forced_call_are_used() -> None:
    """Otherwise the preset can be routed to a backend that rejects tool_choice."""
    assert build_command_body(prompt="x")["provider"] == {"require_parameters": True}


def test_the_model_is_the_hardcoded_preset() -> None:
    assert build_command_body(prompt="x")["model"] == MODEL
    assert MODEL == "@preset/deepseek"


def test_the_command_tool_declares_the_command_and_an_optional_rationale() -> None:
    tool = build_command_body(prompt="x")["tools"][0]["function"]
    assert set(tool["parameters"]["properties"]) == {"command", "rationale"}
    assert tool["parameters"]["required"] == ["command"]


def test_the_context_comes_before_the_prompt() -> None:
    messages = build_command_body(prompt="do the thing", context=["the goal is x"])["messages"]
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"].startswith("the goal is x")
    assert messages[-1]["content"].endswith("do the thing")


# -- the edit request ------------------------------------------------------


def test_the_edit_body_forces_exactly_the_edit_tool() -> None:
    body = build_edit_body(path="src/a.py", content="x = 1\n", prompt="make it two")
    assert body["tool_choice"] == {"type": "function", "function": {"name": EDIT_FILE}}
    names = [entry["function"]["name"] for entry in body["tools"]]
    assert names == [READ_FILE, EDIT_FILE]


def test_the_edit_tool_needs_only_old_and_new_text() -> None:
    parameters = tool(build_edit_body(path="a.py", content="x", prompt="y"), EDIT_FILE)[
        "function"
    ]["parameters"]
    assert set(parameters["properties"]) == {"old_text", "new_text", "rationale"}
    assert parameters["required"] == ["old_text", "new_text"]


# -- the add request -------------------------------------------------------


def test_the_add_body_forces_exactly_the_add_tool() -> None:
    body = build_add_body(prompt="add a module for greeting")
    assert body["tool_choice"] == {"type": "function", "function": {"name": ADD_FILE}}
    assert [tool["function"]["name"] for tool in body["tools"]] == [ADD_FILE]


def test_the_add_tool_needs_a_path_and_the_whole_content() -> None:
    parameters = tool(build_add_body(prompt="add it"), ADD_FILE)["function"]["parameters"]
    assert parameters["required"] == ["path", "content"]
    assert set(parameters["properties"]) == {"path", "content", "rationale"}


def test_an_add_declares_no_reads_because_there_is_nothing_to_read() -> None:
    """The file does not exist yet, so a transcript of reads would be a fiction."""
    body = build_add_body(prompt="add a module for greeting")
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert "read_file" not in json.dumps(body)


async def test_an_add_tool_call_becomes_a_new_file_proposal() -> None:
    client = FakeHttp(response=FakeResponse(200, add_response()))
    proposal = await propose_add(Executor(client=client))

    assert proposal.ok
    assert isinstance(proposal, AddProposal)
    assert proposal.path == "pkg/greet.py"
    assert proposal.content == 'def greet():\n    return "hi"\n'
    assert proposal.rationale == "a new module"
    assert proposal.tool == ADD_FILE


async def test_an_add_without_a_path_is_a_problem() -> None:
    payload = tool_response(ADD_FILE, {"content": "x = 1\n"})
    proposal = await propose_add(Executor(client=FakeHttp(response=FakeResponse(200, payload))))
    assert not proposal.ok
    assert "no path" in proposal.problem


async def test_a_blank_add_path_is_a_problem() -> None:
    payload = tool_response(ADD_FILE, {"path": "   ", "content": "x = 1\n"})
    proposal = await propose_add(Executor(client=FakeHttp(response=FakeResponse(200, payload))))
    assert not proposal.ok
    assert "no path" in proposal.problem


async def test_an_add_without_content_is_a_problem() -> None:
    payload = tool_response(ADD_FILE, {"path": "pkg/greet.py"})
    proposal = await propose_add(Executor(client=FakeHttp(response=FakeResponse(200, payload))))
    assert not proposal.ok
    assert "no content" in proposal.problem


async def test_an_empty_file_is_still_a_proposal() -> None:
    """``__init__.py`` is a real thing to add, so empty content is not a refusal."""
    payload = tool_response(ADD_FILE, {"path": "pkg/__init__.py", "content": ""})
    proposal = await propose_add(Executor(client=FakeHttp(response=FakeResponse(200, payload))))
    assert proposal.ok
    assert proposal.content == ""


def test_an_add_is_previewed_as_creating_a_file() -> None:
    proposal = AddProposal(path="pkg/greet.py", content='def greet():\n    return "hi"\n')
    assert proposal.preview() == "add pkg/greet.py: 2 lines"


def test_an_empty_addition_says_it_is_empty_rather_than_counting_a_line() -> None:
    assert AddProposal(path="pkg/__init__.py", content="").preview() == (
        "add pkg/__init__.py: an empty file"
    )


def test_an_addition_without_a_path_or_content_is_not_ok() -> None:
    assert not AddProposal(content="x = 1\n").ok
    assert not AddProposal(path="pkg/greet.py").ok


def test_an_addition_renders_as_a_patch_that_creates_the_file() -> None:
    proposal = AddProposal(path="pkg/greet.py", content='def greet():\n    return "hi"\n')
    patch = proposal.detail()

    assert patch.startswith("diff --git a/pkg/greet.py b/pkg/greet.py\n")
    assert "new file mode 100644" in patch
    assert "index 0000000.." in patch
    assert "--- /dev/null" in patch
    assert '+def greet():' in patch


def test_an_empty_addition_draws_no_patch() -> None:
    """No hunks, so nothing to preview; the preview line is the whole of it."""
    assert AddProposal(path="pkg/__init__.py", content="").detail() == ""


def test_a_command_proposal_has_no_detail() -> None:
    """Only the file-changing proposals have more to show than one line."""
    assert CommandProposal(command="echo hi").detail() == ""


def test_parse_add_refuses_a_non_object() -> None:
    proposal = parse_add_proposal("not a body")
    assert not proposal.ok


def test_the_file_is_read_by_a_hardcoded_tool_call_rather_than_pasted_in() -> None:
    """The file arrives as the result of a call, not as part of the instructions."""
    body = build_edit_body(path="src/a.py", content="    x = 1\n", prompt="make it two")

    call, result = body["messages"][-2], body["messages"][-1]
    assert call["role"] == "assistant" and call["content"] is None
    (tool_call,) = call["tool_calls"]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == READ_FILE
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "src/a.py"}
    assert result["role"] == "tool"
    assert result["tool_call_id"] == tool_call["id"]
    assert result["content"] == "    x = 1\n"


def test_the_file_is_not_in_the_prompt_at_all() -> None:
    """The bytes are the tool result and nowhere else, or the read is decoration."""
    body = build_edit_body(path="src/a.py", content="    x = 1\n", prompt="make it two")
    prompt = body["messages"][1]["content"]
    assert "x = 1" not in prompt
    assert prompt.endswith("make it two")


def test_the_read_result_is_the_file_verbatim_with_no_line_numbers() -> None:
    """`old_text` is copied out of it, so a single added number would break that."""
    content = "def f():\n\treturn 1\n"
    body = build_answer_body(files={"a.py": content}, prompt="q")
    assert body["messages"][3]["content"] == content


# -- the answer request ----------------------------------------------------


def test_the_answer_forbids_a_tool_call_rather_than_forcing_one() -> None:
    """An answer is prose, and the reads it needed are already in the transcript."""
    body = build_answer_body(files={"a.py": "x"}, prompt="q")
    assert body["tool_choice"] == "none"
    assert [entry["function"]["name"] for entry in body["tools"]] == [READ_FILE]
    assert body["model"] == MODEL


def test_every_selected_file_gets_its_own_read_call_and_result() -> None:
    body = build_answer_body(files={"a.py": "x = 1\n", "b.py": "y = 2\n"}, prompt="q")
    transcript = body["messages"][2:]

    assert [message["role"] for message in transcript] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    first = transcript[0]["tool_calls"][0]
    second = transcript[2]["tool_calls"][0]
    assert json.loads(first["function"]["arguments"]) == {"path": "a.py"}
    assert json.loads(second["function"]["arguments"]) == {"path": "b.py"}
    assert first["id"] != second["id"]
    assert transcript[1]["tool_call_id"] == first["id"]
    assert transcript[1]["content"] == "x = 1\n"
    assert transcript[3]["tool_call_id"] == second["id"]
    assert transcript[3]["content"] == "y = 2\n"


def test_the_read_calls_are_deterministic() -> None:
    """Same files, same request: a transcript that shifts between runs is untestable."""
    files = {"a.py": "x", "b.py": "y"}
    first = build_answer_body(files=files, prompt="q")["messages"]
    assert first == build_answer_body(files=files, prompt="q")["messages"]


def test_no_selected_files_is_said_rather_than_left_blank() -> None:
    body = build_answer_body(files={}, prompt="q")
    assert "nothing to read" in body["messages"][1]["content"]
    assert body["messages"][2:] == []


async def test_a_prose_completion_becomes_an_answer() -> None:
    client = FakeHttp(response=FakeResponse(200, answer_response("xg runs workflows")))
    answer = await ask(Executor(client=client))
    assert answer.ok
    assert answer.text == "xg runs workflows"
    assert answer.model == "deepseek/deepseek-v4.1-flash"


async def test_an_answer_is_stripped() -> None:
    client = FakeHttp(response=FakeResponse(200, answer_response("  hi  ")))
    assert (await ask(Executor(client=client))).text == "hi"


async def test_an_empty_answer_is_a_problem() -> None:
    client = FakeHttp(response=FakeResponse(200, answer_response("   ")))
    answer = await ask(Executor(client=client))
    assert not answer.ok
    assert "no answer" in answer.problem


async def test_a_connection_error_on_an_answer_is_a_problem_not_a_raise() -> None:
    client = FakeHttp(raises=httpx.ConnectError("refused"))
    answer = await ask(Executor(client=client))
    assert not answer.ok
    assert "could not reach the executor" in answer.problem


def test_an_answer_renders_as_its_own_text() -> None:
    assert str(Answer(text="hi")) == "hi"
    assert Answer(problem="boom").explain() == "boom"
    assert not Answer(problem="boom").ok


async def test_the_request_goes_to_chat_completions() -> None:
    client = FakeHttp()
    await propose_command(Executor(client=client))
    assert client.calls[0]["path"] == CHAT_COMPLETIONS


async def test_a_model_override_replaces_the_preset() -> None:
    client = FakeHttp()
    await propose_command(Executor(client=client, model="openai/gpt-4o"))
    assert client.calls[0]["json"]["model"] == "openai/gpt-4o"


# -- the happy paths -------------------------------------------------------


async def test_a_command_tool_call_becomes_a_command_and_a_rationale() -> None:
    proposal = await propose_command(Executor(client=FakeHttp()))
    assert proposal.ok
    assert isinstance(proposal, CommandProposal)
    assert proposal.command == "echo hi"
    assert proposal.rationale == "prints hi"
    assert proposal.model == "deepseek/deepseek-v4.1-flash"
    assert proposal.problem is None


async def test_a_command_is_stripped() -> None:
    payload = command_response(command="  echo hi  ")
    proposal = await propose_command(Executor(client=FakeHttp(response=FakeResponse(200, payload))))
    assert proposal.command == "echo hi"


async def test_an_edit_tool_call_becomes_an_edit_for_the_named_file() -> None:
    client = FakeHttp(response=FakeResponse(200, edit_response()))
    proposal = await propose_edit(Executor(client=client), path="src/a.py")
    assert proposal.ok
    assert isinstance(proposal, EditProposal)
    assert proposal.path == "src/a.py"
    assert proposal.old_text == "x = 1"
    assert proposal.new_text == "x = 2"
    assert proposal.rationale == "bump the value"


def test_a_missing_rationale_is_not_invented() -> None:
    proposal = parse_command_proposal(command_response(rationale=None))
    assert proposal.command == "echo hi"
    assert proposal.rationale is None


# -- refusals, none of which are exceptions --------------------------------


async def test_a_connection_error_becomes_a_problem_not_a_raise() -> None:
    client = FakeHttp(raises=httpx.ConnectError("refused"))
    proposal = await propose_command(Executor(client=client))
    assert not proposal.ok
    assert "could not reach the executor" in proposal.problem


async def test_an_http_error_keeps_the_providers_message() -> None:
    body = {"error": {"message": "no credits"}}
    client = FakeHttp(response=FakeResponse(402, body))
    proposal = await propose_command(Executor(client=client))
    assert not proposal.ok
    assert "HTTP 402" in proposal.problem
    assert "no credits" in proposal.problem


async def test_a_non_json_error_body_still_reports_the_status() -> None:
    client = FakeHttp(response=FakeResponse(500, body="boom"))
    proposal = await propose_command(Executor(client=client))
    assert not proposal.ok
    assert "HTTP 500" in proposal.problem


async def test_prose_instead_of_a_tool_call_is_a_problem() -> None:
    payload = {"model": "m", "choices": [{"message": {"role": "assistant", "content": "here you go"}}]}
    proposal = await propose_command(Executor(client=FakeHttp(response=FakeResponse(200, payload))))
    assert not proposal.ok
    assert "without calling" in proposal.problem


async def test_the_wrong_tool_is_a_problem() -> None:
    payload = command_response()
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "other"
    proposal = await propose_command(Executor(client=FakeHttp(response=FakeResponse(200, payload))))
    assert not proposal.ok
    assert "not 'run_command'" in proposal.problem


async def test_arguments_that_are_not_json_are_a_problem() -> None:
    payload = command_response()
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    proposal = await propose_command(Executor(client=FakeHttp(response=FakeResponse(200, payload))))
    assert not proposal.ok
    assert "not JSON" in proposal.problem


async def test_an_empty_command_is_a_problem() -> None:
    payload = command_response(command="   ")
    proposal = await propose_command(Executor(client=FakeHttp(response=FakeResponse(200, payload))))
    assert not proposal.ok
    assert "no command" in proposal.problem


async def test_an_edit_without_old_text_is_a_problem() -> None:
    proposal = parse_edit_proposal(tool_response(EDIT_FILE, {"old_text": "", "new_text": "y"}), path="a.py")
    assert not proposal.ok
    assert "no old_text" in proposal.problem


async def test_an_edit_without_new_text_is_a_problem() -> None:
    proposal = parse_edit_proposal(tool_response(EDIT_FILE, {"old_text": "x"}), path="a.py")
    assert not proposal.ok
    assert "no new_text" in proposal.problem


async def test_no_choices_is_a_problem() -> None:
    proposal = await propose_command(
        Executor(client=FakeHttp(response=FakeResponse(200, {"model": "m", "choices": []})))
    )
    assert not proposal.ok
    assert "no choices" in proposal.problem


async def test_a_missing_key_is_a_problem_and_costs_no_request(monkeypatch) -> None:
    """xg opens and is usable with no key; proposing is what fails."""
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.setattr("xg_project.llm.api_key_from_pass", lambda entry="pi/openrouter": None)
    proposal = await propose_command(Executor(client=None))
    assert not proposal.ok
    assert ENV_API_KEY in proposal.problem
    assert "pi/openrouter" in proposal.problem


def test_parse_command_refuses_a_non_object() -> None:
    assert not parse_command_proposal([]).ok
    assert not parse_command_proposal(None).ok


# -- lifecycle -------------------------------------------------------------


async def test_an_injected_client_is_not_closed() -> None:
    """Passing a client in transfers ownership of closing it to the caller."""
    client = FakeHttp()
    executor = Executor(client=client)
    await propose_command(executor)
    await executor.aclose()
    assert client.closed is False


async def test_a_client_built_for_a_key_is_closed(monkeypatch) -> None:
    monkeypatch.setenv(ENV_API_KEY, "env-key")
    built: list = []

    class StubHttp:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.closed = False
            built.append(self)

        async def post(self, path, json=None):
            return FakeResponse(200, command_response())

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr("xg_project.llm.httpx.AsyncClient", StubHttp)

    executor = Executor()
    assert (await propose_command(executor)).ok
    assert len(built) == 1
    assert built[0].kwargs["headers"]["Authorization"] == "Bearer env-key"

    await executor.aclose()
    assert built[0].closed is True
    assert executor._client is None


def test_preview_reads_as_a_sentence_for_the_state_panel() -> None:
    assert CommandProposal(command="echo hi").preview() == "echo hi"
    assert "edit a.py" in EditProposal(path="a.py", old_text="x", new_text="y").preview()
    assert CommandProposal(problem="boom").preview() == "boom"


def test_available_is_false_until_a_proposal_needs_a_client() -> None:
    assert Executor().available is False
