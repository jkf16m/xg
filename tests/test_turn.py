"""One turn: a node introduces its state, and the run follows its edge.

The point of these tests is the seam. A branching node asks Jev; a single child
is the edge and nobody is asked; a proposing leaf hands back a gate instead of
acting. A routing that fails must leave the session exactly where it was, with
the node's state already recorded, so the user can retype rather than being
dropped somewhere arbitrary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xg_project.graph import (
    ADD,
    ANSWER,
    COMMAND,
    EDIT,
    FILTER,
    ORIGIN,
    SELECT_MODULE,
    SORT,
    default_graph,
)
from xg_project.jev import (
    IMPORTANCE_CRITERIA,
    SELECT_CRITERIA,
    SELECT_INSTRUCTIONS,
    Routing,
    Selection,
)
from xg_project.llm import AddProposal, Answer, CommandProposal, EditProposal
from xg_project.session import Session
from xg_project.turn import ACCEPTED, REJECTED, take_turn


class FakeJev:
    """Stands in for `Jev`: a fixed route, kept set, and relevance scores."""

    def __init__(self, *, choice: str | None = None, choices: dict[str, str] | None = None,
                 scores: dict[str, float] | None = None,
                 kept: set[str] | None = None, decide_problem: str | None = None) -> None:
        self.choice = choice
        self.choices = choices or {}
        self.scores = scores or {}
        self.kept = kept
        self.decide_problem = decide_problem
        self.decide_calls: list[dict] = []
        self.select_calls: list[dict] = []

    async def decide(self, *, current, context, prompt, options):
        self.decide_calls.append(
            {"current": current, "context": list(context), "prompt": prompt, "options": dict(options)}
        )
        if self.decide_problem is not None:
            return Routing(problem=self.decide_problem)
        # An explicit choice only applies where it is actually on offer; otherwise
        # the first option stands, so one fake can serve every branch.
        explicit = self.choices.get(current, self.choice)
        choice = explicit if explicit in options else next(iter(options))
        return Routing(node=choice, confidence=0.9, model="fake")

    async def select(self, *, files, prompt, threshold=0.85, instructions=None, criteria=None):
        # Substitute the same defaults the real Jev does, so a test can assert
        # which question was actually asked.
        self.select_calls.append(
            {
                "files": dict(files),
                "prompt": prompt,
                # The template, not a formatted question: the real select formats
                # it once per file with the path, which is the point being tested.
                "instructions": instructions if instructions is not None else SELECT_INSTRUCTIONS,
                "criteria": criteria if criteria is not None else SELECT_CRITERIA,
            }
        )
        kept = {
            path: content
            for path, content in files.items()
            if self.kept is None or path in self.kept
        }
        return Selection(
            files=kept,
            scores={path: self.scores.get(path, 0.5) for path in files},
            model="fake",
        )

    async def aclose(self) -> None:
        pass


class FakeExecutor:
    """Stands in for `Executor`: a fixed command, edit, and answer."""

    def __init__(self, *, command: str = "echo hi", old: str = "x = 1", new: str = "x = 2",
                 problem: str | None = None, answer: str = "it is a project",
                 path: str = "pkg/new_mod.py", content: str = "def f():\n    pass\n") -> None:
        self.command = command
        self.old = old
        self.new = new
        self.problem = problem
        self.reply = answer
        self.path = path
        self.content = content
        self.command_calls: list[str] = []
        self.edit_calls: list[dict] = []
        self.add_calls: list[dict] = []
        self.answer_calls: list[dict] = []

    async def propose_add(self, *, prompt, context=()):
        self.add_calls.append({"prompt": prompt, "context": list(context)})
        if self.problem is not None:
            return AddProposal(problem=self.problem)
        return AddProposal(path=self.path, content=self.content, rationale="a new file")

    async def propose_command(self, *, prompt, context=()):
        self.command_calls.append(prompt)
        if self.problem is not None:
            return CommandProposal(problem=self.problem)
        return CommandProposal(command=self.command, rationale="prints hi")

    async def propose_edit(self, *, path, content, prompt, context=()):
        self.edit_calls.append({"path": path, "content": content, "prompt": prompt})
        if self.problem is not None:
            return EditProposal(path=path, problem=self.problem)
        return EditProposal(
            path=path,
            old_text=self.old,
            new_text=self.new,
            rationale="change it",
            # The real executor carries the content it was shown, so the change can
            # be previewed as a patch; a fake that dropped it would hide that.
            content=content,
        )

    async def answer(self, *, files, prompt, context=()):
        self.answer_calls.append({"files": dict(files), "prompt": prompt})
        if self.problem is not None:
            return Answer(problem=self.problem)
        return Answer(text=self.reply, model="fake")

    async def aclose(self) -> None:
        pass


@pytest.fixture
def session() -> Session:
    return Session(default_graph())


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def ignored_tree(tmp_path: Path) -> Path:
    """A tree where the ignore files exclude files, as a project's would."""
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / ".xgignore").write_text("secret.py\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "secret.py").write_text("z = 3\n", encoding="utf-8")
    return tmp_path


# -- the origin and routing ------------------------------------------------


async def test_the_origin_records_the_goal_as_its_state(session: Session) -> None:
    turn = await take_turn(session, "edit a file", jev=FakeJev(choice=SELECT_MODULE))
    assert session.state[ORIGIN] == "the user wants: edit a file"
    assert session.goal == "the user wants: edit a file"
    assert turn.produced == "the user wants: edit a file"


async def test_a_branching_node_offers_its_children_and_moves_where_jev_says(
    session: Session,
) -> None:
    jev = FakeJev(choice=SELECT_MODULE)
    turn = await take_turn(session, "edit a file", jev=jev)

    assert set(jev.decide_calls[0]["options"]) == {ADD, COMMAND, SELECT_MODULE}
    assert turn.moved is not None and turn.moved.ok
    assert turn.descended
    assert session.position == SELECT_MODULE
    assert session.trail == (ORIGIN, SELECT_MODULE)


async def test_jev_sees_the_state_the_node_just_introduced(session: Session) -> None:
    jev = FakeJev(choice=SELECT_MODULE)
    await take_turn(session, "edit a file", jev=jev)
    assert f"{ORIGIN}: the user wants: edit a file" in jev.decide_calls[0]["context"]


async def test_a_failed_routing_leaves_the_session_where_it_was(session: Session) -> None:
    jev = FakeJev(decide_problem="could not reach Jev")
    turn = await take_turn(session, "edit a file", jev=jev)

    assert turn.routing is not None and not turn.routing.ok
    assert turn.moved is None
    assert not turn.descended
    assert session.position == ORIGIN
    # The node did its job before Jev was asked; nothing rolls that back.
    assert session.state[ORIGIN] == "the user wants: edit a file"


async def test_no_jev_reports_that_it_cannot_route(session: Session) -> None:
    turn = await take_turn(session, "edit a file")
    assert turn.routing is not None and not turn.routing.ok
    assert "no Jev" in turn.routing.problem
    assert session.position == ORIGIN


# -- filter: the deterministic half ----------------------------------------


async def test_filter_reads_the_tree_and_records_the_files(session: Session, tree: Path) -> None:
    session.move(FILTER)
    turn = await take_turn(session, "go", jev=FakeJev(kept={"a.py", "b.py"}), root=tree)

    assert set(session.state[FILTER]["files"]) == {"a.py", "b.py"}
    assert turn.moved is not None and turn.moved.to == SORT


async def test_filter_asks_jev_whether_each_file_would_help(session: Session, tree: Path) -> None:
    session.move(FILTER)
    jev = FakeJev(kept={"a.py"})
    await take_turn(session, "go", jev=jev, root=tree)

    assert set(jev.select_calls[0]["files"]) == {"a.py", "b.py"}
    assert "help" in jev.select_calls[0]["instructions"].lower()
    assert jev.select_calls[0]["criteria"] is SELECT_CRITERIA


async def test_filter_ignores_before_it_asks_jev(session: Session, ignored_tree: Path) -> None:
    """The deterministic stage runs first: ignored files never reach the model."""
    session.move(FILTER)
    jev = FakeJev(kept={"a.py"})
    await take_turn(session, "go", jev=jev, root=ignored_tree)

    assert set(jev.select_calls[0]["files"]) == {"a.py"}
    assert session.state[FILTER]["read"] == 1
    assert set(session.state[FILTER]["files"]) == {"a.py"}


async def test_the_filters_state_shows_both_halves_of_the_step(
    session: Session, ignored_tree: Path
) -> None:
    """What the read produced, and what the model kept, are both in the state."""
    session.move(FILTER)
    (ignored_tree / "c.py").write_text("w = 4\n", encoding="utf-8")
    await take_turn(session, "go", jev=FakeJev(kept={"a.py"}), root=ignored_tree)

    assert session.state[FILTER]["read"] == 2
    assert set(session.state[FILTER]["files"]) == {"a.py"}


async def test_filter_keeps_only_the_files_jev_found_relevant(session: Session, tree: Path) -> None:
    session.move(FILTER)
    await take_turn(session, "go", jev=FakeJev(kept={"b.py"}), root=tree)
    assert session.state[FILTER]["read"] == 2
    assert set(session.state[FILTER]["files"]) == {"b.py"}
    assert session.state[FILTER]["scores"] == {"a.py": 0.5, "b.py": 0.5}


async def test_a_single_child_is_taken_without_asking_jev(session: Session, tree: Path) -> None:
    """Filter has exactly one next node, so there is nothing to decide."""
    session.move(FILTER)
    jev = FakeJev()
    turn = await take_turn(session, "go", jev=jev, root=tree)

    assert jev.decide_calls == []
    assert turn.routing is None
    assert session.position == SORT


async def test_filter_without_a_root_reports_a_problem(session: Session) -> None:
    session.move(FILTER)
    await take_turn(session, "go", root=None)
    assert session.state[FILTER]["problem"] == "no project root is configured"


async def test_filter_without_jev_reports_a_problem(session: Session, tree: Path) -> None:
    """Nothing survives a selection that could not be made, which is the safe direction."""
    session.move(FILTER)
    await take_turn(session, "go", root=tree)
    assert session.state[FILTER]["files"] == {}
    assert "no Jev" in session.state[FILTER]["problem"]


# -- sort: the model half of selection -------------------------------------


async def test_sort_ranks_every_filtered_file_most_relevant_first(session: Session) -> None:
    session.move(SORT)
    session.record(FILTER, {"files": {"a.py": "x", "b.py": "y"}})
    jev = FakeJev(scores={"a.py": 0.2, "b.py": 0.9})
    await take_turn(session, "go", jev=jev)

    assert list(session.state[SORT]["files"]) == ["b.py", "a.py"]
    assert session.state[SORT]["scores"] == {"a.py": 0.2, "b.py": 0.9}


async def test_sort_keeps_every_file_and_only_reorders(session: Session) -> None:
    """Edit may be asked for a file that is not the single most relevant one."""
    session.move(SORT)
    session.record(FILTER, {"files": {"a.py": "x", "b.py": "y"}})
    await take_turn(session, "go", jev=FakeJev(scores={"a.py": 0.2, "b.py": 0.9}))
    assert set(session.state[SORT]["files"]) == {"a.py", "b.py"}


async def test_sort_forks_and_moves_where_jev_says(session: Session) -> None:
    """Sort now has two leaves: the run either answers the question or edits."""
    session.move(SORT)
    session.record(FILTER, {"files": {"a.py": "x"}})
    jev = FakeJev(choice=EDIT)
    turn = await take_turn(session, "go", jev=jev)

    assert set(jev.decide_calls[0]["options"]) == {EDIT, ANSWER}
    assert session.position == EDIT
    assert turn.moved is not None and turn.moved.to == EDIT


async def test_sort_can_route_to_the_answer_leaf(session: Session) -> None:
    session.move(SORT)
    session.record(FILTER, {"files": {"a.py": "x"}})
    turn = await take_turn(session, "what is this?", jev=FakeJev(choice=ANSWER))
    assert session.position == ANSWER
    assert turn.moved is not None and turn.moved.to == ANSWER


async def test_sort_asks_jev_a_different_question_than_filter(session: Session, tree: Path) -> None:
    """Filter asks whether a file belongs; sort asks which of the survivors matters."""
    session.move(FILTER)
    jev = FakeJev(kept={"a.py", "b.py"})
    await take_turn(session, "go", jev=jev, root=tree)
    await take_turn(session, "go", jev=jev)

    relevance, importance = jev.select_calls
    assert importance["criteria"] is IMPORTANCE_CRITERIA
    assert importance["instructions"] != relevance["instructions"]
    assert set(importance["files"]) == {"a.py", "b.py"}


async def test_sort_only_ranks_what_filter_kept(session: Session, tree: Path) -> None:
    session.move(FILTER)
    jev = FakeJev(kept={"b.py"})
    await take_turn(session, "go", jev=jev, root=tree)
    await take_turn(session, "go", jev=jev)
    assert set(jev.select_calls[1]["files"]) == {"b.py"}


async def test_sort_without_jev_reports_a_problem(session: Session) -> None:
    session.move(SORT)
    session.record(FILTER, {"files": {"a.py": "x"}})
    await take_turn(session, "go")
    assert "no Jev" in session.state[SORT]["problem"]


# -- the gated leaves ------------------------------------------------------


async def test_command_ends_the_turn_on_a_gate(session: Session) -> None:
    session.move(COMMAND)
    executor = FakeExecutor(command="git log -1")
    turn = await take_turn(session, "show the last commit", executor=executor)

    assert turn.awaiting
    assert turn.gate is not None
    assert turn.gate.proposal.command == "git log -1"
    assert session.state[COMMAND] is turn.gate
    assert session.position == COMMAND


async def test_add_ends_the_turn_on_a_gate_for_a_new_file(session: Session) -> None:
    session.move(ADD)
    executor = FakeExecutor(path="pkg/new_mod.py", content="def f():\n    pass\n")
    turn = await take_turn(session, "add a module for f", executor=executor)

    assert turn.awaiting
    assert turn.gate is not None
    assert turn.proposal is not None
    assert turn.proposal.path == "pkg/new_mod.py"
    assert session.state[ADD] is turn.gate
    assert session.position == ADD


async def test_add_reads_nothing_and_the_project_is_not_consulted(session: Session) -> None:
    """There is nothing to read: the file being asked for does not exist yet."""
    session.move(ADD)
    executor = FakeExecutor()
    await take_turn(session, "add a new file", executor=executor)

    assert executor.add_calls[0]["prompt"] == "add a new file"
    assert executor.edit_calls == []
    assert executor.answer_calls == []


async def test_add_without_an_executor_reports_a_problem(session: Session) -> None:
    session.move(ADD)
    turn = await take_turn(session, "add a new file")

    assert turn.awaiting
    assert turn.proposal is not None and not turn.proposal.ok
    assert "no executor" in turn.proposal.problem


async def test_edit_uses_the_file_sort_ranked_first(session: Session) -> None:
    session.move(EDIT)
    session.record(SORT, {"files": {"b.py": "y = 2\n", "a.py": "x = 1\n"}})
    executor = FakeExecutor()
    turn = await take_turn(session, "make it two", executor=executor)

    assert turn.awaiting
    assert executor.edit_calls[0]["path"] == "b.py"
    assert executor.edit_calls[0]["content"] == "y = 2\n"
    assert turn.proposal.path == "b.py"


async def test_edit_with_an_empty_ranking_reports_a_problem_not_a_guess(session: Session) -> None:
    session.move(EDIT)
    session.record(SORT, {"files": {}})
    turn = await take_turn(session, "make it two", executor=FakeExecutor())

    assert turn.awaiting
    assert turn.proposal is not None and not turn.proposal.ok
    assert "no file was ranked" in turn.proposal.problem


# -- answer: the read-only leaf --------------------------------------------


async def test_answer_reads_the_files_sort_ranked(session: Session) -> None:
    session.move(ANSWER)
    session.record(SORT, {"files": {"b.py": "y = 2\n", "a.py": "x = 1\n"}})
    executor = FakeExecutor(answer="a project about x")
    turn = await take_turn(session, "what is this?", executor=executor)

    assert executor.answer_calls[0]["files"] == {"b.py": "y = 2\n", "a.py": "x = 1\n"}
    assert session.state[ANSWER].text == "a project about x"
    assert not turn.awaiting
    assert session.position == ANSWER


async def test_answer_falls_back_to_the_filtered_files(session: Session) -> None:
    """Walked to directly, it still answers from what FILTER kept."""
    session.move(ANSWER)
    session.record(FILTER, {"files": {"a.py": "x = 1\n"}})
    executor = FakeExecutor()
    await take_turn(session, "what is this?", executor=executor)
    assert executor.answer_calls[0]["files"] == {"a.py": "x = 1\n"}


async def test_answer_without_files_reports_a_problem_not_an_invention(session: Session) -> None:
    session.move(ANSWER)
    turn = await take_turn(session, "what is this?", executor=FakeExecutor())
    assert session.state[ANSWER].problem is not None
    assert "nothing to answer from" in session.state[ANSWER].problem
    assert not turn.awaiting


async def test_answer_without_an_executor_says_so(session: Session) -> None:
    session.move(ANSWER)
    session.record(SORT, {"files": {"a.py": "x = 1\n"}})
    await take_turn(session, "what is this?")
    assert "no executor" in session.state[ANSWER].problem


async def test_a_proposing_leaf_without_an_executor_says_so(session: Session) -> None:
    session.move(COMMAND)
    turn = await take_turn(session, "echo hi")
    assert turn.awaiting
    assert turn.proposal is not None and not turn.proposal.ok
    assert "no executor" in turn.proposal.problem


def test_a_gate_reports_pending_then_settled() -> None:
    from xg_project.turn import PENDING, Gate

    gate = Gate(proposal=CommandProposal(command="echo hi"))
    assert gate.status == PENDING
    assert not gate.settled
    assert "pending: echo hi" in gate.preview()

    gate.status = ACCEPTED
    gate.outcome = "exit 0"
    assert gate.settled
    assert "accepted: echo hi" in gate.preview()

    rejected = Gate(proposal=CommandProposal(command="echo hi"), status=REJECTED)
    assert "rejected: echo hi" in rejected.preview()


# -- nodes without behaviour -----------------------------------------------


async def test_a_node_without_a_handler_records_a_problem() -> None:
    """Descriptors with no behaviour are shown, not silently ignored."""
    from xg_project.graph import NodeDeclaration, Registry

    registry = Registry()
    registry.add(NodeDeclaration(name="none", level=0, summary="x"))
    session = Session(registry)
    turn = await take_turn(session, "anything")
    assert "nothing to do" in turn.produced["problem"]
