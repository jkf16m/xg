"""Context modules: what a folder declares it exposes, and the bound it becomes.

Two halves, tested separately and then together. Discovery and scope resolution
are deterministic — same tree, same file set — and the node is the inference on
top: which module the request belongs to, and what the read is allowed to open
once that is decided.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_turn import FakeJev
from xg_project.graph import FILTER, SELECT_MODULE, default_graph
from xg_project.jev import MODULE_CRITERIA
from xg_project.module import discover, scope
from xg_project.read import read_tree
from xg_project.session import Session
from xg_project.turn import take_turn


def write(root: Path, where: str, text: str = "x\n") -> Path:
    """Create a file below ``root``, making the folders it needs."""
    path = root / where
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def declare(root: Path, where: str, **manifest) -> Path:
    """Write a ``.xg/module.json`` in ``root/where`` and return that folder."""
    folder = root if where == "." else root / where
    (folder / ".xg").mkdir(parents=True, exist_ok=True)
    (folder / ".xg" / "module.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def manifest(root: Path, where: str, text: str) -> None:
    """Write a raw, possibly malformed manifest."""
    folder = root if where == "." else root / where
    (folder / ".xg").mkdir(parents=True, exist_ok=True)
    (folder / ".xg" / "module.json").write_text(text, encoding="utf-8")


@pytest.fixture
def session() -> Session:
    return Session(default_graph())


# -- discovery -------------------------------------------------------------


def test_a_module_is_a_folder_whose_manifest_says_what_it_exposes(tmp_path: Path) -> None:
    write(tmp_path, "core/app.py")
    declare(tmp_path, "core", description="the core", expose=["app.py"])

    found = discover(tmp_path)

    assert found.problems == ()
    (only,) = found.found
    assert only.id(tmp_path) == "core"
    assert only.name == "core"  # the folder's name, when the manifest does not give one
    assert only.description == "the core"
    assert only.expose == ("app.py",)
    assert only.imports == ()


def test_the_root_itself_can_be_a_module(tmp_path: Path) -> None:
    write(tmp_path, "app.py")
    declare(tmp_path, ".", description="everything", expose=["."])

    (only,) = discover(tmp_path).found

    assert only.id(tmp_path) == "."
    assert scope(discover(tmp_path), only).files == ("app.py",)


def test_a_manifest_is_never_read_as_project_content(tmp_path: Path) -> None:
    """A manifest is an instruction about the read, like an ignore file."""
    write(tmp_path, "app.py")
    declare(tmp_path, ".", description="everything", expose=["."])

    files = read_tree(tmp_path).files

    assert "app.py" in files
    assert not [path for path in files if ".xg" in path.split("/")]


def test_a_broken_manifest_costs_its_own_module_and_not_the_run(tmp_path: Path) -> None:
    write(tmp_path, "a/app.py")
    declare(tmp_path, "a", description="a", expose=["."])
    manifest(tmp_path, "b", "{not json")

    found = discover(tmp_path)

    assert [module.id(tmp_path) for module in found.found] == ["a"]
    assert len(found.problems) == 1
    assert "not valid JSON" in found.problems[0]


def test_a_manifest_without_a_description_is_refused(tmp_path: Path) -> None:
    write(tmp_path, "a/app.py")
    declare(tmp_path, "a", expose=["."])

    found = discover(tmp_path)

    assert found.found == ()
    assert "description" in found.problems[0]


def test_a_manifest_without_an_exposure_is_refused(tmp_path: Path) -> None:
    write(tmp_path, "a/app.py")
    declare(tmp_path, "a", description="a")

    found = discover(tmp_path)

    assert found.found == ()
    assert "'expose' is required" in found.problems[0]


def test_a_misspelled_key_is_refused_rather_than_read_as_an_empty_context(
    tmp_path: Path,
) -> None:
    """`exposes` would otherwise mean "exposes nothing", which is silent."""
    write(tmp_path, "a/app.py")
    declare(tmp_path, "a", description="a", exposes=["."])

    found = discover(tmp_path)

    assert found.found == ()
    assert "unknown key" in found.problems[0]


def test_a_module_inside_an_ignored_folder_is_not_found(tmp_path: Path) -> None:
    """A module nothing may read could never contribute a file."""
    write(tmp_path, "tests/test_a.py")
    (tmp_path / ".xgignore").write_text("tests/\n", encoding="utf-8")
    declare(tmp_path, "tests", description="the tests", expose=["."])

    assert discover(tmp_path).found == ()


# -- what a module exposes -------------------------------------------------


def test_an_exposed_folder_is_every_file_below_it(tmp_path: Path) -> None:
    write(tmp_path, "core/app.py")
    write(tmp_path, "core/deep/thing.py")
    write(tmp_path, "other/no.py")
    declare(tmp_path, "core", description="the core", expose=["."])

    found = discover(tmp_path)

    assert scope(found, found.found[0]).files == ("core/app.py", "core/deep/thing.py")


def test_a_glob_exposes_only_what_it_matches(tmp_path: Path) -> None:
    write(tmp_path, "core/app.py")
    write(tmp_path, "core/notes.md")
    declare(tmp_path, "core", description="the core", expose=["*.py"])

    found = discover(tmp_path)

    assert scope(found, found.found[0]).files == ("core/app.py",)


def test_an_import_adds_the_imported_modules_context(tmp_path: Path) -> None:
    write(tmp_path, "core/app.py")
    write(tmp_path, "lib/util.py")
    declare(tmp_path, "lib", description="the helpers", expose=["."])
    declare(tmp_path, "core", description="the core", expose=["."], **{"import": ["../lib"]})

    found = discover(tmp_path)
    selected = next(module for module in found.found if module.id(tmp_path) == "core")
    resolved = scope(found, selected)

    assert set(resolved.files) == {"core/app.py", "lib/util.py"}
    assert resolved.modules == ("core", "lib")


def test_an_import_cycle_is_tolerated_rather_than_chased(tmp_path: Path) -> None:
    write(tmp_path, "a/a.py")
    write(tmp_path, "b/b.py")
    declare(tmp_path, "a", description="a", expose=["."], **{"import": ["../b"]})
    declare(tmp_path, "b", description="b", expose=["."], **{"import": ["../a"]})

    found = discover(tmp_path)
    selected = next(module for module in found.found if module.id(tmp_path) == "a")

    assert scope(found, selected).modules == ("a", "b")


def test_an_exposed_file_that_is_ignored_is_not_in_scope(tmp_path: Path) -> None:
    """Exposure says what the module offers; the ignores still say what may be read."""
    write(tmp_path, "core/app.py")
    write(tmp_path, "core/secret.env")
    (tmp_path / ".xgignore").write_text("core/secret.env\n", encoding="utf-8")
    declare(tmp_path, "core", description="the core", expose=["."])

    found = discover(tmp_path)

    assert scope(found, found.found[0]).files == ("core/app.py",)


def test_an_exposure_that_matches_nothing_is_reported(tmp_path: Path) -> None:
    write(tmp_path, "core/app.py")
    declare(tmp_path, "core", description="the core", expose=["app.py", "gone.py"])

    found = discover(tmp_path)

    assert scope(found, found.found[0]).missing == (
        "core: exposes 'gone.py', which matched nothing",
    )


def test_an_import_that_names_no_module_is_reported(tmp_path: Path) -> None:
    write(tmp_path, "core/app.py")
    declare(tmp_path, "core", description="the core", expose=["."], **{"import": ["../nope"]})

    found = discover(tmp_path)

    assert scope(found, found.found[0]).missing == (
        "core: imports '../nope', which is not a module",
    )


# -- the read bound --------------------------------------------------------


def test_include_bounds_the_read(tmp_path: Path) -> None:
    write(tmp_path, "a.py")
    write(tmp_path, "b.py")

    assert set(read_tree(tmp_path, include=["a.py"]).files) == {"a.py"}


def test_an_empty_include_reads_nothing(tmp_path: Path) -> None:
    """Empty is a bound that admits nothing, not the absence of a bound."""
    write(tmp_path, "a.py")

    assert read_tree(tmp_path, include=[]).files == {}
    assert set(read_tree(tmp_path).files) == {"a.py"}


# -- the node --------------------------------------------------------------


async def test_one_module_is_the_context_without_being_asked_about(
    session: Session, tmp_path: Path
) -> None:
    """One declared context is the context; asking would spend a round trip to be told."""
    write(tmp_path, "core/app.py")
    declare(tmp_path, "core", description="the core", expose=["."])
    jev = FakeJev(choice=SELECT_MODULE)
    session.move(SELECT_MODULE)

    await take_turn(session, "go", jev=jev, root=tmp_path)

    state = session.state[SELECT_MODULE]
    assert state["module"] == "core"
    assert state["name"] == "core"
    assert state["includes"] == ["core/app.py"]
    assert state["missing"] == []
    assert jev.select_calls == []
    # One child is the edge, so the run is already at FILTER.
    assert session.position == FILTER


async def test_several_modules_are_put_to_jev_and_the_most_important_is_taken(
    session: Session, tmp_path: Path
) -> None:
    write(tmp_path, "core/app.py")
    write(tmp_path, "docs/readme.md")
    declare(tmp_path, "core", description="the application code", expose=["."])
    declare(tmp_path, "docs", description="the prose", expose=["."])
    jev = FakeJev(scores={"core": 0.9, "docs": 0.2}, kept={"core"})
    session.move(SELECT_MODULE)

    await take_turn(session, "fix the app", jev=jev, root=tmp_path)

    state = session.state[SELECT_MODULE]
    assert state["module"] == "core"
    assert state["scores"] == {"core": 0.9, "docs": 0.2}
    assert set(state["modules"]) == {"core", "docs"}
    assert state["includes"] == ["core/app.py"]
    # The modules' declarations are the question's state: no file content is read
    # to decide which module to read.
    call = jev.select_calls[0]
    assert set(call["files"]) == {"core", "docs"}
    assert "application code" in call["files"]["core"]
    assert call["criteria"] is MODULE_CRITERIA


async def test_no_module_leaves_the_whole_project_in_scope(
    session: Session, tmp_path: Path
) -> None:
    write(tmp_path, "app.py")
    session.move(SELECT_MODULE)

    await take_turn(session, "go", jev=FakeJev(), root=tmp_path)

    state = session.state[SELECT_MODULE]
    assert state["module"] == ""
    assert state["includes"] == []


async def test_a_module_nothing_scores_above_the_threshold_leaves_the_whole_project_in_scope(
    session: Session, tmp_path: Path
) -> None:
    """A wrong module hides the right files completely, so none is safer than one."""
    write(tmp_path, "core/app.py")
    write(tmp_path, "docs/readme.md")
    declare(tmp_path, "core", description="the application code", expose=["."])
    declare(tmp_path, "docs", description="the prose", expose=["."])
    jev = FakeJev(scores={"core": 0.2, "docs": 0.1}, kept=set())
    session.move(SELECT_MODULE)

    await take_turn(session, "what is the weather", jev=jev, root=tmp_path)

    state = session.state[SELECT_MODULE]
    assert state["module"] == ""
    assert state["problem"]
    assert state["includes"] == []


async def test_a_selected_module_bounds_what_filter_may_read(
    session: Session, tmp_path: Path
) -> None:
    write(tmp_path, "core/app.py")
    write(tmp_path, "other/thing.py")
    declare(tmp_path, "core", description="the core", expose=["."])
    jev = FakeJev(choice=SELECT_MODULE)
    session.move(SELECT_MODULE)

    await take_turn(session, "go", jev=jev, root=tmp_path)  # selects, lands on FILTER
    await take_turn(session, "go", jev=jev, root=tmp_path)  # FILTER runs

    state = session.state[FILTER]
    assert state["read"] == 1
    assert set(state["files"]) == {"core/app.py"}
    assert "other/thing.py" not in state["scores"]


async def test_a_module_that_exposes_nothing_admits_nothing(
    session: Session, tmp_path: Path
) -> None:
    """The failure mode this guards: an empty scope widening the read to everything."""
    write(tmp_path, "empty/keep.txt")
    declare(tmp_path, "empty", description="nothing yet", expose=["gone/**"])
    jev = FakeJev(choice=SELECT_MODULE)
    session.move(SELECT_MODULE)

    await take_turn(session, "go", jev=jev, root=tmp_path)

    state = session.state[SELECT_MODULE]
    assert state["module"] == "empty"
    assert state["includes"] == []
    assert state["missing"]

    await take_turn(session, "go", jev=jev, root=tmp_path)
    assert session.state[FILTER]["read"] == 0


async def test_a_missing_root_is_a_problem_rather_than_a_guess(session: Session) -> None:
    session.move(SELECT_MODULE)

    await take_turn(session, "go", jev=FakeJev(), root=None)

    assert "no project root" in session.state[SELECT_MODULE]["problem"]
