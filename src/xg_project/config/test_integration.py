"""Integration tests for xg_project.config."""

import pytest

from xg_project.config import load_module, resolve


def _write_settings(root, body):
    """Write root/.xg/config.json (composed settings)."""
    config_dir = root / ".xg"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(body, encoding="utf-8")


def _write_module(root, body):
    """Write root/.xg/module.json (a module declaration)."""
    module_dir = root / ".xg"
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "module.json").write_text(body, encoding="utf-8")


def _isolate_global(monkeypatch, tmp_path):
    """Point the global settings base at a path that does not exist."""
    import xg_project.config as config_module

    monkeypatch.setattr(
        config_module, "GLOBAL_CONFIG_PATH", tmp_path / "missing" / "config.json"
    )


# --- resolve() settings composition tests ---


@pytest.mark.integration
def test_resolve_without_any_config_returns_defaults(tmp_path, monkeypatch):
    """With no global base and no .xg/config.json, defaults apply."""
    _isolate_global(monkeypatch, tmp_path)

    config = resolve(tmp_path)

    assert config.use_gitignore is True


@pytest.mark.integration
def test_resolve_reads_use_gitignore(tmp_path, monkeypatch):
    """use_gitignore is read from .xg/config.json."""
    _isolate_global(monkeypatch, tmp_path)
    _write_settings(tmp_path, '{"use_gitignore": false}')

    assert resolve(tmp_path).use_gitignore is False


@pytest.mark.integration
def test_resolve_rejects_non_boolean_use_gitignore(tmp_path, monkeypatch):
    """A non-boolean use_gitignore is rejected."""
    _isolate_global(monkeypatch, tmp_path)
    _write_settings(tmp_path, '{"use_gitignore": "yes"}')

    with pytest.raises(ValueError, match="use_gitignore must be a boolean"):
        resolve(tmp_path)


@pytest.mark.integration
def test_resolve_rejects_files_in_settings(tmp_path, monkeypatch):
    """files is a module key and is rejected in .xg/config.json."""
    _isolate_global(monkeypatch, tmp_path)
    _write_settings(tmp_path, '{"files": ["a.py"]}')

    with pytest.raises(ValueError, match="files is a module key"):
        resolve(tmp_path)


@pytest.mark.integration
def test_resolve_includes_global_base(tmp_path, monkeypatch):
    """The global config is the base layer."""
    import xg_project.config as config_module

    global_path = tmp_path / "global" / "config.json"
    global_path.parent.mkdir(parents=True)
    global_path.write_text('{"use_gitignore": false}', encoding="utf-8")
    monkeypatch.setattr(config_module, "GLOBAL_CONFIG_PATH", global_path)

    project = tmp_path / "project"
    project.mkdir()

    assert resolve(project).use_gitignore is False


@pytest.mark.integration
def test_resolve_inherits_farther_when_nearer_omits(tmp_path, monkeypatch):
    """A key a nearer layer omits stays inherited from a farther layer."""
    _isolate_global(monkeypatch, tmp_path)

    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    _write_settings(outer, '{"use_gitignore": false}')
    _write_settings(inner, "{}")

    assert resolve(inner).use_gitignore is False


@pytest.mark.integration
def test_resolve_nearer_layer_overrides_farther(tmp_path, monkeypatch):
    """A nearer .xg/config.json overrides a key set in a farther one."""
    _isolate_global(monkeypatch, tmp_path)

    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    _write_settings(outer, '{"use_gitignore": true}')
    _write_settings(inner, '{"use_gitignore": false}')

    assert resolve(inner).use_gitignore is False


@pytest.mark.integration
def test_resolve_stopwalking_halts_walk(tmp_path, monkeypatch):
    """stopWalking: true stops the walk; ancestors above it are not collected."""
    _isolate_global(monkeypatch, tmp_path)

    top = tmp_path / "top"
    middle = top / "middle"
    leaf = middle / "leaf"
    leaf.mkdir(parents=True)
    _write_settings(top, '{"use_gitignore": false}')
    _write_settings(middle, '{"stopWalking": true}')
    _write_settings(leaf, "{}")

    # top is beyond the stop, so its false is never collected.
    assert resolve(leaf).use_gitignore is True


@pytest.mark.integration
def test_resolve_stopwalking_layer_visible_when_nearer_omits_key(tmp_path, monkeypatch):
    """The stopping layer contributes its keys to the merged settings."""
    _isolate_global(monkeypatch, tmp_path)

    top = tmp_path / "top"
    middle = top / "middle"
    leaf = middle / "leaf"
    leaf.mkdir(parents=True)
    _write_settings(top, '{"use_gitignore": true}')
    _write_settings(middle, '{"use_gitignore": false, "stopWalking": true}')
    _write_settings(leaf, "{}")

    assert resolve(leaf).use_gitignore is False


@pytest.mark.integration
def test_resolve_file_uses_parent_directory(tmp_path, monkeypatch):
    """Passing a file resolves against its parent directory."""
    _isolate_global(monkeypatch, tmp_path)

    project = tmp_path / "project"
    project.mkdir()
    _write_settings(project, '{"use_gitignore": false}')
    target = project / "a.py"
    target.write_text("")

    assert resolve(target).use_gitignore is False


@pytest.mark.integration
def test_resolve_rejects_non_boolean_stopwalking(tmp_path, monkeypatch):
    """A non-boolean stopWalking is rejected."""
    _isolate_global(monkeypatch, tmp_path)
    _write_settings(tmp_path, '{"stopWalking": "yes"}')

    with pytest.raises(ValueError, match="stopWalking must be a boolean"):
        resolve(tmp_path)


# --- load_module() module declaration tests ---


@pytest.mark.integration
def test_load_module_absent_returns_none(tmp_path):
    """A directory without .xg/module.json is not a module."""
    assert load_module(tmp_path) is None


@pytest.mark.integration
def test_load_module_reads_files_allowlist(tmp_path):
    """A files array is read as a tuple of paths relative to the module root."""
    _write_module(tmp_path, '{"files": ["a.py", "b.py"]}')

    assert load_module(tmp_path).files == ("a.py", "b.py")


@pytest.mark.integration
def test_load_module_reads_empty_allowlist(tmp_path):
    """An empty files array stays empty rather than becoming None."""
    _write_module(tmp_path, '{"files": []}')

    assert load_module(tmp_path).files == ()


@pytest.mark.integration
def test_load_module_rejects_non_list_files(tmp_path):
    """A files value that is not an array of strings is rejected."""
    _write_module(tmp_path, '{"files": "a.py"}')

    with pytest.raises(ValueError, match="files must be an array of strings"):
        load_module(tmp_path)


@pytest.mark.integration
def test_load_module_rejects_non_string_files(tmp_path):
    """A files array containing non-strings is rejected."""
    _write_module(tmp_path, '{"files": ["a.py", 3]}')

    with pytest.raises(ValueError, match="files must be an array of strings"):
        load_module(tmp_path)


@pytest.mark.integration
def test_load_module_does_not_inherit_outer_module(tmp_path):
    """A module declaration is read alone; an outer module's files are ignored."""
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    _write_module(outer, '{"files": ["outer.py"]}')
    _write_module(inner, "{}")

    assert load_module(outer).files == ("outer.py",)
    assert load_module(inner).files is None


# --- model setting tests ---


@pytest.mark.integration
def test_resolve_reads_model(tmp_path, monkeypatch):
    """model is read from .xg/config.json."""
    _isolate_global(monkeypatch, tmp_path)
    _write_settings(tmp_path, '{"model": "@preset/deepseek"}')

    assert resolve(tmp_path).model == "@preset/deepseek"


@pytest.mark.integration
def test_resolve_model_defaults_to_none(tmp_path, monkeypatch):
    """Without a model key the setting is None, not a built-in default."""
    _isolate_global(monkeypatch, tmp_path)

    assert resolve(tmp_path).model is None


@pytest.mark.integration
def test_resolve_rejects_non_string_model(tmp_path, monkeypatch):
    """A non-string model is rejected."""
    _isolate_global(monkeypatch, tmp_path)
    _write_settings(tmp_path, '{"model": 3}')

    with pytest.raises(ValueError, match="model must be a string"):
        resolve(tmp_path)


@pytest.mark.integration
def test_resolve_reads_patch_formatter(tmp_path, monkeypatch):
    """patch_formatter is read from .xg/config.json."""
    _isolate_global(monkeypatch, tmp_path)
    _write_settings(tmp_path, '{"patch_formatter": "delta --paging=never"}')

    assert resolve(tmp_path).patch_formatter == "delta --paging=never"


@pytest.mark.integration
def test_resolve_rejects_non_string_patch_formatter(tmp_path, monkeypatch):
    """A non-string patch_formatter is rejected."""
    _isolate_global(monkeypatch, tmp_path)
    _write_settings(tmp_path, '{"patch_formatter": 3}')

    with pytest.raises(ValueError, match="patch_formatter must be a string"):
        resolve(tmp_path)
