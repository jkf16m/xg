"""Integration tests for xg_project.config."""

import pytest

from xg_project.config import load_project_config


def _write_config(root, body):
    config_dir = root / ".xg"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(body, encoding="utf-8")


@pytest.mark.integration
def test_missing_config_returns_defaults(tmp_path):
    """Without .xg/config.json the defaults apply."""
    config = load_project_config(tmp_path)
    assert config.use_gitignore is True
    assert config.files is None


@pytest.mark.integration
def test_config_without_files_has_no_allowlist(tmp_path):
    """A config without a files key leaves the module unrestricted."""
    _write_config(tmp_path, '{"use_gitignore": false}')

    config = load_project_config(tmp_path)

    assert config.use_gitignore is False
    assert config.files is None


@pytest.mark.integration
def test_config_reads_files_allowlist(tmp_path):
    """A files array is read as a tuple of relative paths."""
    _write_config(tmp_path, '{"files": ["a.py", "b.py"]}')

    config = load_project_config(tmp_path)

    assert config.files == ("a.py", "b.py")


@pytest.mark.integration
def test_config_reads_empty_allowlist(tmp_path):
    """An empty files array stays empty rather than becoming None."""
    _write_config(tmp_path, '{"files": []}')

    assert load_project_config(tmp_path).files == ()


@pytest.mark.integration
def test_config_rejects_non_list_files(tmp_path):
    """A files value that is not an array of strings is rejected."""
    _write_config(tmp_path, '{"files": "a.py"}')

    with pytest.raises(ValueError, match="files must be an array of strings"):
        load_project_config(tmp_path)


@pytest.mark.integration
def test_config_rejects_non_string_files(tmp_path):
    """A files array containing non-strings is rejected."""
    _write_config(tmp_path, '{"files": ["a.py", 3]}')

    with pytest.raises(ValueError, match="files must be an array of strings"):
        load_project_config(tmp_path)


@pytest.mark.integration
def test_config_rejects_non_boolean_use_gitignore(tmp_path):
    """A non-boolean use_gitignore is rejected."""
    _write_config(tmp_path, '{"use_gitignore": "yes"}')

    with pytest.raises(ValueError, match="use_gitignore must be a boolean"):
        load_project_config(tmp_path)
