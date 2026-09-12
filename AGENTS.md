# xg-project Agent Instructions

## Build & Lint

Run `build.py` to execute all checks:

```sh
uv run python build.py
```

This runs:
- **ruff** — linting (E, F, I, B, S, UP rules)

## Testing

Run integration tests:

```sh
uv run pytest -m integration -v src/
```

## Module Structure

### `xg_project.llm`

Public API — import only from this:

```python
from xg_project.llm import (
    create,
    load,
    append,
    remove,
    messages,
    stream,
    TOOLS,
)
```

Private implementation lives in `_`-prefixed modules (`_api.py`, `_tools.py`, `_context.py`).

### `xg_project.session`

Public API — import only from this:

```python
from xg_project.session import create, load, append, remove, messages, stream
```

Functions take a `Path` as first argument. No classes — pure functions only.
