# xg-project

## Build & Lint

```sh
uv run python build.py
```

## Tests

```sh
uv run pytest -m integration -v src/
```

## Key Files

- `src/xg_project/llm/__init__.py` — LLM API surface
- `src/xg_project/session/__init__.py` — persistence layer
- `src/xg_project/interface.py` — REPL entry point

## Trust Rules

- Trust the context window. Read files before editing.
- Trust your training on language fundamentals and programming principles.
- Do not trust your training on library APIs. If you need to use a library and its API is not in the context window, stop and ask the user to provide documentation or run research.
