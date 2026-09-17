# xg

An AI coding agent harness for a 1:1 human-guided loop.

`xg` is a terminal REPL that drives a model against your current project. It
loads the project as context, streams a reply, and proposes tool calls for you
to approve before anything runs. Conversations persist to a local SQLite
database, so you can pick up where you left off.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (used for install and build)
- An OpenRouter API key readable with `pass show pi/openrouter`
- `git` (optional, for `.gitignore`-aware file discovery)

## Install

Install user-wide via [pipx](https://pipx.pypa.io/):

```sh
python install.py
```

Install into a local `.venv` for development:

```sh
python install.py --local
```

The pipx install puts `xg` on your `PATH`. The local install prints the venv's
`bin/` path; activate the venv and run `xg` from there.

To uninstall the pipx install:

```sh
pipx uninstall xg-project
```

## Usage

```sh
xg          # start a new conversation
xg -c       # continue the last conversation
xg --info   # show context construction and model information
```

Inside the REPL:

- Type a message and press Enter to add it to the conversation.
- Press Enter on an empty prompt to send the turn to the model.
- When the model proposes tool calls, approve them or replace them with a
  new instruction.
- Press ESC to exit.

### Tool approval

Each proposed tool call is shown with a preview (a diff for `write`/`edit`
calls). At the `tool>` prompt:

| Input    | Action                                        |
| -------- | --------------------------------------------- |
| `Enter`  | allow and run every proposed call             |
| `/v`     | view the raw arguments of the proposed calls  |
| `/e`     | extract and run the command calls in a shell  |
| any text | cancel the calls and send the text as a reply |

## Tools

The model can call four tools, all of which require approval:

- `read` — read a file; the result is appended at the bottom of the context.
- `write` — create a new file. Existing files must be changed with `edit`.
- `edit` — replace one exact, unique occurrence in an existing file.
- `cmd` — run a command in the current project environment.

## How context is built

1. A system message is prepended to every request.
2. Session messages are loaded in database insertion order.
3. New user messages are appended to the session.
4. Each request is sent as: system message + conversation messages.
5. Tool results are appended after approved tool execution.

The initial file context is discovered recursively, excluding `.git` and
`__pycache__`. Module file allowlists, Git-ignored files, and `.xgignore`
patterns are then applied, and the remainder is ordered by mtime.

## Configuration

Settings live in `.xg/` directories and compose down the directory tree.
Every relative path in these files resolves against the directory containing
`.xg/`, never against `.xg/` itself.

### `.xg/config.json` — settings

Composed from `$HOME/.xg/config.json` (the global base) plus ancestor
`.xg/config.json` files, from the farthest ancestor down to the current
directory. Each layer overrides the previous one per key.

| Key               | Type    | Meaning                                              |
| ----------------- | ------- | ---------------------------------------------------- |
| `use_gitignore`   | boolean | respect Git ignore rules during file discovery       |
| `model`           | string  | OpenRouter model to use                              |
| `patch_formatter` | string  | shell command to render patch previews (e.g. `delta`)|
| `stopWalking`     | boolean | stop composing settings from further ancestors       |

### `.xg/module.json` — context modules

A module declaration marks a directory as a context boundary. It never
inherits from an enclosing module.

```json
{
  "files": ["keep.py"]
}
```

`files` is an allowlist of paths relative to the module root. An empty array
keeps nothing in the subtree; omitting `files` does not restrict the subtree.

### `.xg/SYSTEM.md` — system prompt

Replaces the built-in system prompt for the project when present.

## Sessions

All sessions live in a single SQLite database at `~/.xg/sessions.db`. A
session is identified by a logical path key — a composite of directory and
filename — resolved to a canonical absolute path before storage. No session
file is created on disk.

## Development

Build and lint:

```sh
uv run python build.py
```

Run the integration tests:

```sh
uv run pytest -m integration -v src/
```

## Layout

```
src/xg_project/
  __init__.py          CLI entry point (argparse, PTY launcher)
  interface.py         REPL, markdown streaming, tool approval
  terminal.py          PTY input primitives
  config/              settings composition and module declarations
  llm/                 LLM integration, file discovery, tools
  session/             SQLite persistence and schema migrations
```
