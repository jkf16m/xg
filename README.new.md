# xg

An AI coding agent harness for a 1:1 human-guided loop.

`xg` takes a request in plain language, decides what kind of request it is,
finds the files the request touches, shows them to you, and — once you confirm —
creates the new file. Every decision that steers the run is made by a
*decision* model (TypeSafe Jev) or by deterministic code. A generative model is
reached only at the last moment, to fill in the file's path and contents.

## How it works

The run is a [LangGraph](https://langchain-ai.github.io/langgraph/) state graph.
No tools are bound and the graph calls none.

```
START -> classify -> { local_research, reroute }
local_research -> confirm -> { classify, write, END }
write -> check -> { reroute, context }
context -> generate -> END
reroute -> { classify, END }
```

| Node             | What it does                                                                |
| ---------------- | --------------------------------------------------------------------------- |
| `classify`       | One Jev `system_one` call classifies the request (kind, operation, complexity, risk). |
| `local_research` | Lists the repository and asks one Jev `Noul` question per candidate file: *is this file needed?* Selection is decided from file contents, not names. |
| `confirm`        | Interrupt. Shows the selected files and their relevance; you continue, reprompt, or quit. |
| `write`          | Asks the generative model for the path of the new file.                      |
| `check`          | Plain filesystem code validates the path and checks that it is free.         |
| `context`        | Builds a deterministic context window from the selected files.               |
| `generate`       | Asks the generative model for the file's contents, then writes the file.     |
| `reroute`        | Interrupt. Shown when the graph will not guess; you supply a clearer request. |

Two nodes ask the user, both as LangGraph interrupts:

- **`confirm`** — after research. Continue to the write path, or reprompt, which
  re-enters at `classify` because a different request can route differently.
- **`reroute`** — when `classify` could not place the request, the model produced
  no usable path, or the target path already exists. Any text you type is the
  new request.

The write path deliberately keeps the model out of the collision decision: the
model proposes (or returns no) path, code checks it, and only a free path is
handed back to the model. A model that refuses or explains instead of answering
cannot have its prose mistaken for a filename, and a model cannot write outside
the repository (`..` and absolute paths are rejected).

Interrupts need a checkpointer, so the graph compiles with an `InMemorySaver`
whose serializer is given an explicit allowlist of the state dataclasses.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for install and build
- A TypeSafe API key — `TYPESAFE_API_KEY`, or readable with `pass show jev`
- An OpenRouter API key readable with `pass show pi/openrouter`
- `git` (optional; used for `.gitignore`-aware file listing)

## Install

Run from the checkout with `uv`:

```sh
uv run xg "add a module that parses xgignore files"
```

Install the `xg` script user-wide with [pipx](https://pipx.pypa.io/):

```sh
pipx install .
```

Uninstall with `pipx uninstall xg-project`.

## Usage

```sh
xg "add a module that parses xgignore files"   # run one request
xg                                             # type requests, one per line
```

Omit the prompt to read prompts from stdin; the client and graph are reused
across requests.

| Option            | Meaning                                                             |
| ----------------- | ------------------------------------------------------------------- |
| `--model`         | Jev model name (default: `.xg` `jev_model`, else `jev-latest`)       |
| `--llm-model`     | generative model name (default: `.xg` `model`, else `@preset/mimo`)  |
| `--json`          | print the run state as JSON                                         |

At the confirmation prompt:

| Input                  | Action                                             |
| ---------------------- | -------------------------------------------------- |
| `c`, `continue`, `y`, `yes` | continue into the write path                  |
| `r`, `reprompt`        | enter a new prompt; the run re-enters at `classify` |
| text other than the above | treated as a new prompt                         |
| `q`, `quit`, `n`, `no`, Ctrl-C, Ctrl-D | stop the run                     |

At the clarification prompt any text is the new request; an empty reply, `q`,
Ctrl-C, or Ctrl-D stops the run.

With `--json` a stopped run is printed with `awaiting` set (`confirmation` or
`clarification`) and the command exits, so it stays scriptable. Alongside the
classification fields, the payload carries `route`, `reason`, the research
counts (`candidate_count`, `listed_count`, `content_bytes`, `relevance`), and
whichever of `proposed_path`, `target_path`, `context_path`, `context_files`,
`context_bytes`, `written_path`, `written_bytes` the run reached.

## How decisions are made

`xg_project.jev` is the decision layer. Jev returns typed
`Noul`/`Choice`/`Score` answers, never text and never tool calls, and it answers
everything about one state in a single parallel pass.

Classification asks, in one `system_one` call:

- `request_kind` (`Choice`) — `question`, `edit_feature`, `new_feature`,
  `bug_fix`, `refactor`, `test`, `docs`, `config`, `confused`, `other`.
- `operation` (`Choice`) — `write`, `edit`, `answer`: what the work will produce.
- `complexity` (`Score`) — a four-level rubric.
- `changes_code` (`Noul`) — must the repository change at all?
- `is_destructive` (`Noul`) — could the request lose data or rewrite history?

Confidence is banded: at or above `0.75` is `high` and routes automatically;
below `0.5` the graph will not act and asks for a clearer request. `confused`
and `other` reroute regardless of confidence — guessing would be worse than
asking. `scope` is deliberately *not* part of classification: it is an outcome
of research, measured by how many files match.

Research sends the repository listing and the full text of each candidate to
Jev, then asks one `Noul` per file ("is this file needed?"), all in one call.
Files clearing a `0.5` probability are selected, most relevant first. Selection
is content-based; a lexical prefilter only runs when the repository exceeds the
per-call candidate cap.

## How context is built

A context window is an ordered, fully-read view of the files research chose.
Nothing in the context step calls a model, so the same files with the same
mtimes always produce the same text.

- Files are read whole. There is no prefilter and no truncation — research
  already chose them, and a silent trim would hide the evidence the choice was
  made on.
- Entries are ordered by mtime, newest first, with the path as a tie-break.
- A file that cannot be read is reported in place rather than dropped.
- The rendered window (the request, then every file) is written to a temporary
  file outside the repository, so it can be inspected after a run without
  polluting the project or its git listing.

## Configuration

Settings live in `.xg/` directories and compose down the directory tree. Every
relative path in these files resolves against the directory containing `.xg/`,
never against `.xg/` itself.

### `.xg/config.json` — settings

Composed from `$HOME/.xg/config.json` (the global base) plus ancestor
`.xg/config.json` files, from the farthest ancestor down to the current
directory. Each layer overrides the previous one per key; a key a nearer layer
omits stays inherited. Walking upward stops after the first file that sets
`"stopWalking": true` — that layer is included, nothing above it is.

| Key               | Type    | Meaning                                                       |
| ----------------- | ------- | ------------------------------------------------------------- |
| `use_gitignore`   | boolean | respect Git ignore rules during file listing                   |
| `model`           | string  | OpenRouter model for the generative steps                      |
| `jev_model`       | string  | Jev model for the decision steps                               |
| `patch_formatter` | string  | shell command to render patch previews (e.g. `delta`)          |
| `stopWalking`     | boolean | stop composing settings from further ancestors                 |
| `session_path`    | string  | session storage location                                       |

### `.xg/module.json` — context modules

A module declaration marks a directory as a context boundary. It never inherits
from an enclosing module.

```json
{
  "files": ["keep.py"]
}
```

`files` is an allowlist of paths relative to the module root. An empty array
keeps nothing in the subtree; omitting `files` does not restrict the subtree.

## Development

Lint:

```sh
uv run python build.py
```

Run the integration tests (`-m integration` marks tests that need an API key):

```sh
uv run pytest -m integration -v src/
```

## Layout

```
src/xg_project/
  __init__.py        CLI entry point
  agent/             the LangGraph agent: graph, state, CLI
  jev/               TypeSafe Jev decision layer: classify, taxonomy, client
  research/          local_research — find the relevant files
  context/           deterministic context windows
  config/            .xg settings composition and module declarations
  llm/               OpenRouter generative layer: propose a path, write a file
```

## Status

The agent has been rebuilt on LangGraph with the TypeSafe Jev SDK. The previous
implementation (LLM turns, tools, SQLite persistence, REPL) has been removed.

Only the **write** path is implemented end to end: `xg` classifies a request,
researches the repository, and creates one new file. Requests classified as
`edit` or `answer` are understood but have no path yet — the run ends at
`confirm`. Requests that would be shell commands fall to `other` and are
rerouted to you rather than routed to a node that does not exist.

Some declared dependencies (`pexpect`, `rich`, `sqlite-utils`, `pathspec`) are
left over from the removed implementation.