# xg

A terminal agent for coding tasks that **decides before it writes**.

`xg` is a TUI, not a chat. You state a goal at a prompt and a workflow walks a
small graph of nodes — narrow the context to one module, filter the files that
matter, rank them, then answer or propose an edit. Every decision that steers the
run is made by a deterministic rule or by a *decision* model. A generative model
is reached only at the very end, and only to fill in something concrete: an edit,
a new file, a command, or a sentence.

Nothing that changes anything happens on its own. An edit, a new file or a command
is shown as a proposal and waits for you.

```
 ⭘                                              xg
 _XG_ORIGIN ◀ you are here
 ├─ _XG_ADD
 ├─ _XG_COMMAND
 └─ _XG_SELECT_MODULE
    └─ _XG_FILTER
       └─ _XG_SORT
          ├─ _XG_ANSWER
          └─ _XG_EDIT
  you are at _XG_ORIGIN · Where a run starts: records the user's request as the goal.
  trail: _XG_ORIGIN
──────────────────────────────────────────────────────────────────────────────────
  _XG_ORIGIN ← make the retry loop stop on a 4xx
  _XG_FILTER  4 of 20 files: llm.py, jev.py, app.py, turn.py
  _XG_EDIT    pending: edit llm.py: replace 3 lines → 6 lines
    diff --git a/llm.py b/llm.py
    index 4a1c9de..b7f0e21 100644
    --- a/llm.py
    +++ b/llm.py
    @@ -214,3 +214,6 @@
```

The display is two windows that answer two questions. The **graph** (with the
trail, and the current node's own description) is "where can I go". The **state**
is "where am I and what do I know", keyed by the node that introduced each piece.
Only the state scrolls, so where you are can never be scrolled out of sight.

The graph is drawn as a graph rather than as a tree, because it is one: a node
may be the child of two others, and a node may lead back to one of its own
ancestors. Each node is drawn under the one that reached it first, and every
other edge into it is named on that same line — so a shared node and a cycle both
stay visible without costing a line. A tree drawing would show the shared node
under one parent and silently lose the other edge.

It draws **from where you are, downward**. The nodes above have already been
walked and lead nowhere new, so leaving them out is what keeps the diagram
small — and the whole thing is capped, counting the nodes it did not draw rather
than growing until it pushes the input line off the screen.

There is no log of actions to read past. What each node produced *is* the state.

## Install

```sh
pipx install xg     # or: pip install xg
xg
```

`xg` reads the project in the working directory:

```sh
cd path/to/project
xg
```

## Requirements

- **Python 3.12+**
- **`TYPESAFE_API_KEY`** — the decision model ([TypeSafe Jev](https://docs.typesafe.ai)),
  which asks the routing questions. Read with `pass show jev` if the variable is unset.
- **`OPENROUTER_API_KEY`** — the executor, which speaks OpenRouter's
  OpenAI-compatible API. Read with `pass show pi/openrouter` if the variable is unset.

Both are optional at start-up: `xg` opens and works with nothing configured, and
says on the status line what is missing. Without a decision key, routing is off
and only explicit `/go` navigation is available.

## Using it

Type a line and press Enter. What a line means depends on its first character.

**A prompt** is free text: it is the goal, and it walks the workflow one node per
prompt. So `make the retry loop stop on a 4xx` reaches `_XG_FILTER`, and the next
line you type continues from there. A request that names a new file — `add a
CHANGELOG` — is routed to `_XG_ADD` instead, which shows you the file it would
create and waits for `ctrl+y`.

**A command** starts with `/` and is answered by the interface itself — no model
is asked anything.

| Command | Short | What it does |
| --- | --- | --- |
| `/go <node>` | `/g` | Go to a node. `/go ..` goes to the node above this one. |
| `/clear` | `/c` | Empty the prompt line. |
| `/help` | `/?` | List the commands. |

`/` is reserved, so a prompt cannot begin with one. A `/`-line that names no
command is always reported as a bad command rather than guessed to be prose.

| Key | What it does |
| --- | --- |
| `ctrl+y` | Accept the pending proposal. |
| `ctrl+n` | Reject it. |
| `ctrl+u` | Go to the parent node. |
| `pgup` / `pgdn` | Scroll the state. |
| `ctrl+l` | Empty the prompt line. |
| `ctrl+q` | Quit. |

The prompt line keeps what you typed after submitting, so a line that was routed
badly can be edited and sent again. Only `ctrl+l` and `/clear` empty it.

## The workflow

Every built-in node key is namespaced `_XG_`, because users add their own nodes
and branches.

```
_XG_ORIGIN ─┬─ _XG_COMMAND
            ├─ _XG_ADD
            └─ _XG_SELECT_MODULE ─ _XG_FILTER ─ _XG_SORT ─┬─ _XG_EDIT
                                                          └─ _XG_ANSWER
```

| Node | What it does |
| --- | --- |
| `_XG_ORIGIN` | Records the goal. Jev chooses the branch. |
| `_XG_COMMAND` | Proposes one shell command, with no project context at all. Gated. |
| `_XG_ADD` | Proposes one new file — a path that does not exist yet, and its whole content. Gated. |
| `_XG_SELECT_MODULE` | Finds the context module the request belongs to and bounds the read to it. |
| `_XG_FILTER` | Reads what is in scope, then asks Jev once per file whether it is relevant. |
| `_XG_SORT` | Ranks what survived, most important first, and Jev picks the leaf. |
| `_XG_EDIT` | Proposes one edit to the top-ranked file. Gated. |
| `_XG_ANSWER` | Answers in prose from the ranked files. Not gated — reading changes nothing. |

`_XG_COMMAND` and `_XG_ADD` are self-contained: the request says what to do and
there is nothing in the project to consult, so they hang directly off the origin.
The project workflow is the one that reads, which is why it is the one with
several steps.

A node with one child is taken without asking. Jev is consulted only where the
graph actually branches, and it may only ever descend: the graph is a workflow you
can walk, not a state you can wander.

## How decisions are made

**Deterministic first.** `.gitignore` and `.xgignore` decide what is never read,
and therefore what is never shown to a model. `.xgignore` is applied last, so it
wins, and it is the place to exclude local material — `.env`, keys — that git may
not track but nothing should read.

**Jev decides what is left.** Selection is one Noul question per file, answered in
a *single* request. If the state does not fit, the largest files are dropped and
the request is re-asked rather than split into batches: Jev scores each question
independently, so batching would make a file's score depend on which files
happened to share its request.

**Generative models are forced and they are last.** The executor is given the
files as hardcoded `read_file` tool calls — one per file, with its own result —
and then forced to call exactly one tool by name. `provider.require_parameters` is
set so the request cannot be routed to a backend that ignores that.

**Proposals are held.** A gated leaf shows what it would do and stops. `_XG_EDIT`
is previewed as a real git patch, built against the content the model was shown,
and `git apply` accepts it. `_XG_ADD` is previewed as a creation patch —
`new file mode`, `/dev/null` on the from side — so what you read is the file being
added. The write itself re-reads the file and refuses an ambiguous or stale match,
so a preview that has gone out of date cannot become a wrong write; an add
refuses a path that turns out to exist, rather than overwriting it.

## Context modules

A **context module** is any folder containing `.xg/module.json`. The manifest says
what the folder offers as context and what it needs:

```json
{
  "name": "graph",
  "description": "The node registry, the routing policy, and the workflows.",
  "expose": ["*.py", "read_tree.py"],
  "import": ["../read"]
}
```

- `description` is required and is the routing signal: it is what Jev reads when
  deciding whether this module's context is what the request is about.
- `expose` is what the module contributes, as paths and globs relative to its own
  folder. `"."` means the whole folder.
- `import` adds another module's exposures, transitively.

Exposure is a **bound, not a hint**. Once a module is selected, the files it and
its imports expose are the only files the rest of the run may read — enforced by
the read itself, so there is no path around it. Ignores still win over exposure,
and `.xg` is never read as content nor exposed by `"."`.

One declared module is the context without being asked about. Several are put to
Jev, over their manifests only — no file content is read to choose a module. No
module above the threshold means the whole project is in scope, which is the
behaviour before modules existed, reached by the absence of a declaration.

Unknown keys in a manifest are **refused**, not ignored: a misspelled `expose`
would otherwise silently mean "exposes nothing".

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check src tests
```

The test suite is offline by design: every network boundary is injected, so a run
passes or fails on the code rather than on a key and a round trip.

## License

MIT. See [LICENSE](LICENSE).
