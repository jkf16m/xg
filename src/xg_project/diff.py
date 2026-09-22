"""A proposed change, rendered as the patch that would describe it.

An edit proposal is a literal ``old_text`` swapped for ``new_text``. That is
enough to apply, but not enough to *read*: two fragments do not say where in the
file they sit or what surrounds them. A unified diff does, and git's flavour of
one is the format a reader already knows how to scan for the changed lines.

The patch is built against the content the executor was shown, not the file on
disk. Those are the same thing unless the file changed while the proposal waited,
and when they differ the patch is the truthful record of what was proposed.
``apply_edit`` re-reads the file and refuses anything ambiguous, so a stale
preview cannot turn into a wrong write.

The ``index`` line carries real git blob hashes, so what comes out is the patch
git would have printed. The mode is assumed to be ``100644``: nothing here
changes a mode, and reading the real one would make a function that renders a
value it already holds go and touch the filesystem for it.

Creating a file is the same rendering with a different header. Git marks it with
``new file mode``, an all-zero "before" hash, and ``/dev/null`` on the from side
of the hunk; writing those markers rather than a modification's is what makes
``git apply`` accept the result as the creation it is.
"""

from __future__ import annotations

import difflib
import hashlib


def _blob(text: str) -> str:
    """Git's abbreviated object id for ``text`` read as a blob.

    Git hashes ``blob <length>\\0<content>``, and this is that hash, so the
    patch's ``index`` line is the line git would have written rather than
    something shaped like one.
    """
    encoded = text.encode("utf-8")
    header = f"blob {len(encoded)}\0".encode()
    return hashlib.sha1(header + encoded, usedforsecurity=False).hexdigest()[:7]


def unified_patch(
    *, path: str, before: str, after: str, context: int = 3, new_file: bool = False
) -> str:
    """A git patch turning ``before`` into ``after`` at ``path``, or ``""``.

    Empty when there is nothing to show: content that did not change has no
    hunks, and a patch with no hunks is not a preview of anything.

    ``new_file`` renders the change as the creation of a file that was not there,
    which is a different patch rather than a modification against nothing: git
    wants ``new file mode``, a zeroed from-hash, and ``/dev/null`` where the old
    side of the hunk would be. ``before`` is expected to be empty when it is set.

    The result is newline-terminated, which is what makes it a patch git will
    accept rather than one it calls corrupt. A file whose last line carries no
    terminating newline is the one case rendered without git's
    ``\\ No newline at end of file`` marker; the hunks still read correctly, and
    the trade is deliberate against reimplementing the diff to place a marker
    almost no file needs.
    """
    if before == after:
        return ""

    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="/dev/null" if new_file else f"a/{path}",
            tofile=f"b/{path}",
            n=context,
            lineterm="",
        )
    )
    if not lines:
        return ""

    header = [f"diff --git a/{path} b/{path}"]
    if new_file:
        header += ["new file mode 100644", f"index {'0' * 7}..{_blob(after)}"]
    else:
        header.append(f"index {_blob(before)}..{_blob(after)} 100644")

    return "\n".join([*header, *lines]) + "\n"
