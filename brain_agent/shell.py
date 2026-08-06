"""PyShell — the persistent Python namespace.

This is the execution primitive only. `kernel.py` puts it in a daemon so it
outlives the caller, and `orchestrator.py` hands it to an agent as a real tool.

REMOVED in v0.8.0: `BrainShell`, along with ROOT_SYSTEM, `_extract_code`,
`_describe` and the `P` plumbing. That was a driver loop that scraped fenced
code blocks out of the model's prose, capped the agent at `max_iters` turns, and
sat beside the brain pattern instead of being it. Every one of those was a limit
imposed by the harness rather than by the idea. The agent now calls its shell as
a tool and decides for itself when it is finished — see `Orchestrator`.
"""
from __future__ import annotations

import ast
import inspect
import re
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import trace          # stdlib-only; recording is a no-op unless enabled

STDOUT_PREVIEW = 2000          # chars of stdout the root is allowed to see
MAX_ITERS = 24


# ── the policy hook (what "don't call insane shit" means, and it's yours) ─────

DEFAULT_DENY = ("shutil.rmtree", "os.system", "subprocess", "os.remove",
                "os.unlink", "sys.exit", "__import__('os').system")


def _shell_print(*args, sep=" ", end="\n", file=None, flush=False):
    """The shell's own `print`.

    heaven_base monkeypatches `builtins.print` (verified: `builtins.print.
    __module__ == 'heaven_base'`), and its replacement writes to neither
    sys.stdout nor fd 1 — so agent `print()` output vanished entirely and no
    amount of redirect_stdout or dup2 could catch it. Binding print in the
    exec namespace makes the agent's output ours again, whatever the ambient
    builtins have been patched to.
    """
    stream = file if file is not None else sys.stdout
    stream.write(sep.join(str(a) for a in args) + end)
    if flush:
        stream.flush()


def default_policy(src: str) -> Optional[str]:
    """Return a refusal string to block, or None to allow. Replace wholesale by
    passing `policy=` to BrainShell — this is a knob, not a sandbox."""
    for bad in DEFAULT_DENY:
        if bad in src:
            return f"blocked by policy: {bad!r}"
    return None


# ── the persistent namespace ─────────────────────────────────────────────────

@dataclass
class PyShell:
    """A persistent Python namespace. Variables, imports, functions and brains
    built in one call are still there on the next."""
    ns: dict = field(default_factory=dict)
    policy: Optional[Callable] = default_policy

    async def run(self, src: str) -> str:
        """Execute a cell; return captured stdout (plus traceback on error).

        Capture is at the FILE DESCRIPTOR level, not via `contextlib.
        redirect_stdout`. An agent shell runs arbitrary code: libraries rebind
        `builtins.print`, replace `sys.stdout`, or write to fd 1 from C — all of
        which silently defeat redirect_stdout. (Observed: importing the heaven
        chain breaks redirect_stdout outright.) dup2 catches every case.
        """
        if self.policy:
            refusal = self.policy(src)
            if refusal:
                return refusal

        err: Optional[str] = None
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as tmp:
            saved_fd = os.dup(1)
            saved_stdout = sys.stdout
            try:
                sys.stdout.flush()
                os.dup2(tmp.fileno(), 1)
                # Also point sys.stdout at fd 1 so Python-level writes land there
                # even if something replaced the original object.
                sys.stdout = os.fdopen(os.dup(1), "w", encoding="utf-8", errors="replace")
                self.ns["print"] = _shell_print     # beat any patched builtins.print
                try:
                    code = compile(src, "<brain-shell>", "exec",
                                   flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                    result = eval(code, self.ns)      # noqa: S307 — that is the point
                    if inspect.iscoroutine(result):
                        await result
                except BaseException:                  # noqa: BLE001 — report, never raise
                    err = traceback.format_exc(limit=4)
                finally:
                    try:
                        sys.stdout.flush()
                        sys.stdout.close()
                    except Exception:
                        pass
            finally:
                os.dup2(saved_fd, 1)
                os.close(saved_fd)
                sys.stdout = saved_stdout
            tmp.seek(0)
            out = tmp.read()
        return out + (err or "")
