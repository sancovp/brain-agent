"""The shell — ONE interactive Python REPL handed to the top-level agent, with
the brain-agent SDK preloaded.

This is the piece that was missing. Previously the traversal was a fixed Python
program with the LLM as a scoring function inside it. Here it is inverted: the
LLM writes the program, at runtime, per query.

  * The prompt/corpus is a VARIABLE (`P`) in a live namespace — never pasted
    into the root context. The root only ever sees metadata about it.
  * The SDK is preloaded: Neuron, Synthesizer, Brain, fanout, sub_llm, from_dir,
    the routers. The agent CONSTRUCTS brains, hierarchies, synthesizers and
    neurons in the shell and calls whatever it wants.
  * `sub_rlm(...)` opens a NESTED shell with the same SDK → that sub-agent is
    itself an RLM, so recursion is structural, not special-cased.
  * The loop ends when the agent sets `Final` in the namespace. The answer comes
    out of a variable, so it is not bounded by the root's output window.

    shell = BrainShell(P=big_string_or_path)
    answer = await shell.run("which invoices disagree with the ledger?")
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

STDOUT_PREVIEW = 2000          # chars of stdout the root is allowed to see
MAX_ITERS = 24


# ── the policy hook (what "don't call insane shit" means, and it's yours) ─────

DEFAULT_DENY = ("shutil.rmtree", "os.system", "subprocess", "os.remove",
                "os.unlink", "sys.exit", "__import__('os').system")


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


# ── the root loop (Algorithm 1) ──────────────────────────────────────────────

ROOT_SYSTEM = """You are the root of a Recursive Language Model. You do NOT
answer from your own context — you write Python in a persistent shell and let
sub-calls read the data.

The corpus is already in the shell as the variable `P`. You have never seen it
and you never will; inspect it with code.

Preloaded (brain-agent SDK):
  Neuron(content, name=..., prompt=...)      one chunk + one prompt; await it
  Synthesizer(prompt=...)                    folds [(name, text), ...] -> answer
  Brain(neurons=[...], synthesizer=..., router=...)   N neurons + 1 synthesizer;
                                             a Brain is usable AS a neuron, so
                                             nest them for hierarchies
  await brain.vote(q)                        raw relevance scores; route yourself
  await fanout(chunks, q)                    N parallel sub-calls -> list[str]
  await sub_llm(prompt, content)             one sub-call
  await sub_rlm(task, P=...)                 a NESTED shell agent (a sub-RLM)
  from_dir(path)                             build a Brain from a directory tree
  open_all_router / top_k_router(k, threshold) / threshold_router(t)
Plus the whole standard library.

Rules:
  * Emit ONE ```python cell per turn. Only stdout comes back, truncated — do not
    print the corpus, print what you concluded.
  * Build the decomposition yourself: slice P, construct neurons, choose N,
    write your own router if the default is wrong.
  * When done, assign the answer to `Final` in the shell. That ends the loop and
    its value is returned verbatim, so it is not limited by your output window.
"""


def _extract_code(text: str) -> Optional[str]:
    if "```" not in text:
        return None
    block = text.split("```", 2)[1]
    if block.startswith("python"):
        block = block[len("python"):]
    return block.strip("\n") or None


def _describe(value: Any) -> str:
    if isinstance(value, Path):
        return f"P is a Path: {value} (dir)" if value.is_dir() else \
               f"P is a Path: {value} ({value.stat().st_size} bytes)"
    if isinstance(value, str):
        return f"P is a str of {len(value)} chars. First 400:\n{value[:400]}"
    return f"P is a {type(value).__name__}: {str(value)[:400]}"


@dataclass
class BrainShell:
    """One shell + one root agent driving it."""
    P: Any = ""
    policy: Optional[Callable] = default_policy
    max_iters: int = MAX_ITERS
    depth: int = 0
    max_depth: int = 2
    transcript: list = field(default_factory=list)

    def __post_init__(self):
        # Lazy so PyShell stays importable heaven-free (the v0.3.1 packaging goal).
        from . import sdk
        from .sdk import (Brain, Neuron, Synthesizer, fanout, sub_llm, from_dir,
                          open_all_router, top_k_router, threshold_router)
        self.shell = PyShell(policy=self.policy)
        self.shell.ns.update({
            "P": self.P, "Final": None,
            "Neuron": Neuron, "Synthesizer": Synthesizer, "Brain": Brain,
            "fanout": fanout, "sub_llm": sub_llm, "from_dir": from_dir,
            "open_all_router": open_all_router, "top_k_router": top_k_router,
            "threshold_router": threshold_router, "sdk": sdk, "Path": Path,
            "sub_rlm": self._make_sub_rlm(),
        })

    def _make_sub_rlm(self) -> Callable:
        async def sub_rlm(task: str, P: Any = "", **kw) -> str:
            """Open a nested shell with the same SDK. The child is an RLM too."""
            if self.depth >= self.max_depth:
                from .sdk import sub_llm
                return await sub_llm(task, P)
            child = BrainShell(P=P, policy=self.policy, depth=self.depth + 1,
                               max_depth=self.max_depth, **kw)
            return await child.run(task)
        return sub_rlm

    async def run(self, task: str) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        from .hierarchical import _batch, _content
        hist = [SystemMessage(content=ROOT_SYSTEM),
                HumanMessage(content=f"{_describe(self.P)}\n\nTask: {task}")]
        for i in range(self.max_iters):
            resp = await _batch([hist], max_tokens=2000)
            reply = _content(resp[0]) if resp else ""
            hist.append(SystemMessage(content=f"[assistant turn {i}]\n{reply}"))
            code = _extract_code(reply)
            if code is None:
                hist.append(HumanMessage(content=
                    "No ```python cell found. Emit one, or set Final."))
                continue
            out = await self.shell.run(code)
            self.transcript.append({"code": code, "stdout": out})
            final = self.shell.ns.get("Final")
            if final is not None:
                return final if isinstance(final, str) else str(final)
            meta = (f"[stdout {len(out)} chars]\n{out[:STDOUT_PREVIEW]}"
                    + ("\n…truncated" if len(out) > STDOUT_PREVIEW else ""))
            hist.append(HumanMessage(content=meta or "[no output]"))
        final = self.shell.ns.get("Final")
        return str(final) if final is not None else \
            f"NOT_FOUND (root hit {self.max_iters} iterations without setting Final)"
