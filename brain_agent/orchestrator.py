"""The orchestrator — an agent with ONE persistent runtime context in which it
MAKES brain systems and calls them.

This replaces `BrainShell`, which was the wrong shape: it was a second construct
sitting next to the brain pattern, with a turn cap, a magic `P` variable, and a
loop that scraped fenced code blocks out of the model's prose. All three were
limits I imposed, not properties of the idea.

What this is instead. A brain is N neurons + 1 synthesizer. In an RLM the
synthesizer stops being a one-shot fold and becomes the thing that RUNS THE
TURN: it calls neurons however it wants, as many as it wants, as many at once as
it wants, until it is satisfied with the answer. So:

  * The shell is a real TOOL (`ShellTool`), invoked through heaven's native tool
    loop. The model's output formatting is no longer load-bearing.
  * There is NO turn cap. The agent manages its own work and stops when it is
    done, not when a counter runs out.
  * The kernel is a single runtime context. Neurons, brains, whole systems it
    builds are live Python objects there, and results pipe between them as
    values — never re-serialized through the model's context.
  * A Brain is usable as a neuron, so a system it builds can contain brains
    whose synthesizers drive in turn. Recursion needs no special case.

    orch = Orchestrator(kernel="work")
    orch.bind(corpus=big_string)              # anything, by name, into the context
    answer = await orch.run("find every disputed invoice and reconcile them")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import trace

# Effectively unbounded. The agent decides when it is finished; this exists only
# so a runaway cannot spin forever, and is deliberately far above any real run.
UNBOUNDED_TOOL_CALLS = int(os.environ.get("BRAIN_MAX_TOOL_CALLS", "1000"))

# What the agent sees per call. The FULL result stays in the runtime context as a
# variable — truncation here bounds the model's context, never its data.
TOOL_OUTPUT_LIMIT = 4000

_CURRENT_KERNEL = "rlm"


def set_current_kernel(name: str) -> None:
    global _CURRENT_KERNEL
    _CURRENT_KERNEL = name


async def shell_func(code: str, kernel: Optional[str] = None) -> str:
    """Execute code in the persistent runtime context and return its stdout."""
    from .kernel import Kernel
    k = Kernel(kernel or _CURRENT_KERNEL)
    k.start()
    # Span the tool call itself: moving from a scraped-cell loop to a real tool
    # otherwise drops every cell from the call graph, which is exactly the
    # visibility needed to see what the agent built.
    first = next((ln for ln in code.strip().splitlines() if ln.strip()), "")
    with trace.span("shell", first[:80], chars=len(code)):
        out = k.exec(code, parent=trace.current_parent())
    if not out.strip():
        return "(no output — the cell ran; print what you want to see)"
    if len(out) > TOOL_OUTPUT_LIMIT:
        return (out[:TOOL_OUTPUT_LIMIT]
                + f"\n…[{len(out) - TOOL_OUTPUT_LIMIT} more chars withheld; the full "
                  "value is still in the context — slice the variable]")
    return out


def _build_tool_classes():
    """Built lazily so this module imports without heaven present."""
    from heaven_base import BaseHeavenTool, ToolArgsSchema

    class ShellToolArgsSchema(ToolArgsSchema):
        arguments: Dict[str, Dict[str, Any]] = {
            "code": {
                "name": "code",
                "type": "str",
                "description": (
                    "Python to run in your persistent runtime context. Variables, "
                    "imports, functions, and any neurons/brains/systems you build "
                    "stay alive between calls. `await` works at top level. Print "
                    "what you want to see; everything else stays as a variable."),
                "required": True,
            },
        }

    class ShellTool(BaseHeavenTool):
        name = "ShellTool"
        description = (
            "Your runtime context: a persistent Python shell with the brain-agent "
            "SDK preloaded. Build neurons, synthesizers, brains and whole systems "
            "here and call them. State persists across every call.")
        args_schema = ShellToolArgsSchema
        func = shell_func
        is_async = True

    return ShellTool


BOOTSTRAP = """
import asyncio as asyncio
from asyncio import gather as gather
from pathlib import Path as Path
import brain_agent.sdk as sdk
import brain_agent.trace as trace
from brain_agent.sdk import (Neuron, Synthesizer, Brain, fanout, sub_llm,
                             from_dir, open_all_router, top_k_router,
                             threshold_router)
"""

SYSTEM = """You are a synthesizer that builds and runs brain systems.

You have ONE persistent runtime context, reached with ShellTool. Everything you
make there stays alive: variables, functions, neurons, brains, whole systems.
Results pipe between them as live Python values — they never have to pass back
through you as text.

What you build with (already in scope — do NOT import from brain_agent, the
package exports a different, directory-based Brain that would shadow these):

  Neuron(content=..., name=..., prompt=...)
      One chunk of context read through a LENS. `prompt=` IS the lens: it
      decides what this neuron reports and what it ignores. The same content
      under different lenses gives independent readings that cannot contaminate
      each other. `cognize_prompt=` is the separate lens it uses to score its
      own relevance. `await n(query)` reads it.

  Synthesizer(prompt=...)
      Folds readings into one result; `prompt=` controls the form (JSON, a
      table, a ranking). `await synth(parts)` where parts is a dict
      {name: text}, a list of (name, text), or a list of strings.

  Brain(neurons=[...], synthesizer=..., router=...)
      N neurons + 1 synthesizer. A Brain is itself usable AS a neuron, so
      brains nest into systems of any depth. `await brain.query(q)` runs it;
      `await brain.vote(q)` returns raw relevance scores if you want to route
      it yourself. Routers are plain callables (query, votes, neurons) -> list
      of indices; open_all_router / top_k_router(k, threshold) /
      threshold_router(t) are there if you want them, and writing your own is
      normal.

  await fanout(items, query)     N calls at once. Items may be Neurons (each
                                 keeps its own lens) or raw content.
  await sub_llm(prompt, content) one call.
  await gather(...)              run anything concurrently, including whole
                                 brains — as many at once as you want.
  from_dir(path)                 build a Brain from a directory tree.

How to work:

  * Make the system the problem actually needs. Chunk the material, choose the
    lenses, decide how many neurons, compose them into brains, nest brains into
    larger systems. Build it on the fly and rebuild it when it is wrong.
  * Call whatever you want, whenever you want, as many at once as you want.
    There is no turn limit. You decide when the answer is good enough.
  * Never re-run work you have already done — assign it to a variable and reuse
    it. The context persists across every call.
  * Keep large data in variables, not in your own context. Print conclusions,
    not corpora. Tool output is truncated; the variable is not.
  * When you are satisfied, give the final answer in your reply. If it is large,
    also leave it in a variable and say which one.
"""


@dataclass
class Orchestrator:
    """One agent, one runtime context, no turn cap."""
    kernel: str = "rlm"
    reset: bool = False
    max_tool_calls: int = UNBOUNDED_TOOL_CALLS
    model: Optional[str] = None
    system_prompt: str = SYSTEM
    _bound: dict = field(default_factory=dict)

    def __post_init__(self):
        from .kernel import Kernel
        self.k = Kernel(self.kernel)
        if self.reset and self.k.alive():
            self.k.shutdown()
        self.k.start()
        set_current_kernel(self.kernel)
        if "Neuron" not in self.k.vars():
            self.k.exec(BOOTSTRAP)

    def bind(self, **objects: Any) -> "Orchestrator":
        """Put values into the runtime context by name. Large strings go via a
        file so they are never sent through a socket payload twice."""
        import json
        import tempfile
        for name, value in objects.items():
            if isinstance(value, str) and len(value) > 8000:
                p = os.path.join(tempfile.gettempdir(), f"brain-bind-{self.kernel}-{name}.txt")
                with open(p, "w", encoding="utf-8", errors="replace") as fh:
                    fh.write(value)
                self.k.exec(f"{name} = Path({p!r}).read_text(errors='replace')\n")
            else:
                self.k.exec(f"{name} = {json.dumps(value) if not isinstance(value, str) else repr(value)}\n")
            self._bound[name] = type(value).__name__
        return self

    def exec(self, code: str) -> str:
        return self.k.exec(code)

    def vars(self) -> list:
        return self.k.vars()

    async def run(self, task: str) -> str:
        from heaven_base import HeavenAgentConfig, UnifiedChat, ProviderEnum
        from heaven_base.baseheavenagent import BaseHeavenAgent
        from .hierarchical import MODEL, PROVIDER

        set_current_kernel(self.kernel)
        ShellTool = _build_tool_classes()
        inventory = ("\n\nAlready in your runtime context: "
                     + ", ".join(f"{n} ({t})" for n, t in self._bound.items())
                     if self._bound else "")
        config = HeavenAgentConfig(
            name="Orchestrator",
            system_prompt=self.system_prompt + inventory,
            tools=[ShellTool],
            provider=PROVIDER,
            model=self.model or MODEL,
            temperature=0.2,
            max_tokens=8000,
        )
        with trace.span("orchestrator", task[:120], kernel=self.kernel):
            agent = BaseHeavenAgent(config, UnifiedChat(),
                                    max_tool_calls=self.max_tool_calls)
            result = await agent.run(prompt=task)
        return _final_text(result)


def _message_text(msg: Any) -> str:
    """Text of one message, ignoring tool_use/thinking blocks."""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type", "text") == "text")
    return ""


def _final_text(result: Any) -> str:
    """Pull the agent's answer out of whatever heaven hands back.

    heaven returns a DICT ({"history": History(...)}), not an object with a
    .history attribute — assuming otherwise made this return a 39,000-character
    repr of the entire conversation instead of the answer.
    """
    if isinstance(result, str):
        return result
    hist = result.get("history") if isinstance(result, dict) else getattr(result, "history", result)
    messages = getattr(hist, "messages", None)
    if messages is None and isinstance(hist, dict):
        messages = hist.get("messages")
    for msg in reversed(messages or []):
        if type(msg).__name__ not in ("AIMessage", "AIMessageChunk"):
            continue
        text = _message_text(msg).strip()
        if text:
            return text
    return str(result)
