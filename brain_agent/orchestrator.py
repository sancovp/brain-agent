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

import contextvars

_CURRENT_KERNEL: contextvars.ContextVar[str] = contextvars.ContextVar(
    "brain_current_kernel", default="rlm")


def set_current_kernel(name: str) -> None:
    _CURRENT_KERNEL.set(name)


async def shell_func(code: str, kernel: Optional[str] = None) -> str:
    """Execute code in the persistent runtime context and return its stdout."""
    from .kernel import Kernel
    k = Kernel(kernel or _CURRENT_KERNEL.get())
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
# The hierarchical digest-pyramid brain, under its own name (its class is also
# called Brain, which is why importing it raw would shadow the composable one).
from brain_agent.hierarchical import Brain as DigestBrain
from brain_agent.hierarchical import build_digests as build_digests

# THE state dict: context slots, each persisted to a chunk file on write
from brain_agent.context import ContextDict
import os as _os
CONTEXT = ContextDict(_os.path.join(
    _os.environ.get("BRAIN_CONTEXT_DIR", "/tmp/brain-contexts"), __kernel_name__))
FINAL = None

# existing brains: the brain_configs registry the original BrainAgent uses

def _brain_registry():
    # Read the registry FILE. This heaven version formats every registry_util
    # operation (get and get_all alike) into a display string, so the on-disk
    # JSON is the only machine-readable surface.
    import json, os
    from heaven_base.utils.get_env_value import EnvConfigUtil
    path = os.path.join(EnvConfigUtil.get_heaven_data_dir(), "registry",
                        "brain_configs_registry.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def list_brains():
    "Names of every registered brain."
    return sorted(_brain_registry())

def load_brain(name, router=None):
    "A REGISTERED brain as a composable Brain value (usable as a neuron)."
    import os
    entries = _brain_registry()
    if name not in entries:
        raise KeyError(f"brain {name!r} not registered; see list_brains()")
    cfg = entries[name]
    cfg = cfg.get("value_dict", cfg) if isinstance(cfg, dict) else cfg
    src = cfg.get("neuron_source") or cfg.get("directory")
    if not os.path.isabs(src):
        from heaven_base.utils.get_env_value import EnvConfigUtil
        src = os.path.join(EnvConfigUtil.get_heaven_data_dir(), src)
    return from_dir(src, router=router, name=name)

def register_brain(directory, brain_name, chunk_size=-1):
    from brain_agent.brain_agent import register_brain as _rb
    return _rb(directory, brain_name, chunk_size)

# recursion: another synthesizer AGENT as a callable

__sub_seq__ = [0]

async def sub_rlm(task, bind=None, kernel=None):
    "Delegate to ANOTHER synthesizer agent with its OWN runtime context. It"
    "builds whatever brain systems it needs and returns its answer. bind is a"
    "dict of values placed into its context by name. Children can call"
    "sub_rlm themselves, so synthesizers stack as deep as the task needs."
    from brain_agent.orchestrator import Orchestrator
    __sub_seq__[0] += 1
    name = kernel or f"{__kernel_name__}-sub{__sub_seq__[0]}"
    with trace.span("sub_rlm", str(task)[:80], child_kernel=name):
        child = Orchestrator(kernel=name)
        if bind:
            child.bind(**bind)
        return await child.run(task)
"""
SYSTEM = """You are a synthesizer agent. You run a Recursive Language Model.

Your working memory is CONTEXT — a state dict in your persistent Python shell
(ShellTool). Each slot is a chunk of context, persisted to a file the moment
you set it. You never hold the material yourself; you ENGINEER it:

  THE LOOP
  1. Get material into CONTEXT slots: CONTEXT["a"] = text, or slice an
     existing slot into smaller ones. Inspect with print(CONTEXT.slots()).
  2. Make vars and PIPE them: call neurons/brains ON slot files, hold results
     in vars, write what matters back into new slots.
  3. Repeat — reslice, relens, rebuild — until you are satisfied.
  4. FINAL = the answer. That ends the work; FINAL is returned from the shell,
     so its length is not limited by your reply.

The LLMs you call are neurons and brains reading those chunk FILES:

  Neuron(content=CONTEXT.path("a"), name=..., prompt=...)
      one slot read through a LENS. prompt= IS the lens — what to report,
      what to ignore. Same slot + different lenses = independent readings.
      cognize_prompt= is its self-relevance lens. await n(query) reads it.
  Synthesizer(prompt=...)          folds readings: await synth(parts) where
      parts is a dict {name: text}, list of (name, text), or list of strings.
  Brain(neurons=[...], synthesizer=..., router=...)
      N neurons + 1 synthesizer; a Brain is usable AS a neuron, so systems
      nest. await brain.query(q); await brain.vote(q) for raw scores; routers
      are plain callables and writing your own is normal.
  CONTEXT.brain()                  the whole CONTEXT as a brain — every slot a
      neuron. Cognize/judge your own working context.
  await fanout(items, query)       N calls at once (Neurons keep their lenses).
  await sub_llm(prompt, content)   one call.
  await gather(...)                anything concurrently, including brains.
  from_dir(path)                   a directory tree as a brain.
  list_brains() / load_brain(name) the registry of EXISTING brains; a loaded
      brain is a value — query it or stack it as a neuron.
  register_brain(directory, name)  register a reusable brain (CONTEXT.dir
      works: your engineered context can become a permanent brain).
  DigestBrain(Path(dir)) / await build_digests(Path(dir))
      the digest-pyramid brain for BIG corpora: fold once, then witnessed
      descent. Prefer it over hand-made neurons for large trees.
  await sub_rlm(task, bind={...})  ANOTHER synthesizer agent with its own
      shell and its own CONTEXT; bind puts values into its slots. It can call
      sub_rlm itself — stack synthesizers as deep as the task needs. Run
      several at once with gather.

Granularity is a cost dial — choose chunk size per task, not by habit:
  * A cognize sweep reads the whole corpus regardless of chunking; what scales
    with 1/chunk-size is the duplicated lens+query prompt, the per-call output
    (N reasoning blobs), and rate-limit pressure. Measured on a 186k-token
    corpus: ~30k-token chunks -> 8 neurons, perfect recall; ~120k -> 3 neurons,
    still perfect on a needle task; ~3k -> 63 neurons, rate-limited before it
    could finish.
  * Default COARSE (30k-120k tokens per slot) for retrieval and routing. Go
    fine only for dense per-line work (classify/aggregate every row), where big
    chunks degrade first. Relevance scores weaken as chunks grow (10 -> 8
    observed); if a coarse sweep scores are soft, that region wants rechunking.
  * Cascade: route coarse, then reslice ONLY the winning slot finer. Total cost
    ~ corpus + winner, instead of corpus at fine granularity.

Rules:
  * Everything above is ALREADY IN SCOPE in the shell. Do not import from
    brain_agent — the package exports a different, directory-based Brain that
    would shadow yours.
  * The shell PERSISTS between calls. Never re-run a neuron, fanout, or brain
    you already ran — its result is in a var or a slot; reuse it.
  * Print conclusions and CONTEXT.slots(), never whole chunks. Tool output is
    truncated for you; slots and vars are not.
  * You decide when the answer is good enough. There is no turn limit. End by
    setting FINAL in the shell.
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
        self.k.exec(f"__kernel_name__ = {self.kernel!r}")
        if "Neuron" not in self.k.vars():
            self.k.exec(BOOTSTRAP)

    def bind(self, **objects: Any) -> "Orchestrator":
        """Put values into CONTEXT slots by name. Large strings travel via a
        file so they never ride a socket payload twice. Raises on kernel
        errors — a bind that silently fails leaves the agent working on
        context that is not there (observed)."""
        import json
        import tempfile
        for name, value in objects.items():
            if isinstance(value, str) and len(value) > 8000:
                p = os.path.join(tempfile.gettempdir(), f"brain-bind-{self.kernel}-{name}.txt")
                with open(p, "w", encoding="utf-8", errors="replace") as fh:
                    fh.write(value)
                out = self.k.exec(f"CONTEXT[{name!r}] = Path({p!r}).read_text(errors='replace')\n")
            else:
                lit = repr(value) if isinstance(value, str) else json.dumps(value)
                out = self.k.exec(f"CONTEXT[{name!r}] = {lit}\n")
            if "Error" in out or "Traceback" in out:
                raise RuntimeError(f"bind({name!r}) failed in kernel: {out[:300]}")
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
        # FINAL lives in the kernel, so the answer is not bounded by the
        # model's output window; the reply text is the fallback.
        final = self.k.getvar("FINAL")
        return final if final is not None else _final_text(result)


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
