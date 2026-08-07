"""The brain-agent SDK — the primitives a SHELL AGENT composes at runtime.

This is the layer `shell.py` preloads into the agent's REPL. Everything here is
a plain Python object the agent can construct, inspect, subclass, put in a list,
loop over, and call. Nothing walks a directory unless the agent asks it to.

  Neuron       one context chunk + one prompt. Callable.
  Synthesizer  one prompt that folds N neuron outputs into one answer. Callable.
  Brain        N neurons + 1 synthesizer + a ROUTER the agent supplies. Callable.
               A Brain is itself usable as a neuron → hierarchies nest.
  fanout()     the raw N-parallel-neuron primitive (one heaven abatch).
  from_dir()   adapter: build a Brain out of a directory (the old behavior,
               now just one constructor among many).

The router is a plain callable `(query, votes, neurons) -> list[index]`. The old
hardcoded top-k lives in `top_k_router` as a DEFAULT, not as the mechanism —
the agent is free to write its own in the shell, which is the point.

All LLM calls go through heaven (`hierarchical._batch`) exactly as before.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from langchain_core.messages import SystemMessage, HumanMessage

from . import trace
from .hierarchical import (
    COGNIZE_SYSTEM, INSTRUCT_SYSTEM, BRAIN_COGNIZE_SYSTEM,
    _batch, _content, _score, _neuron_messages,
)

Content = Union[str, Path]

DEFAULT_SYNTH_PROMPT = (
    "You are a Brain synthesizer. Combine the instruction blocks into one "
    "coherent answer to the query. Every claim must trace to a verbatim quote "
    "in the blocks; cite the source for each fact; never add information not "
    "present in the blocks. If the blocks do not answer the query, say NOT_FOUND.")


# Appended to EVERY neuron system prompt, built-in or custom. Without it a terse
# custom lens ("Report only amounts.") lets the model answer the query from
# general knowledge and ignore its chunk entirely — observed live, and it fails
# silently, which is the worst way for a lens to fail. Grounding is an invariant
# of what a neuron IS, not something each lens should have to restate.
NEURON_FRAME = ("\n\nThe <neuron content> block below is your ENTIRE knowledge for "
                "this task. Answer only from it, never from general knowledge or "
                "from the wording of the query.")


def _messages(content: Content, query: str, system: str) -> list:
    """Render one neuron call. A Path goes through heaven's path= block (so file
    handling stays identical to tools.py); a str is inlined directly."""
    system = system + NEURON_FRAME
    if isinstance(content, Path):
        return _neuron_messages(str(content), query, system)
    return [SystemMessage(content=f"{system}\n\n<neuron content>\n{content}\n</neuron content>"),
            HumanMessage(content=f"Query: {query}")]


# ── the three primitives ─────────────────────────────────────────────────────

@dataclass
class Neuron:
    """One chunk of context + one prompt. The atom."""
    content: Content
    name: str = "neuron"
    prompt: str = INSTRUCT_SYSTEM
    cognize_prompt: str = COGNIZE_SYSTEM

    def instruct_msgs(self, query: str) -> list:
        return _messages(self.content, query, self.prompt)

    def cognize_msgs(self, query: str) -> list:
        return _messages(self.content, query, self.cognize_prompt)

    async def instruct(self, query: str, *, max_tokens: int = 3000) -> str:
        with trace.span("neuron", self.name):
            resp = await _batch([self.instruct_msgs(query)], max_tokens=max_tokens)
            return _content(resp[0]) if resp else "NOT_FOUND"

    async def cognize(self, query: str) -> dict:
        resp = await _batch([self.cognize_msgs(query)], max_tokens=700)
        return _score(_content(resp[0])) if resp else {"score": 0, "reasoning": "no response"}

    async def __call__(self, query: str, **kw) -> str:
        return await self.instruct(query, **kw)


@dataclass
class Synthesizer:
    """Folds N (name, text) blocks into one answer."""
    prompt: str = DEFAULT_SYNTH_PROMPT
    name: str = "synthesizer"
    max_tokens: int = 4000

    @staticmethod
    def _normalize(parts) -> list:
        """Accept whatever the caller naturally has.

        The documented form is [(name, text)], but an agent holding readings in
        a dict or a plain list is the common case, and both used to raise
        `ValueError: too many values to unpack` from inside the join — observed
        live, after which the agent gave up on Synthesizer entirely and
        hand-rolled a prompt. Meeting the caller where it is costs four lines.
        """
        if isinstance(parts, dict):
            return [(str(k), str(v)) for k, v in parts.items()]
        out = []
        for i, p in enumerate(parts):
            if isinstance(p, (tuple, list)) and len(p) == 2:
                out.append((str(p[0]), str(p[1])))
            else:
                out.append((f"source_{i}", str(p)))
        return out

    async def __call__(self, query=None, parts=None) -> str:
      # `synth(parts)` is what callers reach for first — the query is often
      # implicit in the lens prompt. Observed live: an agent burned three turns
      # on the arity and then abandoned Synthesizer for a hand-rolled prompt.
      # One positional arg means it IS the parts.
      if parts is None:
          query, parts = "", query
      if parts is None:
          return "NOT_FOUND (synthesizer: nothing to combine)"
      parts = self._normalize(parts)
      with trace.span("synth", self.name, n=len(parts)):
        blocks = "\n\n".join(f"<from source='{n}'>\n{t}\n</from>" for n, t in parts)
        user = f"Query: {query}\n\n{blocks}" if query else blocks
        resp = await _batch([[SystemMessage(content=self.prompt),
                              HumanMessage(content=user)]],
                            max_tokens=self.max_tokens)
        return _content(resp[0]) if resp else "NOT_FOUND"


# ── routers (plain callables — write your own in the shell) ──────────────────

def open_all_router(query, votes, neurons) -> list:
    """Open every neuron. Correct and expensive; the honest default for small N."""
    return list(range(len(neurons)))


def top_k_router(k: int = 2, threshold: int = 6, always_argmax: bool = True) -> Callable:
    """The previous hardcoded behavior, now an explicit, replaceable choice."""
    def route(query, votes, neurons) -> list:
        order = sorted(range(len(neurons)), key=lambda i: votes[i]["score"], reverse=True)
        chosen = {i for i in order[:k] if votes[i]["score"] > 0}
        chosen |= {i for i in range(len(neurons)) if votes[i]["score"] >= threshold}
        if always_argmax and order:
            chosen.add(order[0])
        return sorted(chosen)
    return route


def threshold_router(threshold: int = 6) -> Callable:
    def route(query, votes, neurons) -> list:
        return [i for i in range(len(neurons)) if votes[i]["score"] >= threshold]
    return route


# ── the composable brain ─────────────────────────────────────────────────────

@dataclass
class Brain:
    """N neurons + 1 synthesizer + a router. Callable, and usable AS a neuron.

    `neurons` may contain Neurons or other Brains — that is the hierarchy, and
    it is built by the agent in the shell, not inferred from a directory.
    """
    neurons: list = field(default_factory=list)
    synthesizer: Synthesizer = field(default_factory=Synthesizer)
    name: str = "brain"
    router: Callable = field(default=open_all_router)
    digest: Optional[Content] = None      # this brain's neuron-face for a parent
    trace: list = field(default_factory=list)

    # ── as a NEURON (what a parent scores) ──
    def cognize_msgs(self, query: str) -> list:
        face = self.digest if self.digest is not None else f"brain '{self.name}' with {len(self.neurons)} neurons"
        return _messages(face, query, BRAIN_COGNIZE_SYSTEM)

    async def cognize(self, query: str) -> dict:
        resp = await _batch([self.cognize_msgs(query)], max_tokens=700)
        return _score(_content(resp[0])) if resp else {"score": 0, "reasoning": "no response"}

    async def instruct(self, query: str, **kw) -> str:
        return await self.query(query, **kw)

    # ── as a BRAIN ──
    async def vote(self, query: str) -> list:
        """Stage 1: every neuron scores in ONE batch. Returns the raw votes so
        the agent can route on them itself."""
        if not self.neurons:
            return []
        with trace.span("vote", self.name, n=len(self.neurons)) as sp:
            resps = await _batch([n.cognize_msgs(query) for n in self.neurons],
                                 max_tokens=700)
            votes = [_score(_content(r)) for r in resps]
            failed = sum(1 for v in votes if v.get("parse_failed"))
            sp.set(scores=[v["score"] for v in votes],
                   **({"parse_failed": failed} if failed else {}))
            return votes

    async def query(self, query: str, *, router: Optional[Callable] = None,
                    max_tokens: int = 3000) -> str:
      with trace.span("brain", self.name, n=len(self.neurons)) as _sp:
          if not self.neurons:
              return f"NOT_FOUND (brain {self.name}: empty)"
          votes = await self.vote(query)
          chosen = (router or self.router)(query, votes, self.neurons)
          _sp.set(opened=len(list(chosen)))
          self.trace = [{"name": getattr(n, "name", str(i)), "score": votes[i]["score"],
                         "opened": i in set(chosen)} for i, n in enumerate(self.neurons)]
          picked = [self.neurons[i] for i in chosen]
          if not picked:
              return f"NOT_FOUND (brain {self.name}: router opened nothing)"
          flat = [n for n in picked if isinstance(n, Neuron)]
          deep = [n for n in picked if not isinstance(n, Neuron)]
          flat_out, deep_out = await asyncio.gather(
              _batch([n.instruct_msgs(query) for n in flat], max_tokens=max_tokens),
              asyncio.gather(*(b.instruct(query) for b in deep)),
          )
          parts = [(n.name, _content(r)) for n, r in zip(flat, flat_out)]
          parts += [(getattr(b, "name", "sub"), o) for b, o in zip(deep, deep_out)]
          parts = [(n, t) for n, t in parts if "NOT_FOUND" not in t[:200]]
          if not parts:
              return f"NOT_FOUND (brain {self.name}: opened neurons had no answer)"
          return await self.synthesizer(query, parts)

    async def __call__(self, query: str, **kw) -> str:
        return await self.query(query, **kw)


# ── the raw fan-out primitive ────────────────────────────────────────────────

async def fanout(chunks: Sequence[Content], query: str, *,
                 prompt: str = INSTRUCT_SYSTEM, max_tokens: int = 3000) -> list:
    """N neurons, one batch, raw list of strings back. The thing to reach for
    when you want N sub-calls and will fold them yourself."""
    if not chunks:
        return []
    # A list of Neurons is the natural thing to fan out over, and passing them
    # here used to fall through to str-interpolation of the dataclass repr —
    # it "worked" while silently discarding each neuron's own lens. Dispatch
    # properly instead: a Neuron brings its own prompt, anything else uses the
    # shared one.
    msgs = [c.instruct_msgs(query) if isinstance(c, Neuron)
            else _messages(c, query, prompt) for c in chunks]
    with trace.span("fanout", f"{len(chunks)} chunks", n=len(chunks)):
        resps = await _batch(msgs, max_tokens=max_tokens)
        return [_content(r) for r in resps]


async def sub_llm(prompt: str, content: Content = "", *, max_tokens: int = 3000) -> str:
    """One sub-call. The simplest recursion step."""
    with trace.span("sub_llm", prompt[:60]):
        out = await fanout([content], prompt, prompt=INSTRUCT_SYSTEM,
                           max_tokens=max_tokens)
        return out[0] if out else ""


# ── adapters ─────────────────────────────────────────────────────────────────

def from_dir(root: Union[str, Path], *, router: Optional[Callable] = None,
             include: Optional[Callable] = None, name: Optional[str] = None) -> Brain:
    """Build a Brain out of a directory tree — the old behavior, now an explicit
    constructor the agent calls when it wants it. Sub-dirs become sub-Brains;
    a `_digest.md` (if present) becomes that Brain's neuron-face."""
    from .hierarchical import DIGEST_NAME, _should_include_file
    root = Path(root)
    keep = include or (lambda p: _should_include_file(str(p)))
    kids: list = []
    for p in sorted(root.iterdir()):
        if p.name.startswith((".", "_")):
            continue
        if p.is_dir():
            kids.append(from_dir(p, router=router, include=include))
        elif p.is_file() and keep(p):
            kids.append(Neuron(content=p, name=p.name))
    dig = root / DIGEST_NAME
    return Brain(neurons=kids, name=name or root.name, digest=dig if dig.exists() else None,
                 router=router or top_k_router())
