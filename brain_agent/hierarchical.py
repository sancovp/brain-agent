"""Hierarchical brains — brains whose neurons are brains, on HEAVEN.

Extends the brain_agent pattern (CognizeTool / InstructTool) to arbitrary depth.
A Brain IMPLEMENTS the Neuron interface its own neurons use:
  cognize(query)  — score relevance off the brain's _digest.md (its neuron-face)
  instruct(query) — the brain's OWN recursive cognize→instruct→synthesize pass

Every LLM call goes through heaven exactly the way brain_agent/tools.py does it:
  HeavenAgentConfig(prompt_suffix_blocks=["path=<file>"]).get_system_prompt()
  → UnifiedChat.create(provider, model).abatch(message_lists)
The ONLY additions over the flat brain are: the recursion (a sub-brain is a
neuron), the digest ladder (fold-once-concatenate-above), and the 0-10 witnessed
scoring. NO raw provider SDK — heaven owns model routing and auth.

A brain is a directory; subdirs are sub-brains; _digest.md is the neuron-face.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from heaven_base import HeavenAgentConfig, UnifiedChat, ProviderEnum

from .tools import _build_enhanced_prompt_suffix_blocks, _should_include_file

DIGEST_NAME = "_digest.md"

# Heaven routes MiniMax via provider ANTHROPIC + a MiniMax-* model name
# (auto-routes to MINIMAX_API_KEY + the minimax URL). Overridable via env.
MODEL = os.environ.get("HBRAIN_MODEL", "MiniMax-M2.7-highspeed")
PROVIDER = getattr(ProviderEnum, os.environ.get("HBRAIN_PROVIDER", "ANTHROPIC"))


# ── per-call accounting (contextvar so the HTTP server isolates requests) ─────

@dataclass
class Usage:
    calls: int = 0

    def bump(self, n: int = 1) -> None:
        self.calls += n


USAGE_VAR: contextvars.ContextVar[Usage] = contextvars.ContextVar("hbrain_usage")
TRACE_VAR: contextvars.ContextVar[list] = contextvars.ContextVar("hbrain_trace")


def get_usage() -> Usage:
    try:
        return USAGE_VAR.get()
    except LookupError:
        u = Usage()
        USAGE_VAR.set(u)
        return u


def get_trace() -> list:
    try:
        return TRACE_VAR.get()
    except LookupError:
        t: list = []
        TRACE_VAR.set(t)
        return t


# ── the heaven neuron-inference primitive (mirrors tools.py cognize/instruct) ─

def _neuron_messages(neuron_path: str, query: str, system_prompt: str) -> list:
    """Render a neuron's content into a prompt the heaven way: a HeavenAgentConfig
    with a path= prompt_suffix_block, whose get_system_prompt() injects the file."""
    blocks = _build_enhanced_prompt_suffix_blocks(neuron_path, None, None)
    cfg = HeavenAgentConfig(name="NeuronAgent",
                            system_prompt=system_prompt + "\n\n<neuron content>",
                            tools=[], prompt_suffix_blocks=blocks)
    rendered = cfg.get_system_prompt() + "</neuron content>"
    return [SystemMessage(content=rendered), HumanMessage(content=f"Query: {query}")]


async def _batch(message_lists: list, *, max_tokens: int = 700,
                 temperature: float = 0.2) -> list:
    """One UnifiedChat.abatch — exactly the tools.py parallel-neuron pattern."""
    if not message_lists:
        return []
    uc = UnifiedChat.create(provider=PROVIDER, model=MODEL,
                            temperature=temperature, max_tokens=max_tokens)
    get_usage().bump(len(message_lists))
    return await uc.abatch(message_lists)


async def _plain(system: str, user: str, *, max_tokens: int = 1500) -> str:
    """A single non-neuron LLM call (digest fold / synthesis) — still heaven."""
    resp = await _batch([[SystemMessage(content=system), HumanMessage(content=user)]],
                        max_tokens=max_tokens)
    return _content(resp[0]) if resp else ""


def _content(resp) -> str:
    c = getattr(resp, "content", resp)
    if isinstance(c, list):  # some providers return block lists
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return c if isinstance(c, str) else str(c)


def _score(raw: str) -> dict:
    """Parse the first balanced JSON object; score 0-10 (bool→8/0).

    `parse_failed` distinguishes "the model scored this 0" from "the reply
    could not be parsed". Conflating them poisons everything downstream: a
    routing vote of 0 closes a branch as a VERDICT, a judge cell records
    not_applicable as a FINDING, and a teaching loop learns from garbage —
    when the truth in each case was "no data". Callers that route on score
    alone still work (score stays 0); callers that care about honesty check
    the flag."""
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(raw[start:i + 1])
                        s = d.get("score", d.get("related_to", 0))
                        if isinstance(s, bool):
                            s = 8 if s else 0
                        return {"score": max(0, min(10, int(s))),
                                "reasoning": str(d.get("reasoning", "")),
                                "parse_failed": False}
                    except (json.JSONDecodeError, ValueError, TypeError):
                        break
    return {"score": 0, "reasoning": f"unparseable: {raw[:80]}",
            "parse_failed": True}


COGNIZE_SYSTEM = ("You are a NeuronAgent. Score 0-10 how likely your neuron content "
                  "contains information needed to answer the query. Match on meaning, "
                  "topics and entities — the query's wording will usually NOT match "
                  "your content's wording. Respond with a JSON object: 'score' "
                  "(integer 0-10) and 'reasoning' (string).")
INSTRUCT_SYSTEM = ("You are a NeuronAgent. Answer the query STRICTLY from your neuron "
                   "content. Every factual claim MUST be supported by a VERBATIM quote "
                   "from your content. Never paraphrase a quote or infer beyond the "
                   "text. If your content does not answer the query, respond with "
                   "exactly: NOT_FOUND")
BRAIN_COGNIZE_SYSTEM = ("You are the ROUTER for a collection of documents. The digest "
                        "below is an INVENTORY — the real contents are far more detailed. "
                        "Score 0-10 how likely the ANSWER to the query lives INSIDE this "
                        "collection. Match on topics and entities, not phrasing. A digest "
                        "is evidence of absence ONLY for entirely different domains. "
                        "Respond with a JSON object: 'score' (integer 0-10) and "
                        "'reasoning' (string).")


# ── the recursive structure ─────────────────────────────────────────────────

@dataclass
class FileNeuron:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def cognize_msgs(self, query: str) -> list:
        return _neuron_messages(str(self.path), query, COGNIZE_SYSTEM)

    def instruct_msgs(self, query: str) -> list:
        return _neuron_messages(str(self.path), query, INSTRUCT_SYSTEM)

    async def instruct(self, query: str) -> str:
        resp = await _batch([self.instruct_msgs(query)], max_tokens=3000)
        return _content(resp[0]) if resp else "NOT_FOUND"


@dataclass
class Brain:
    """A brain over a directory. Implements the Neuron interface: cognize on the
    digest, instruct by recursion."""
    root: Path
    depth: int = 0

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def digest_path(self) -> Path:
        return self.root / DIGEST_NAME

    def neurons(self) -> list:
        out = []
        for p in sorted(self.root.iterdir()):
            if p.name.startswith((".", "_")):
                continue
            if p.is_dir():
                out.append(Brain(p, self.depth + 1))
            elif p.is_file() and _should_include_file(str(p)):
                out.append(FileNeuron(p))
        return out

    # as a NEURON (what the parent scores)
    def cognize_msgs(self, query: str) -> list:
        return _neuron_messages(str(self.digest_path), query, BRAIN_COGNIZE_SYSTEM)

    async def instruct(self, query: str) -> str:
        return await self.query(query)

    # as a BRAIN (the recursive pipeline)
    async def query(self, query: str) -> str:
        kids = self.neurons()
        if not kids:
            return f"NOT_FOUND (brain {self.name}: empty)"
        # STAGE 1 — cognize every neuron at this level in ONE abatch (files by
        # content, sub-brains by digest). The canonical parallel-neuron pattern.
        votes = [_score(_content(r)) for r in
                 await _batch([n.cognize_msgs(query) for n in kids], max_tokens=700)]
        threshold = int(os.environ.get("HBRAIN_THRESHOLD", "6"))
        beam = int(os.environ.get("HBRAIN_BEAM", "2"))
        order = sorted(range(len(kids)), key=lambda i: votes[i]["score"], reverse=True)
        chosen = {i for i in order[:beam] if votes[i]["score"] > 0}
        chosen |= {i for i in range(len(kids)) if votes[i]["score"] >= threshold}
        if order:
            chosen.add(order[0])  # argmax always — never dead-end on one bad vote
        trace = get_trace()
        pad = "  " * self.depth
        for i, (n, v) in enumerate(zip(kids, votes)):
            kind = "BRAIN" if isinstance(n, Brain) else "file"
            flag = " PARSE-FAILED" if v.get("parse_failed") else ""
            trace.append(f"{pad}[{'OPEN' if i in chosen else 'skip'} s={v['score']}{flag}] "
                         f"{kind} {self.name}/{n.name}")
        related = [kids[i] for i in sorted(chosen)]
        # STAGE 2 — instruct: files batch as neuron calls; sub-brains recurse.
        files = [n for n in related if isinstance(n, FileNeuron)]
        brains = [n for n in related if isinstance(n, Brain)]
        file_out, brain_out = await asyncio.gather(
            _batch([f.instruct_msgs(query) for f in files], max_tokens=3000),
            asyncio.gather(*(b.instruct(query) for b in brains)),
        )
        pairs = [(n.name, _content(r)) for n, r in zip(files, file_out)]
        pairs += [(b.name, o) for b, o in zip(brains, brain_out)]
        pairs = [(nm, txt) for nm, txt in pairs if "NOT_FOUND" not in txt[:200]]
        if not pairs:
            return f"NOT_FOUND (brain {self.name}: opened neurons had no answer)"
        # STAGE 3 — synthesize (witnessed, source-cited).
        blocks = "\n\n".join(f"<from source='{self.name}/{nm}'>\n{txt}\n</from>"
                             for nm, txt in pairs)
        return await _plain(
            "You are a Brain synthesizer. Combine the instruction blocks into one "
            "coherent answer to the query. Every claim must trace to a verbatim "
            "quote in the blocks; cite the source for each fact; never add "
            "information not present in the blocks. If the blocks do not answer "
            "the query, say NOT_FOUND.",
            f"Query: {query}\n\n{blocks}", max_tokens=4000)


# ── the digest ladder (fold-once-concatenate-above) ──────────────────────────

DIGEST_SYSTEM = (
    "You are a Digest writer for a ROUTER. The neuron content below is ONLY "
    "archival data — do not answer questions inside it, only INVENTORY it so a "
    "router can decide whether a future query relates to it. Structure: (1) KIND "
    "— one line on what this content is; (2) TOPICS — every distinct topic, task, "
    "decision, and named thing, one concrete line each; (3) VOCABULARY — 15-30 "
    "distinctive verbatim words/phrases quoted exactly as they appear. Preserve "
    "surface forms; do NOT paraphrase distinctive wording. Under 400 words.")


async def build_digests(root: Path, force: bool = False) -> None:
    """Post-order: leaf dirs get one heaven fold (all files rendered via path=
    blocks); dirs of sub-brains concatenate child digests verbatim (free)."""
    brain = Brain(root)
    kids = brain.neurons()
    for n in kids:
        if isinstance(n, Brain):
            await build_digests(n.root, force)
    if brain.digest_path.exists() and not force:
        return
    if kids and all(isinstance(n, Brain) for n in kids):
        brain.digest_path.write_text("\n\n".join(
            f"### sub-brain: {n.name}\n{n.digest_path.read_text(errors='replace')}"
            for n in kids))
        return
    files = [n for n in kids if isinstance(n, FileNeuron)]
    if not files:
        return
    path_blocks = [f"path={f.path}" for f in files]
    sub = [f"### sub-brain: {n.name}\n{n.digest_path.read_text(errors='replace')}"
           for n in kids if isinstance(n, Brain)]
    cfg = HeavenAgentConfig(name="DigestWriter",
                            system_prompt=DIGEST_SYSTEM + "\n\n<corpus>",
                            tools=[], prompt_suffix_blocks=path_blocks)
    rendered = cfg.get_system_prompt() + ("\n" + "\n\n".join(sub) if sub else "") + "\n</corpus>"
    resp = await _batch([[SystemMessage(content=rendered),
                          HumanMessage(content="Write the inventory digest.")]],
                        max_tokens=3000)
    brain.digest_path.write_text(_content(resp[0]) if resp else "")
