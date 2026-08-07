"""RLM — Recursive Language Model over an unbounded, GROWING corpus, on HEAVEN.

The class the whole stack was building toward. An RLM replaces the context
window with a filesystem pyramid: the model only ever sees O(node) tokens per
call while the corpus is unbounded. Every LLM call goes through the heaven
neuron-inference primitives in hierarchical.py (UnifiedChat + HeavenAgentConfig)
— no raw provider SDK.

  INGEST — live session folding: messages → iterations → phase dirs.
           The iteration boundary rule: a user message with real text STARTS
           a new iteration (same rule as the session folders everywhere else).
  INDEX  — incremental digest pyramid: only DIRTY leaf dirs re-fold (one heaven
           fold each); every ancestor concatenates child digests verbatim
           (fold-once-concatenate-above), so reindex cost is O(changed).
  QUERY  — hierarchical Brain descent (witnessed synthesis).
  JUDGE  — exhaustive part × rule incidence sweep (coverage by construction),
           each cell a heaven neuron call with a machine-verified verbatim witness.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import hierarchical as hbrain
from .hierarchical import (Brain, Usage, USAGE_VAR, TRACE_VAR, build_digests,
                           get_trace, _neuron_messages, _batch, _content, _score)

DIGEST = hbrain.DIGEST_NAME


@dataclass
class RLMResult:
    answer: str
    refs: list          # descent trace — which branches/files opened
    usage: dict


def _usage_dict(u: Usage) -> dict:
    return {"calls": u.calls, "model": hbrain.MODEL}


JUDGE_SYSTEM = (
    "You are a JudgeNeuron. Judge your neuron content (one PART of a larger "
    "system) against the RULE in the user message. Respond with a JSON object, "
    "keys: 'verdict' ('complies' | 'violates' | 'not_applicable'), 'score' "
    "(0-10), 'witness' (a VERBATIM quote from your content evidencing the "
    "verdict; empty ONLY for not_applicable), 'reasoning'. No witness, no "
    "verdict. Never invent or paraphrase quotes.")


class RLM:
    """One RLM = one corpus root. Ingest forever; query any time."""

    def __init__(self, root, session: Optional[str] = None, phase_size: int = 8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = session or time.strftime("session_%Y%m%d_%H%M%S")
        self.phase_size = phase_size
        self._cur: Optional[dict] = None      # the open iteration
        self._dirty: set = set()              # leaf dirs needing a re-fold
        self._n = self._existing_iterations()

    # ── paths ────────────────────────────────────────────────────────────────

    @property
    def session_dir(self) -> Path:
        return self.root / self.session

    def _phase_dir(self, iter_num: int) -> Path:
        return self.session_dir / f"phase_{iter_num // self.phase_size:02d}"

    def _existing_iterations(self) -> int:
        if not self.session_dir.is_dir():
            return 0
        return len(list(self.session_dir.glob("phase_*/iter_*.md")))

    # ── INGEST ───────────────────────────────────────────────────────────────

    def ingest_message(self, role: str, text: str,
                       tools: Optional[list] = None) -> None:
        """A user message with real text starts a new iteration (the boundary
        rule); everything else accretes onto the open one. Iterations flush to
        disk on boundary or flush()."""
        text = text or ""
        if role == "user" and text.strip() and not text.startswith("<command"):
            self._flush_current()
            self._cur = {"user": text, "agent": [], "tools": list(tools or [])}
        elif self._cur is not None:
            if text.strip():
                self._cur["agent"].append(text)
            self._cur["tools"].extend(tools or [])

    def ingest_iteration(self, user: str, agent: str = "",
                         tools: Optional[list] = None) -> Optional[Path]:
        self._flush_current()
        self._cur = {"user": user, "agent": [agent] if agent else [],
                     "tools": list(tools or [])}
        return self._flush_current()

    def flush(self) -> Optional[Path]:
        return self._flush_current()

    def _flush_current(self) -> Optional[Path]:
        if self._cur is None:
            return None
        it, self._cur = self._cur, None
        phase = self._phase_dir(self._n)
        phase.mkdir(parents=True, exist_ok=True)
        # NO truncation, ever — oversized neurons are the chunker's problem.
        body = (f"# Iteration {self._n}\n\n## User\n{it['user']}\n\n"
                f"## Agent\n" + "\n\n".join(it["agent"]) +
                f"\n\n## Tools used\n{', '.join(it['tools']) or 'none'}\n")
        path = phase / f"iter_{self._n:03d}.md"
        path.write_text(body)
        self._n += 1
        self._dirty.add(phase)
        return path

    # ── INDEX (incremental) ─────────────────────────────────────────────────

    async def reindex(self, force: bool = False) -> dict:
        """Re-fold ONLY dirty leaf dirs (one heaven fold each), then refresh the
        concatenation digests up the ancestry. O(changed), not O(corpus)."""
        self._flush_current()
        u = Usage()
        USAGE_VAR.set(u)
        if force:
            await build_digests(self.root, force=True)
            self._dirty.clear()
            return {"folded": "all", "usage": _usage_dict(u)}
        folded = []
        for leaf in sorted(self._dirty):
            await build_digests(leaf, force=True)   # leaf has only files → one fold
            folded.append(str(leaf))
        self._dirty.clear()
        for anc in (self.session_dir, self.root):
            kids = [Brain(p) for p in sorted(anc.iterdir())
                    if p.is_dir() and not p.name.startswith((".", "_"))]
            if kids and all(k.digest_path.exists() for k in kids):
                (anc / DIGEST).write_text("\n\n".join(
                    f"### sub-brain: {k.name}\n{k.digest_path.read_text(errors='replace')}"
                    for k in kids))
        return {"folded": folded, "usage": _usage_dict(u)}

    # ── QUERY ────────────────────────────────────────────────────────────────

    async def query(self, goal: str) -> RLMResult:
        if self._dirty or self._cur is not None:
            await self.reindex()
        u = Usage()
        USAGE_VAR.set(u)
        TRACE_VAR.set([])
        answer = await Brain(self.root).query(goal)
        return RLMResult(answer=answer, refs=list(get_trace()), usage=_usage_dict(u))

    # ── JUDGE (exhaustive — every leaf gets a cell) ─────────────────────────

    async def judge(self, rule: str, parts: Optional[list] = None) -> dict:
        if self._dirty or self._cur is not None:
            await self.reindex()
        u = Usage()
        USAGE_VAR.set(u)
        targets = list(parts) if parts else sorted(self.root.rglob("iter_*.md"))
        # one heaven neuron call per part (content rendered via path= block)
        responses = await _batch(
            [_neuron_messages(str(p), rule, JUDGE_SYSTEM) for p in targets],
            max_tokens=2500)

        cells = []
        for p, resp in zip(targets, responses):
            content = p.read_text(errors="replace")   # for witness verification
            d = _first_json(_content(resp)) or {}
            verdict = d.get("verdict", "not_applicable")
            witness = str(d.get("witness", ""))
            if verdict not in ("complies", "violates", "not_applicable"):
                verdict = "not_applicable"
            if verdict != "not_applicable" and not witness.strip():
                verdict = "not_applicable"
            # d is {} when the reply had no parseable JSON — record that as a
            # parse failure, not as a finding of not_applicable.
            cells.append({"id": str(p.relative_to(self.root)), "verdict": verdict,
                          "parse_failed": not d,
                          "score": _score(_content(resp))["score"],
                          "witness": witness,
                          "witness_verified": bool(witness) and witness in content,
                          "reasoning": str(d.get("reasoning", ""))})
        verdicts = [c["verdict"] for c in cells]
        return {"rule": rule, "cells": cells,
                "summary": {"parts": len(cells),
                            "complies": verdicts.count("complies"),
                            "violates": verdicts.count("violates"),
                            "not_applicable": verdicts.count("not_applicable"),
                            "global_section": verdicts.count("violates") == 0},
                "usage": _usage_dict(u)}


# ── small parser ──────────────────────────────────────────────────────────────

def _first_json(raw: str):
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    import json
                    return json.loads(raw[start:i + 1])
                except Exception:
                    return None
    return None
