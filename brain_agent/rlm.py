"""RLM — Recursive Language Model over an unbounded, GROWING corpus.

The class the whole stack was building toward. An RLM replaces the context
window with a filesystem pyramid: the model only ever sees O(node) tokens per
call while the corpus is unbounded.

  INGEST — live session folding: messages → iterations → phase dirs.
           The iteration boundary rule: a user message with real text STARTS
           a new iteration (same rule as the session folders everywhere else).
  INDEX  — incremental digest pyramid: only DIRTY leaf dirs re-fold (one LLM
           call each); every ancestor concatenates child digests verbatim
           (fold-once-concatenate-above), so reindex cost is O(changed), not
           O(corpus).
  QUERY  — hierarchical Brain descent (witnessed synthesis).
  JUDGE  — exhaustive part × rule incidence sweep (coverage by construction).

Layout it maintains:
  root/
    session_<id>/phase_NN/iter_NNN.md     (leaf neurons)
    session_<id>/phase_NN/_digest.md      (folded once, on reindex)
    session_<id>/_digest.md               (concatenation)
    _digest.md                            (concatenation)
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import hierarchical as hbrain
from .hierarchical import (Brain, FileNeuron, Usage, USAGE_VAR, TRACE_VAR,
                    build_digests, get_trace, get_usage, _llm)

DIGEST = hbrain.DIGEST_NAME


@dataclass
class RLMResult:
    answer: str
    refs: list[str]          # descent trace — which branches/files were opened
    usage: dict


def _usage_dict(u: Usage) -> dict:
    return {"calls": u.calls, "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens, "model": hbrain.MODEL}


class RLM:
    """One RLM = one corpus root. Ingest forever; query any time."""

    def __init__(self, root: str | Path, session: Optional[str] = None,
                 phase_size: int = 8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = session or time.strftime("session_%Y%m%d_%H%M%S")
        self.phase_size = phase_size
        self._cur: Optional[dict] = None      # the open iteration
        self._dirty: set[Path] = set()        # leaf dirs needing a re-fold
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
                       tools: Optional[list[str]] = None) -> None:
        """Feed messages as they happen. A user message with real text starts
        a new iteration (the boundary rule); everything else accretes onto the
        open one. Iterations flush to disk on boundary or flush()."""
        text = text or ""
        if role == "user" and text.strip() and not text.startswith("<command"):
            self._flush_current()
            self._cur = {"user": text, "agent": [], "tools": list(tools or [])}
        elif self._cur is not None:
            if text.strip():
                self._cur["agent"].append(text)
            self._cur["tools"].extend(tools or [])

    def ingest_iteration(self, user: str, agent: str = "",
                         tools: Optional[list[str]] = None) -> Path:
        """Ingest a whole iteration at once (batch path)."""
        self._flush_current()
        self._cur = {"user": user, "agent": [agent] if agent else [],
                     "tools": list(tools or [])}
        return self._flush_current()

    def flush(self) -> Optional[Path]:
        """Close the open iteration (end of turn / end of session)."""
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
        """Re-fold ONLY dirty leaf dirs (one LLM call each), then refresh the
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
        # ancestors concatenate verbatim — no LLM, just string assembly
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
        return RLMResult(answer=answer, refs=list(get_trace()),
                         usage=_usage_dict(u))

    # ── JUDGE (exhaustive — every leaf gets a cell) ─────────────────────────

    JUDGE_SYSTEM = (
        "You are a JudgeNeuron. Judge the relationship between your neuron "
        "content (one PART of a larger system) and the RULE in the user "
        "message. Respond with a JSON object, keys: 'verdict' ('complies' | "
        "'violates' | 'not_applicable'), 'score' (0-10), 'witness' (a VERBATIM "
        "quote evidencing the verdict; empty ONLY for not_applicable), "
        "'reasoning'. No witness, no verdict. Never invent or paraphrase "
        "quotes.\n\n<neuron content>\n{content}\n</neuron content>")

    async def judge(self, rule: str,
                    parts: Optional[list[Path]] = None) -> dict:
        if self._dirty or self._cur is not None:
            await self.reindex()
        u = Usage()
        USAGE_VAR.set(u)
        targets = parts or sorted(p for p in self.root.rglob("iter_*.md"))

        async def one(p: Path) -> dict:
            content = p.read_text(errors="replace")
            raw = await _llm(self.JUDGE_SYSTEM.format(content=content),
                             f"RULE: {rule}", 2500)
            d = _first_json(raw) or {}
            verdict = d.get("verdict", "not_applicable")
            witness = str(d.get("witness", ""))
            if verdict not in ("complies", "violates", "not_applicable"):
                verdict = "not_applicable"
            if verdict != "not_applicable" and not witness.strip():
                verdict = "not_applicable"
            return {"id": str(p.relative_to(self.root)), "verdict": verdict,
                    "score": _int(d.get("score", 0)),
                    "witness": witness,
                    "witness_verified": bool(witness) and witness in content,
                    "reasoning": str(d.get("reasoning", ""))}

        cells = await asyncio.gather(*(one(p) for p in targets))
        verdicts = [c["verdict"] for c in cells]
        return {"rule": rule, "cells": list(cells),
                "summary": {"parts": len(cells),
                            "complies": verdicts.count("complies"),
                            "violates": verdicts.count("violates"),
                            "not_applicable": verdicts.count("not_applicable"),
                            "global_section": verdicts.count("violates") == 0},
                "usage": _usage_dict(u)}


# ── small parsers ─────────────────────────────────────────────────────────────

def _int(x, lo=0, hi=10) -> int:
    try:
        return max(lo, min(hi, int(x)))
    except (ValueError, TypeError):
        return 0


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
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
