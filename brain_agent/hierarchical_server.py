"""brain-agent HTTP server — judge and fill for coordinate/configuration engines.

The contract:
  The CALLER owns addressing — it knows which parts/cells exist and which
  slots are empty (zero-able). This server owns two verbs against them:
    /judge — one neuron per part: part x rule -> verdict + VERBATIM witness
             (exhaustive: every part sent gets judged, none skipped)
    /fill  — generative: propose completions for an empty spectrum slot,
             optionally GROUNDED in a hierarchical brain corpus
  Plus the brain primitives: /brains/build, /brains/query, /neuron/*.

Stateless per call (digests persist on disk next to the corpus).

Run:  brain-agent-http            (console script)
      HBRAIN_PORT=8177 python -m brain_agent.hierarchical_server
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import hierarchical as hbrain
from .hierarchical import (Brain, FileNeuron, Usage, USAGE_VAR, TRACE_VAR,
                    get_trace, get_usage, _llm, _parse_vote, build_digests)

app = FastAPI(title="hbrain", version="0.1.0")


def _fresh() -> None:
    USAGE_VAR.set(Usage())
    TRACE_VAR.set([])


def _usage() -> dict:
    u = get_usage()
    return {"calls": u.calls, "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens, "model": hbrain.MODEL}


# ── JUDGE mode: part × rule → verdict + verbatim witness ─────────────────────

JUDGE_SYSTEM = (
    "You are a JudgeNeuron. Judge the relationship between your neuron content "
    "(one PART of a larger system) and the RULE in the user message.\n"
    "Respond with a JSON object, keys:\n"
    "  'verdict': one of 'complies' | 'violates' | 'not_applicable'\n"
    "  'score': integer 0-10 confidence in the verdict\n"
    "  'witness': a VERBATIM quote from your neuron content that evidences the "
    "verdict (empty string ONLY for not_applicable)\n"
    "  'reasoning': one or two sentences\n"
    "RULES OF JUDGMENT: a verdict of complies/violates REQUIRES a verbatim "
    "witness quote — no witness, no verdict (use not_applicable). Never invent "
    "or paraphrase quotes. Judge ONLY against the given rule, not general "
    "quality.\n\n<neuron content>\n{content}\n</neuron content>")


def _parse_judgment(raw: str) -> dict:
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
                        verdict = str(d.get("verdict", "not_applicable"))
                        if verdict not in ("complies", "violates", "not_applicable"):
                            verdict = "not_applicable"
                        witness = str(d.get("witness", ""))
                        if verdict != "not_applicable" and not witness.strip():
                            verdict = "not_applicable"  # no witness, no verdict
                        return {"verdict": verdict,
                                "score": max(0, min(10, int(d.get("score", 0)))),
                                "witness": witness,
                                "reasoning": str(d.get("reasoning", ""))}
                    except (json.JSONDecodeError, ValueError, TypeError):
                        break
    return {"verdict": "not_applicable", "score": 0, "witness": "",
            "reasoning": f"unparseable judgment: {raw[:100]}"}


class Part(BaseModel):
    id: str
    path: Optional[str] = None      # read content from disk
    content: Optional[str] = None   # or inline


class JudgeReq(BaseModel):
    parts: list[Part]
    rule: str


@app.post("/judge")
async def judge(req: JudgeReq) -> dict:
    """Exhaustive judge mode: EVERY part judged against the rule. No descent,
    no skipping — coverage by construction. Returns one row per part
    (the incidence-matrix row for this rule)."""
    _fresh()

    async def one(p: Part) -> dict:
        content = p.content if p.content is not None else Path(p.path).read_text(errors="replace")
        raw = await _llm(JUDGE_SYSTEM.format(content=content),
                         f"RULE: {req.rule}", 2500)
        j = _parse_judgment(raw)
        # witness must actually appear in the content — checkable, not vibes
        if j["witness"] and j["witness"] not in content:
            j["witness_verified"] = False
        else:
            j["witness_verified"] = bool(j["witness"])
        return {"id": p.id, **j}

    cells = await asyncio.gather(*(one(p) for p in req.parts))
    verdicts = [c["verdict"] for c in cells]
    return {
        "rule": req.rule,
        "cells": list(cells),
        "summary": {
            "parts": len(cells),
            "complies": verdicts.count("complies"),
            "violates": verdicts.count("violates"),
            "not_applicable": verdicts.count("not_applicable"),
            # the global-section indicator (CB side: kernelSanctuaryDegree lane)
            "global_section": verdicts.count("violates") == 0,
        },
        "usage": _usage(),
    }


# ── FILL mode: generative completion for a zero/empty spectrum slot ─────────

FILL_SYSTEM = (
    "You are a SpectrumFiller for a coordinate engine. A node's children ARE "
    "its spectrum — the set of choices at that slot. A spectrum needs at least "
    "a high and a low, and its members must be mutually exclusive alternatives "
    "at the SAME level of abstraction as the existing siblings.\n"
    "Respond with a JSON object, key 'candidates': a list of objects with keys "
    "'label' (short, concrete), 'rationale' (one sentence), 'confidence' "
    "(0-10 integer). Propose exactly the requested number. No prose outside "
    "the JSON.")


class FillReq(BaseModel):
    slot_label: str                      # the node whose spectrum needs filling
    parent_label: Optional[str] = None
    siblings: list[str] = []             # existing children (may be empty)
    space_context: Optional[str] = None  # scry readout / kernel context from CB
    n: int = 3
    brain_root: Optional[str] = None     # ground the fill in a brain corpus


@app.post("/fill")
async def fill(req: FillReq) -> dict:
    """Generative fill for a Born-0 / empty slot. If brain_root is given, a
    brain query runs first and its witnessed synthesis grounds the proposals."""
    _fresh()
    grounding = ""
    if req.brain_root:
        root = Path(req.brain_root)
        if not root.is_dir():
            raise HTTPException(400, f"brain_root not found: {root}")
        q = (f"What is known about '{req.slot_label}'"
             + (f" in the context of '{req.parent_label}'" if req.parent_label else "")
             + "? Collect concrete facts, names, and distinctions useful for "
               "enumerating its variants.")
        grounding = await Brain(root).query(q)

    user = (f"SLOT: {req.slot_label}\n"
            + (f"PARENT: {req.parent_label}\n" if req.parent_label else "")
            + (f"EXISTING SIBLINGS: {', '.join(req.siblings)}\n" if req.siblings else "")
            + (f"SPACE CONTEXT:\n{req.space_context}\n" if req.space_context else "")
            + (f"GROUNDING (witnessed synthesis from the brain corpus):\n{grounding}\n"
               if grounding and "NOT_FOUND" not in grounding[:60] else "")
            + f"\nPropose {req.n} candidates for this spectrum.")
    raw = await _llm(FILL_SYSTEM, user, 3000)

    start = raw.find("{")
    candidates: list[dict] = []
    if start != -1:
        try:
            end = raw.rindex("}")
            data = json.loads(raw[start:end + 1])
            for c in data.get("candidates", [])[:req.n]:
                candidates.append({
                    "label": str(c.get("label", ""))[:120],
                    "rationale": str(c.get("rationale", "")),
                    "confidence": max(0, min(10, int(c.get("confidence", 0)))),
                })
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return {"slot": req.slot_label, "candidates": candidates,
            "grounded": bool(grounding and "NOT_FOUND" not in grounding[:60]),
            "grounding": grounding if grounding else None,
            "usage": _usage()}


# ── Brain primitives ─────────────────────────────────────────────────────────

class BuildReq(BaseModel):
    root: str
    force: bool = False


@app.post("/brains/build")
async def brains_build(req: BuildReq) -> dict:
    _fresh()
    root = Path(req.root)
    if not root.is_dir():
        raise HTTPException(400, f"root not found: {root}")
    await build_digests(root, force=req.force)
    return {"root": str(root), "built": True, "usage": _usage()}


class QueryReq(BaseModel):
    root: str
    query: str


@app.post("/brains/query")
async def brains_query(req: QueryReq) -> dict:
    _fresh()
    root = Path(req.root)
    if not root.is_dir():
        raise HTTPException(400, f"root not found: {root}")
    answer = await Brain(root).query(req.query)
    return {"answer": answer, "trace": get_trace(), "usage": _usage()}


class NeuronReq(BaseModel):
    query: str
    path: Optional[str] = None
    content: Optional[str] = None


@app.post("/neuron/cognize")
async def neuron_cognize(req: NeuronReq) -> dict:
    _fresh()
    content = req.content if req.content is not None else Path(req.path).read_text(errors="replace")
    raw = await _llm(hbrain.COGNIZE_SYSTEM.format(content=content),
                     f"Query: {req.query}", 2500)
    return {**_parse_vote(raw), "usage": _usage()}


@app.post("/neuron/instruct")
async def neuron_instruct(req: NeuronReq) -> dict:
    _fresh()
    content = req.content if req.content is not None else Path(req.path).read_text(errors="replace")
    out = await _llm(hbrain.INSTRUCT_SYSTEM.format(content=content),
                     f"Query: {req.query}", 3000)
    return {"instructions": out, "not_found": "NOT_FOUND" in out[:200],
            "usage": _usage()}


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model": hbrain.MODEL}


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HBRAIN_HOST", "127.0.0.1"),
                port=int(os.environ.get("HBRAIN_PORT", "8177")))


if __name__ == "__main__":
    main()
