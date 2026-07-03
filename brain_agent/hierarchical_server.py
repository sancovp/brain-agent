"""brain-agent HTTP server — judge, fill, and RLM sessions, on HEAVEN.

The contract:
  The CALLER owns addressing — it knows which parts/cells exist and which
  slots are empty (zero-able). This server owns judgment and generation:
    /judge  — one heaven neuron per part: part x rule -> verdict + VERBATIM witness
    /fill   — generative spectrum completion, optionally brain-grounded
    /rlm/*  — stateful growing-corpus sessions (ingest / query / judge / flush)
  Plus the brain primitives: /brains/build, /brains/query, /neuron/*.

Every LLM call goes through heaven (UnifiedChat) via the hierarchical primitives.

Run:  brain-agent-http     (or: python -m brain_agent.hierarchical_server)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from . import hierarchical as hbrain
from .hierarchical import (Brain, USAGE_VAR, TRACE_VAR, Usage, build_digests,
                           get_trace, _neuron_messages, _batch, _plain,
                           _content, _score, COGNIZE_SYSTEM, INSTRUCT_SYSTEM)
from .rlm import RLM, JUDGE_SYSTEM

app = FastAPI(title="brain-agent", version="0.4.0")


def _fresh() -> Usage:
    u = Usage()
    USAGE_VAR.set(u)
    TRACE_VAR.set([])
    return u


def _usage(u: Usage) -> dict:
    return {"calls": u.calls, "model": hbrain.MODEL}


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


# ── health / brain primitives ────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model": hbrain.MODEL}


class BuildReq(BaseModel):
    root: str
    force: bool = False


@app.post("/brains/build")
async def brains_build(req: BuildReq) -> dict:
    u = _fresh()
    root = Path(req.root)
    if not root.is_dir():
        raise HTTPException(400, f"root not found: {root}")
    await build_digests(root, force=req.force)
    return {"root": str(root), "built": True, "usage": _usage(u)}


class QueryReq(BaseModel):
    root: str
    query: str


@app.post("/brains/query")
async def brains_query(req: QueryReq) -> dict:
    u = _fresh()
    root = Path(req.root)
    if not root.is_dir():
        raise HTTPException(400, f"root not found: {root}")
    answer = await Brain(root).query(req.query)
    return {"answer": answer, "trace": get_trace(), "usage": _usage(u)}


class NeuronReq(BaseModel):
    query: str
    path: str


@app.post("/neuron/cognize")
async def neuron_cognize(req: NeuronReq) -> dict:
    u = _fresh()
    resp = await _batch([_neuron_messages(req.path, req.query, COGNIZE_SYSTEM)],
                        max_tokens=700)
    return {**_score(_content(resp[0])), "usage": _usage(u)}


@app.post("/neuron/instruct")
async def neuron_instruct(req: NeuronReq) -> dict:
    u = _fresh()
    resp = await _batch([_neuron_messages(req.path, req.query, INSTRUCT_SYSTEM)],
                        max_tokens=3000)
    out = _content(resp[0]) if resp else "NOT_FOUND"
    return {"instructions": out, "not_found": "NOT_FOUND" in out[:200],
            "usage": _usage(u)}


# ── JUDGE: exhaustive part × rule incidence row ──────────────────────────────

class Part(BaseModel):
    id: str
    path: Optional[str] = None
    content: Optional[str] = None


class JudgeReq(BaseModel):
    parts: list[Part]
    rule: str


def _judge_messages(part: Part, rule: str) -> list:
    """Render a part for judgment the heaven way (path= block) or inline."""
    if part.path:
        return _neuron_messages(part.path, rule, JUDGE_SYSTEM)
    body = (JUDGE_SYSTEM + "\n\n<neuron content>\n" + (part.content or "")
            + "\n</neuron content>")
    return [SystemMessage(content=body), HumanMessage(content=f"RULE: {rule}")]


@app.post("/judge")
async def judge(req: JudgeReq) -> dict:
    """Exhaustive judge mode: EVERY part judged, none skipped — coverage by
    construction. One heaven neuron call per part; the incidence-matrix row."""
    u = _fresh()
    responses = await _batch([_judge_messages(p, req.rule) for p in req.parts],
                             max_tokens=2500)
    cells = []
    for p, resp in zip(req.parts, responses):
        content = (p.content if p.content is not None
                   else Path(p.path).read_text(errors="replace") if p.path else "")
        d = _first_json(_content(resp)) or {}
        verdict = d.get("verdict", "not_applicable")
        witness = str(d.get("witness", ""))
        if verdict not in ("complies", "violates", "not_applicable"):
            verdict = "not_applicable"
        if verdict != "not_applicable" and not witness.strip():
            verdict = "not_applicable"
        cells.append({"id": p.id, "verdict": verdict,
                      "score": _score(_content(resp))["score"], "witness": witness,
                      "witness_verified": bool(witness) and witness in content,
                      "reasoning": str(d.get("reasoning", ""))})
    verdicts = [c["verdict"] for c in cells]
    return {"rule": req.rule, "cells": cells,
            "summary": {"parts": len(cells),
                        "complies": verdicts.count("complies"),
                        "violates": verdicts.count("violates"),
                        "not_applicable": verdicts.count("not_applicable"),
                        "global_section": verdicts.count("violates") == 0},
            "usage": _usage(u)}


# ── FILL: generative completion for an empty spectrum slot ────────────────────

FILL_SYSTEM = (
    "You are a SpectrumFiller for a coordinate engine. A node's children ARE its "
    "spectrum — the set of choices at that slot. A spectrum needs at least a high "
    "and a low, and its members must be mutually exclusive alternatives at the "
    "SAME level of abstraction as the existing siblings. Respond with a JSON "
    "object, key 'candidates': a list of objects with keys 'label' (short, "
    "concrete), 'rationale' (one sentence), 'confidence' (0-10 integer). Propose "
    "exactly the requested number. No prose outside the JSON.")


class FillReq(BaseModel):
    slot_label: str
    parent_label: Optional[str] = None
    siblings: list[str] = []
    space_context: Optional[str] = None
    n: int = 3
    brain_root: Optional[str] = None


@app.post("/fill")
async def fill(req: FillReq) -> dict:
    u = _fresh()
    grounding = ""
    if req.brain_root:
        root = Path(req.brain_root)
        if not root.is_dir():
            raise HTTPException(400, f"brain_root not found: {root}")
        grounding = await Brain(root).query(
            f"What is known about '{req.slot_label}'"
            + (f" in the context of '{req.parent_label}'" if req.parent_label else "")
            + "? Collect concrete facts, names, and distinctions useful for "
              "enumerating its variants.")
    grounded = bool(grounding and "NOT_FOUND" not in grounding[:60])
    user = (f"SLOT: {req.slot_label}\n"
            + (f"PARENT: {req.parent_label}\n" if req.parent_label else "")
            + (f"EXISTING SIBLINGS: {', '.join(req.siblings)}\n" if req.siblings else "")
            + (f"SPACE CONTEXT:\n{req.space_context}\n" if req.space_context else "")
            + (f"GROUNDING (witnessed synthesis):\n{grounding}\n" if grounded else "")
            + f"\nPropose {req.n} candidates for this spectrum.")
    raw = await _plain(FILL_SYSTEM, user, max_tokens=3000)
    data = _first_json(raw) or {}
    cands = []
    for c in (data.get("candidates", []) or [])[:req.n]:
        try:
            cands.append({"label": str(c.get("label", ""))[:120],
                          "rationale": str(c.get("rationale", "")),
                          "confidence": max(0, min(10, int(c.get("confidence", 0))))})
        except (ValueError, TypeError):
            continue
    return {"slot": req.slot_label, "candidates": cands, "grounded": grounded,
            "grounding": grounding or None, "usage": _usage(u)}


# ── RLM: stateful growing-corpus sessions ────────────────────────────────────

_rlms: dict = {}


def _rlm(root: str, session: Optional[str]) -> RLM:
    key = f"{root}::{session or 'session_default'}"
    if key not in _rlms:
        _rlms[key] = RLM(root, session=session or "session_default")
    return _rlms[key]


class RLMIngestReq(BaseModel):
    root: str
    session: Optional[str] = None
    role: str
    text: str
    tools: list[str] = []


@app.post("/rlm/ingest")
async def rlm_ingest(req: RLMIngestReq) -> dict:
    r = _rlm(req.root, req.session)
    r.ingest_message(req.role, req.text, req.tools)
    return {"ok": True, "session": r.session, "iterations": r._n}


class RLMQueryReq(BaseModel):
    root: str
    session: Optional[str] = None
    goal: str


@app.post("/rlm/query")
async def rlm_query(req: RLMQueryReq) -> dict:
    res = await _rlm(req.root, req.session).query(req.goal)
    return {"answer": res.answer, "refs": res.refs, "usage": res.usage}


class RLMJudgeReq(BaseModel):
    root: str
    session: Optional[str] = None
    rule: str


@app.post("/rlm/judge")
async def rlm_judge(req: RLMJudgeReq) -> dict:
    return await _rlm(req.root, req.session).judge(req.rule)


class RLMFlushReq(BaseModel):
    root: str
    session: Optional[str] = None


@app.post("/rlm/flush")
async def rlm_flush(req: RLMFlushReq) -> dict:
    r = _rlm(req.root, req.session)
    path = r.flush()
    res = await r.reindex()
    return {"flushed": str(path) if path else None, **res}


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HBRAIN_HOST", "127.0.0.1"),
                port=int(os.environ.get("HBRAIN_PORT", "8177")))


if __name__ == "__main__":
    main()
