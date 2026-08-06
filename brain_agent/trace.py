"""Call-graph tracing — every agent call recorded as a node, emitted as a graph.

The shape is a tree (each call has exactly one caller), but it is recorded as a
DiGraph because it spans PROCESSES: a `sub_rlm` child runs in its own kernel, so
its nodes are written by a different process and only join the tree at load
time. Each process appends JSONL to `{BRAIN_TRACE_DIR}/{kernel}.jsonl`; a
`sub_rlm` node carries the child's kernel name, and `load_run()` stitches the
files together on that edge.

Recording is off unless `BRAIN_TRACE_DIR` is set (or `enable()` is called), so
it costs nothing by default.

    BRAIN_TRACE_DIR=/tmp/run1 python my_rlm_script.py
    python -m brain_agent.trace /tmp/run1            # text tree + summary

    from brain_agent.trace import load_run
    G = load_run("/tmp/run1")        # networkx.DiGraph
    G.nodes["par:0"]["kind"], G.nodes["par:0"]["elapsed_s"]

Nodes carry: kind (root | cell | brain | vote | neuron | fanout | sub_llm |
sub_rlm | synth), label, model, elapsed_s, ok, error, plus kind-specific extras
(a `vote` node records the scores, a `brain` node records which neurons opened).
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

_TRACE_DIR: Optional[Path] = None
_KERNEL = os.environ.get("BRAIN_KERNEL_SELF", "main")
_SEQ = 0
_LOCK = threading.Lock()
_PARENT: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("brain_trace_parent",
                                                                       default=None)


def enable(directory, kernel: Optional[str] = None) -> None:
    """Turn recording on for this process."""
    global _TRACE_DIR, _KERNEL
    _TRACE_DIR = Path(directory)
    _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    if kernel:
        _KERNEL = kernel


def _auto_enable() -> None:
    d = os.environ.get("BRAIN_TRACE_DIR")
    if d and _TRACE_DIR is None:
        enable(d)


_auto_enable()


def active() -> bool:
    return _TRACE_DIR is not None


def _next_id() -> str:
    global _SEQ
    with _LOCK:
        _SEQ += 1
        return f"{_KERNEL}:{_SEQ}"


def _write(rec: dict) -> None:
    if _TRACE_DIR is None:
        return
    path = _TRACE_DIR / f"{_KERNEL}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


@contextlib.contextmanager
def span(kind: str, label: str = "", **attrs: Any):
    """Record one call. Nesting is automatic via contextvar, so a neuron opened
    inside a brain inside a cell lands under both without threading anything."""
    if _TRACE_DIR is None:
        yield _NullSpan()
        return
    node_id = _next_id()
    rec = {"id": node_id, "parent": _PARENT.get(), "kind": kind, "label": label[:200],
           "t_start": time.time(), **attrs}
    token = _PARENT.set(node_id)
    handle = _NullSpan()
    handle.id = node_id
    handle.extra = {}
    try:
        yield handle
        rec["ok"] = True
    except BaseException as exc:
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _PARENT.reset(token)
        rec["elapsed_s"] = round(time.time() - rec.pop("t_start"), 3)
        rec.update(handle.extra)
        _write(rec)


class _NullSpan:
    """Handle yielded by `span`; `extra` is merged into the node on exit."""
    id: Optional[str] = None
    extra: dict

    def __init__(self):
        self.extra = {}

    def set(self, **kw) -> None:
        self.extra.update(kw)


def current_parent() -> Optional[str]:
    return _PARENT.get()


# ── loading ──────────────────────────────────────────────────────────────────

def load_records(directory) -> list:
    out = []
    for f in sorted(Path(directory).glob("*.jsonl")):
        for line in f.read_text(errors="replace").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def load_run(directory):
    """Merge every process's records into one networkx.DiGraph.

    Cross-process edges: a `sub_rlm` node records `child_kernel`; that child's
    own root node is whichever node in the child's file has no parent."""
    import networkx as nx           # optional: only needed to LOAD a graph
    records = load_records(directory)
    G = nx.DiGraph()
    roots_by_kernel = {}
    for r in records:
        G.add_node(r["id"], **{k: v for k, v in r.items() if k not in ("id", "parent")})
        if r.get("parent") is None:
            roots_by_kernel.setdefault(r["id"].split(":", 1)[0], r["id"])
    for r in records:
        if r.get("parent") and r["parent"] in G:
            G.add_edge(r["parent"], r["id"])
    for r in records:
        child_root = roots_by_kernel.get(r.get("child_kernel") or "")
        if child_root and child_root != r["id"]:
            G.add_edge(r["id"], child_root, cross_process=True)
    return G


def text_tree(directory) -> str:
    """Render the run without needing networkx."""
    records = load_records(directory)
    by_id = {r["id"]: r for r in records}
    kids: dict = {}
    roots = []
    kernel_roots = {}
    for r in records:
        if r.get("parent") is None:
            kernel_roots.setdefault(r["id"].split(":", 1)[0], r["id"])
    for r in records:
        p = r.get("parent")
        if p is None:
            roots.append(r["id"])
        else:
            kids.setdefault(p, []).append(r["id"])
    for r in records:
        cr = kernel_roots.get(r.get("child_kernel") or "")
        if cr and cr != r["id"]:
            kids.setdefault(r["id"], []).append(cr)
            if cr in roots:
                roots.remove(cr)

    def _order(nid):
        """Sort by sequence number, not string: 'k:10' must follow 'k:9'."""
        kern, _, n = nid.rpartition(":")
        return (kern, int(n) if n.isdigit() else 0)

    lines = []

    def walk(nid, depth):
        r = by_id[nid]
        mark = "" if r.get("ok", True) else "  !! " + str(r.get("error", ""))[:60]
        extra = ""
        if "scores" in r:
            extra = f" scores={r['scores']}"
        elif "opened" in r:
            extra = f" opened={r['opened']}/{r.get('n', '?')}"
        elif "n" in r:
            extra = f" n={r['n']}"
        lines.append(f"{'  ' * depth}{r['kind']:<8} {r.get('label', '')[:52]:<52}"
                     f" {r.get('elapsed_s', 0):>6.2f}s{extra}{mark}")
        for k in sorted(kids.get(nid, []), key=_order):
            walk(k, depth + 1)

    for root in sorted(roots, key=_order):
        walk(root, 0)
    total = sum(r.get("elapsed_s", 0) for r in records)
    llm = [r for r in records if r["kind"] in ("neuron", "fanout", "sub_llm", "synth", "vote", "turn")]
    lines.append(f"\n{len(records)} nodes | {len(llm)} model-calling nodes | "
                 f"{len({r['id'].split(':', 1)[0] for r in records})} processes | "
                 f"sum of spans {total:.1f}s")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(text_tree(sys.argv[1] if len(sys.argv) > 1 else os.environ["BRAIN_TRACE_DIR"]))
