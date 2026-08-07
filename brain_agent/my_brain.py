"""my_brain() — every agent gets THE brain that fits it, and keeps it.

The endgame contract:

  brain = await my_brain("avi", task="qualify B2B leads against an ICP")

  1. If "avi" is already bound to a brain, that brain is returned. An agent
     uses ITS brain — identity, not lookup-per-call.
  2. Otherwise the configurator routes the task over the registry (each brain
     cognizing its own fitness, tempered by the bandit record). A good enough
     fit is SET as the agent's brain.
  3. No good fit -> a new brain is BORN: a directory under
     $HEAVEN_DATA_DIR/brains/<agent>_brain seeded with the task charter (and
     any seed content), digested, registered, and bound.

  compose(agent, name, {...slots})   the agent's work products written as a
     SUBDIRECTORY of its brain. Directories are brains and subdirectories are
     sub-brains, so every composition IS a smaller, more specialized brain
     nested under the agent's own — specialization by accretion, no extra
     machinery.

Bindings ledger: $HEAVEN_DATA_DIR/registry/agent_brains.json {agent: brain}.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional


def _data_dir() -> Path:
    return Path(os.environ.get("HEAVEN_DATA_DIR", "/tmp/heaven-data"))


def _bindings_path() -> Path:
    return _data_dir() / "registry" / "agent_brains.json"


def bindings() -> dict:
    f = _bindings_path()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _bind(agent: str, brain: str) -> None:
    f = _bindings_path()
    f.parent.mkdir(parents=True, exist_ok=True)
    d = bindings()
    d[str(agent)] = {"brain": brain, "bound_at": time.time()}
    f.write_text(json.dumps(d, indent=1))


def bound_brain(agent: str) -> Optional[str]:
    e = bindings().get(str(agent))
    return e["brain"] if e else None


_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


def _brain_name(agent: str) -> str:
    return f"{_NAME_RE.sub('_', agent).strip('_').lower()}_brain"


async def my_brain(agent: str, task: Optional[str] = None, *,
                   seed: Optional[dict] = None, min_fit: float = 0.5,
                   min_gauge: int = 3, brain: Optional[str] = None,
                   digest: bool = True):
    """The agent's brain — existing binding, best registry fit, or a newborn.
    Returns a composable Brain value (sdk.Brain via the registry loader).

    `brain=` binds explicitly (no routing). `seed` is {slot_name: content}
    written into a newborn. `digest=False` is NO-MODEL mode: skips
    face-building AND routing (routing is itself a model call — a test caught
    it leaking), so an unbound agent births directly."""
    from .orchestrator import _registry_load_brain, _route_over_registry

    existing = bound_brain(agent)
    if existing:
        return _registry_load_brain(existing)

    if brain is not None:
        _bind(agent, brain)
        return _registry_load_brain(brain)

    if task and digest:
        rows = await _route_over_registry(task, digest=digest)
        # Binding needs BOTH: blend >= min_fit AND gauge >= min_gauge. The
        # bandit record is task-agnostic — one perfect reward floors blend at
        # 0.5 for ANY task — so without a gauge floor a video-planning agent
        # got bound to the contracts brain (observed live). Relevance is the
        # gauge's job; the record only tempers among relevant brains.
        if rows and rows[0][1] >= min_fit and rows[0][2] >= min_gauge:
            _bind(agent, rows[0][0])
            return _registry_load_brain(rows[0][0])

    # ── birth ────────────────────────────────────────────────────────────────
    name = _brain_name(agent)
    root = _data_dir() / "brains" / name
    root.mkdir(parents=True, exist_ok=True)
    charter = f"# {name}\n\nAgent: {agent}\n\nCharter: {task or 'general assistant brain'}\n"
    (root / "charter.md").write_text(charter)
    for slot, content in (seed or {}).items():
        fn = _NAME_RE.sub("_", str(slot)).strip("_") or "seed"
        (root / f"{fn}.md").write_text(content if isinstance(content, str) else str(content))
    if digest:
        from .hierarchical import build_digests
        await build_digests(root, force=True)
    from .brain_agent import register_brain
    try:
        register_brain(directory=str(root), brain_name=name)
    except Exception:
        pass                      # already registered: binding is what matters
    _bind(agent, name)
    return _registry_load_brain(name)


async def compose(agent: str, name: str, slots: dict, *, digest: bool = True) -> Path:
    """Write a composition as a SUB-BRAIN of the agent's brain. The agent's
    brain gains a smaller, more specialized nested brain — that is the whole
    mechanism, because subdirectories are already sub-brains."""
    from .orchestrator import _registry_brain_dir
    b = bound_brain(agent)
    if not b:
        raise KeyError(f"agent {agent!r} has no brain yet — call my_brain() first")
    root = _registry_brain_dir(b)
    comp = root / "compositions" / (_NAME_RE.sub("_", name).strip("_") or "comp")
    comp.mkdir(parents=True, exist_ok=True)
    for slot, content in slots.items():
        fn = _NAME_RE.sub("_", str(slot)).strip("_") or "slot"
        (comp / f"{fn}.md").write_text(content if isinstance(content, str) else str(content))
    if digest:
        from .hierarchical import build_digests
        await build_digests(comp, force=True)      # fold the new leaf only
    return comp


# ── CLI — the skill's script ─────────────────────────────────────────────────

def main(argv=None):
    import argparse
    import asyncio
    import sys
    ap = argparse.ArgumentParser(
        prog="python -m brain_agent.my_brain",
        description="Get (or make) THE brain for an agent; optionally query it.")
    ap.add_argument("agent", help="agent identity, e.g. 'avi' or 'research-dept'")
    ap.add_argument("task", nargs="?", default=None,
                    help="what this agent is for (routes/configures on first call)")
    ap.add_argument("--seed", action="append", default=[], metavar="NAME=FILE",
                    help="seed a newborn brain with FILE as slot NAME (repeatable)")
    ap.add_argument("--query", default=None, help="ask the bound brain this and print the answer")
    args = ap.parse_args(argv)

    async def go():
        seed = {}
        from pathlib import Path
        for spec in args.seed:
            name, _, path = spec.partition("=")
            seed[name] = Path(path).read_text(errors="replace")
        b = await my_brain(args.agent, task=args.task, seed=seed or None)
        sys.stdout.write(f"brain: {b.name} ({len(b.neurons)} neurons)\n")
        if args.query:
            sys.stdout.write(await b.query(args.query) + "\n")
    asyncio.run(go())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
