"""The brain bandit — handler-grade memory of which brain earns its calls.

Cognize scores are a GAUGE: a brain's own claim of fitness, read off its digest
face. The bandit is the HANDLER side: accumulated evidence of how each brain
actually performed. Routing blends both — the gauge proposes, the record
tempers — and exploration comes from an optimism bonus on thin records, so a
newborn specialist gets pulled before its record exists.

Rewards must be handler-grade: a witnessed judge verdict, or explicit caller
acceptance. Never let a brain reward itself with its own gauge.

Ledger: one JSON file, {brain: {"pulls": int, "wins": float, "log": [...]}}.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Optional

LEDGER_ENV = "BRAIN_BANDIT_LEDGER"


def _ledger_path() -> Path:
    p = os.environ.get(LEDGER_ENV)
    if p:
        return Path(p)
    root = os.environ.get("HEAVEN_DATA_DIR", "/tmp")
    return Path(root) / "registry" / "brain_bandit.json"


def _load() -> dict:
    f = _ledger_path()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save(d: dict) -> None:
    f = _ledger_path()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=1))


def record(brain: str, reward: float, *, source: str, note: str = "") -> dict:
    """One handler-grade outcome for one brain. reward in [0, 1].
    source: where the judgment came from — 'judge', 'caller', 'witness'.
    """
    reward = max(0.0, min(1.0, float(reward)))
    d = _load()
    arm = d.setdefault(brain, {"pulls": 0, "wins": 0.0, "log": []})
    arm["pulls"] += 1
    arm["wins"] += reward
    arm["log"].append({"t": time.time(), "reward": reward,
                       "source": source, "note": note[:200]})
    arm["log"] = arm["log"][-200:]
    _save(d)
    return {"brain": brain, "pulls": arm["pulls"],
            "mean": arm["wins"] / arm["pulls"]}


def stats(brain: Optional[str] = None) -> dict:
    d = _load()
    if brain is not None:
        a = d.get(brain, {"pulls": 0, "wins": 0.0})
        return {"pulls": a["pulls"],
                "mean": (a["wins"] / a["pulls"]) if a["pulls"] else None}
    return {k: {"pulls": v["pulls"],
                "mean": (v["wins"] / v["pulls"]) if v["pulls"] else None}
            for k, v in d.items()}


def blend(gauge_score: float, brain: str, *, total_pulls: Optional[int] = None) -> float:
    """Routing value: the gauge (cognize 0-10, normalized) blended with the
    record, with the gauge's weight DECAYING as the record grows.

    w = 1/(1+pulls): a newborn is judged entirely on its face; after ten
    handler verdicts the face counts ~9%. This is what stops a confident
    liar — a fixed gauge weight let an arm with ten straight failures outrank
    an honest 0.7 purely on self-report (observed in test). The optimism
    bonus sqrt(ln(total)/(2n)) keeps thin records explored without drowning
    thick evidence of failure.
    """
    d = _load()
    arm = d.get(brain)
    g = max(0.0, min(1.0, gauge_score / 10.0))
    if not arm or not arm["pulls"]:
        return g
    n = arm["pulls"]
    mean = arm["wins"] / n
    total = total_pulls if total_pulls is not None else max(
        sum(a["pulls"] for a in d.values()), n)
    bonus = math.sqrt(math.log(max(total, 2)) / (2.0 * n))
    w = 1.0 / (1.0 + n)
    return w * g + (1.0 - w) * min(1.0, mean + bonus)
