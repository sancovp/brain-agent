"""Neuromorphic context — the activation graph and the membrane.

The design (aios-research/NEUROMORPHIC-CONTEXT-DESIGN.md): context units are
REFS with a resting form (boxed summary) and a body. A numeric activation pass
over a kuzu concept graph decides which refs FIRE; the membrane renders the
view — fired refs disclose, everything else stays a one-line box. cognize
scores TEACH the graph (gauge->amplitude annealing), so routing anneals from
LLM-taught to purely numeric. Requires the optional `kuzu` dependency for the
graph; the Membrane alone is stdlib.
"""
from __future__ import annotations

import math
import shutil

try:
    import kuzu
except ImportError:                     # membrane works without the graph
    kuzu = None


class ActivationGraph:
    """Persistent by default — the graph IS long-term memory. reset=True wipes
    (kuzu 0.11 stores the DB as a FILE + .wal; plain rmtree misses files)."""

    def __init__(self, path="./neurodb", reset=False):
        if kuzu is None:
            raise ImportError("ActivationGraph needs the optional 'kuzu' package "
                              "(pip install brain-agent[neuro]); the Membrane "
                              "alone works without it")
        import os
        if reset:
            for p in (path, str(path) + ".wal"):
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.exists(p):
                    os.unlink(p)
        existed = os.path.exists(path)
        self.db = kuzu.Database(str(path))
        self.conn = kuzu.Connection(self.db)
        if not existed or reset:
            self.conn.execute(
                "CREATE NODE TABLE Concept(name STRING, kind STRING, amplitude DOUBLE, "
                "PRIMARY KEY (name))")
            self.conn.execute(
                "CREATE REL TABLE RELATES(FROM Concept TO Concept, weight DOUBLE)")

    def add(self, name, kind="ref", amplitude=0.5):
        self.conn.execute(
            "MERGE (c:Concept {name: $n}) SET c.kind = $k, c.amplitude = $a",
            {"n": name, "k": kind, "a": float(amplitude)})

    def wire(self, a, b, weight=0.5):
        self.conn.execute(
            "MATCH (x:Concept {name: $a}), (y:Concept {name: $b}) "
            "MERGE (x)-[r:RELATES]->(y) SET r.weight = $w",
            {"a": a, "b": b, "w": float(weight)})

    def hebbian(self, a, b, lr=0.2):
        """Co-disclosure in a rewarded run strengthens the edge."""
        r = self.conn.execute(
            "MATCH (x:Concept {name: $a})-[r:RELATES]->(y:Concept {name: $b}) "
            "RETURN r.weight", {"a": a, "b": b})
        w = r.get_next()[0] if r.has_next() else 0.0
        self.wire(a, b, w + lr * (1.0 - w))

    def spread(self, stimulus: dict, hops=2, decay=0.6) -> dict:
        """Numeric spreading activation. stimulus: {name: initial_activation}."""
        act = dict(stimulus)
        frontier = dict(stimulus)
        for _ in range(hops):
            nxt = {}
            for name, a in frontier.items():
                r = self.conn.execute(
                    "MATCH (x:Concept {name: $n})-[e:RELATES]->(y:Concept) "
                    "RETURN y.name, e.weight, y.amplitude", {"n": name})
                while r.has_next():
                    y, w, amp = r.get_next()
                    delta = a * w * decay * (0.5 + amp / 2)
                    if delta > 0.01:
                        nxt[y] = max(nxt.get(y, 0.0), delta)
            for k, v in nxt.items():
                act[k] = max(act.get(k, 0.0), v)
            frontier = nxt
        return act

    def teach(self, stimulus: dict, ref_scores: dict, lr: float = 0.3) -> None:
        """The gauge->amplitude annealing: cognize scores SET the weights.

        For each ref, amplitude moves toward score/10 (EMA — repeated lessons
        converge, one outlier doesn't overwrite). For each stimulus concept,
        the concept->ref edge moves toward the same target, so next time the
        SAME kind of stimulus fires this ref numerically, without the teacher.
        parse_failed scores must be filtered out BEFORE calling this — a
        teaching loop that learns from garbage anneals garbage.
        """
        for ref, score in ref_scores.items():
            target = max(0.0, min(1.0, score / 10.0))
            r = self.conn.execute(
                "MATCH (c:Concept {name: $n}) RETURN c.amplitude", {"n": ref})
            if not r.has_next():
                self.add(ref, "ref", target)
                cur = target
            else:
                cur = r.get_next()[0]
                self.conn.execute(
                    "MATCH (c:Concept {name: $n}) SET c.amplitude = $a",
                    {"n": ref, "a": cur + lr * (target - cur)})
            for concept in stimulus:
                r = self.conn.execute(
                    "MATCH (x:Concept {name: $a})-[e:RELATES]->(y:Concept {name: $b}) "
                    "RETURN e.weight", {"a": concept, "b": ref})
                w = r.get_next()[0] if r.has_next() else 0.0
                self.wire(concept, ref, w + lr * (target - w))

    def kinds(self, names) -> dict:
        if not names:
            return {}
        r = self.conn.execute(
            "MATCH (c:Concept) WHERE c.name IN $ns RETURN c.name, c.kind",
            {"ns": list(names)})
        out = {}
        while r.has_next():
            n, k = r.get_next()
            out[n] = k
        return out

    def disclose(self, stimulus: dict, budget: int = 3, temp: float = 0.5,
                 kind: str = "ref"):
        """The softmax under budget: which REFS fire (slinky-show) this turn.
        Competition is among disclosable units only — stimulus concepts are
        inputs, not contestants (the first version echoed them back and they
        crowded refs out of the budget)."""
        act = self.spread(stimulus)
        km = self.kinds(act)
        act = {n: a for n, a in act.items() if km.get(n) == kind or kind is None}
        if not act:
            return []
        z = [(n, math.exp(a / temp)) for n, a in act.items()]
        total = sum(v for _, v in z)
        probs = sorted(((n, v / total) for n, v in z), key=lambda x: -x[1])
        return probs[:budget]




class Membrane:
    def __init__(self, store: dict):
        """store: {ref: (summary, full_content)} — the resting forms + bodies."""
        self.store = dict(store)
        self.disclosed: set = set()

    def fire(self, refs) -> None:
        self.disclosed |= {r for r in refs if r in self.store}

    def inhibit(self, refs) -> None:
        self.disclosed -= set(refs)

    def set_firing(self, refs) -> None:
        """Replace the active set wholesale (one activation pass = one state)."""
        self.disclosed = {r for r in refs if r in self.store}

    def render(self, char_budget: int = 8000) -> str:
        """The view. Disclosed refs spend the budget in order; a fired ref that
        no longer fits is DEMOTED to its box rather than truncated mid-chunk —
        a half-disclosed chunk is worse than an honest box."""
        lines = []
        spent = 0
        for ref in sorted(self.store):
            summary, content = self.store[ref]
            if ref in self.disclosed and spent + len(content) <= char_budget:
                lines.append(f"<{ref}>\n{content}\n</{ref}>")
                spent += len(content)
            else:
                lines.append(f"[📦 {ref}] {summary}")
        return "\n".join(lines)

    def stats(self) -> dict:
        full = sum(len(c) for _, c in self.store.values())
        view = len(self.render(char_budget=10**9))
        cur = len(self.render())
        return {"refs": len(self.store), "disclosed": len(self.disclosed),
                "full_chars": full, "current_view_chars": cur,
                "compression": round(full / max(cur, 1), 1)}
