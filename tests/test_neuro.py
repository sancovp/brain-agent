"""Neuromorphic layer tests — graph, membrane, and the orchestrator wiring.
Deterministic: kuzu yes, model no.

Run: python tests/test_neuro.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["BRAIN_CONTEXT_DIR"] = tempfile.mkdtemp(prefix="neuro-ctx-")

from brain_agent.neuro import ActivationGraph, Membrane  # noqa: E402


def _g(name="db"):
    return ActivationGraph(os.path.join(tempfile.mkdtemp(), name))


def test_spread_softmax_and_ref_filter():
    g = _g()
    g.add("pii", "concept"); g.add("dpa", "concept")
    g.add("ref_counsel", "ref"); g.add("ref_video", "ref")
    g.wire("pii", "dpa", 0.9); g.wire("dpa", "ref_counsel", 0.8)
    fired = g.disclose({"pii": 1.0}, budget=2)
    names = [n for n, _ in fired]
    assert "ref_counsel" in names, names
    assert "pii" not in names and "dpa" not in names, "stimulus concepts must not compete"
    print("ok  spread            2-hop associative recall; refs-only competition")


def test_teach_annealing():
    g = _g()
    g.add("billing", "concept")
    assert g.disclose({"billing": 1.0}, budget=2) == []      # blind before
    g.teach({"billing": 1.0}, {"ref_ledger": 9, "ref_story": 1})
    after = dict(g.disclose({"billing": 1.0}, budget=3))
    assert after["ref_ledger"] > after.get("ref_story", 0)
    for _ in range(4):
        g.teach({"billing": 1.0}, {"ref_ledger": 9})
    g.teach({"billing": 1.0}, {"ref_ledger": 0})              # one outlier
    final = dict(g.disclose({"billing": 1.0}, budget=3))
    assert final["ref_ledger"] > final.get("ref_story", 0)
    print("ok  teach             cognize lesson routes; EMA survives an outlier")


def test_graph_persists_across_instances():
    p = os.path.join(tempfile.mkdtemp(), "p")
    g1 = ActivationGraph(p); g1.add("a", "concept", 0.7); del g1
    g2 = ActivationGraph(p)
    r = g2.conn.execute("MATCH (c:Concept) RETURN c.amplitude")
    assert abs(r.get_next()[0] - 0.7) < 1e-9
    ActivationGraph(p, reset=True)                             # wipe works too
    print("ok  persistence       the graph IS long-term memory; reset wipes")


def test_membrane_budget_demotes_never_truncates():
    store = {f"it{i}": (f"sum {i}", "BODY " + "x" * 900) for i in range(8)}
    m = Membrane(store)
    assert m.render().count("📦") == 8
    m.set_firing(list(store))
    tight = m.render(char_budget=2000)
    assert tight.count("</") <= 2 and "📦" in tight
    for boxed_line in [l for l in tight.splitlines() if "📦" in l]:
        assert "xxxx" not in boxed_line                        # box, not a cut chunk
    print("ok  membrane          budget demotes to boxes; no half-disclosed chunks")


def test_orchestrator_view_wiring():
    from brain_agent.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)                     # no kernel needed
    o.kernel = "wiretest"; o.view_budget = 4000
    o._record_run("analyze disputed invoice acct 44-9", "dispute confirmed $9999")
    o._record_run("storyboard the explainer video", "three beats drafted")
    g = o._graph()
    g.add("billing_dispute", "concept")
    g.teach({"billing_dispute": 1.0}, {"run_000": 9, "run_001": 1})
    t = o._transcript()
    assert sorted(t) == ["run_000", "run_001"]
    store = {r: (b.splitlines()[0][:110], b) for r, b in t.items()}
    m = Membrane(store)
    m.set_firing([n for n, _ in g.disclose({"billing_dispute": 1.0}, budget=1)])
    v = m.render(4000)
    assert "<run_000>" in v and "[📦 run_001]" in v
    print("ok  wiring            prior runs recorded; relevant discloses, rest boxed")


if __name__ == "__main__":
    test_spread_softmax_and_ref_filter()
    test_teach_annealing()
    test_graph_persists_across_instances()
    test_membrane_budget_demotes_never_truncates()
    test_orchestrator_view_wiring()
    print("\nall neuro tests passed")
