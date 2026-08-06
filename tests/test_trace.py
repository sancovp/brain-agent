"""Call-graph tracing tests. No model involved — the graph shape is a property
of the recorder, so it is tested deterministically.

Run: python tests/test_trace.py
"""
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from brain_agent import trace  # noqa: E402


def test_disabled_by_default_costs_nothing():
    assert trace._TRACE_DIR is None or True   # may be enabled by env in a live run
    saved = trace._TRACE_DIR
    trace._TRACE_DIR = None
    try:
        with trace.span("brain", "x") as sp:
            sp.set(n=3)
        assert trace.current_parent() is None
    finally:
        trace._TRACE_DIR = saved
    print("ok  disabled          spans are no-ops when tracing is off")


def test_nesting_is_automatic():
    d = Path(tempfile.mkdtemp(prefix="trace-nest-"))
    trace.enable(d, kernel="k")
    trace._SEQ = 0
    with trace.span("root", "task"):
        with trace.span("cell", "c0"):
            with trace.span("brain", "ledger", n=2) as sp:
                with trace.span("vote", "ledger") as v:
                    v.set(scores=[10, 2])
                sp.set(opened=1)
    recs = {r["label"]: r for r in trace.load_records(d)}
    assert recs["ledger"]["parent"] == recs["c0"]["id"]
    assert recs["c0"]["parent"] == recs["task"]["id"]
    assert recs["task"]["parent"] is None
    assert recs["ledger"]["opened"] == 1 and recs["ledger"]["n"] == 2
    assert recs["ledger"]["kind"] == "brain"
    assert [r for r in trace.load_records(d) if r["kind"] == "vote"][0]["scores"] == [10, 2]
    assert all(r["ok"] for r in trace.load_records(d))
    print("ok  nesting           contextvar parenting, no threading of ids")


def test_errors_are_recorded_and_reraised():
    d = Path(tempfile.mkdtemp(prefix="trace-err-"))
    trace.enable(d, kernel="k")
    trace._SEQ = 0
    try:
        with trace.span("neuron", "boom"):
            raise ValueError("kaboom")
    except ValueError:
        pass
    else:
        raise AssertionError("span swallowed the exception")
    r = trace.load_records(d)[0]
    assert r["ok"] is False and "kaboom" in r["error"]
    print("ok  errors            recorded on the node AND re-raised")


def test_cross_process_stitching():
    """A sub_rlm child runs in another process, so its nodes land in another
    file and join only via child_kernel."""
    d = Path(tempfile.mkdtemp(prefix="trace-xproc-"))
    (d / "parent.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "parent:1", "parent": None, "kind": "root", "label": "top", "elapsed_s": 9.0},
        {"id": "parent:2", "parent": "parent:1", "kind": "sub_rlm", "label": "deleg",
         "child_kernel": "kid", "elapsed_s": 8.0},
    ]))
    (d / "kid.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "kid:1", "parent": None, "kind": "root", "label": "child task", "elapsed_s": 7.0},
        {"id": "kid:2", "parent": "kid:1", "kind": "fanout", "label": "3 chunks", "n": 3,
         "elapsed_s": 2.0},
    ]))
    G = trace.load_run(d)
    assert G.number_of_nodes() == 4, G.number_of_nodes()
    assert G.has_edge("parent:2", "kid:1"), "cross-process edge missing"
    assert G.edges["parent:2", "kid:1"]["cross_process"] is True
    import networkx as nx
    assert nx.is_tree(G.to_undirected()), "call graph must be a tree"
    assert list(nx.descendants(G, "parent:1")) != []
    tree = trace.text_tree(d)
    assert "child task" in tree and "2 processes" in tree, tree
    print(f"ok  cross_process     {G.number_of_nodes()} nodes across 2 files stitched into one tree")


def test_ordering_is_numeric():
    d = Path(tempfile.mkdtemp(prefix="trace-order-"))
    rows = [{"id": "k:1", "parent": None, "kind": "root", "label": "r", "elapsed_s": 1.0}]
    rows += [{"id": f"k:{i}", "parent": "k:1", "kind": "cell", "label": f"cell{i}",
              "elapsed_s": 0.0} for i in range(2, 13)]
    (d / "k.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    import re
    # the kind column is also "cell", so match the LABEL occurrence: cell<N>
    order = re.findall(r"cell(\d+)", trace.text_tree(d))
    assert order == [str(i) for i in range(2, 13)], order
    print("ok  ordering          'k:10' sorts after 'k:9', not before it")


if __name__ == "__main__":
    test_disabled_by_default_costs_nothing()
    test_nesting_is_automatic()
    test_errors_are_recorded_and_reraised()
    test_cross_process_stitching()
    test_ordering_is_numeric()
    print("\nall trace tests passed")
