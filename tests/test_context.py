"""ContextDict tests — the state dict at the center of the RLM. Deterministic;
no model, no heaven.

Run: python tests/test_context.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain_agent.context import ContextDict, _slot_file  # noqa: E402


def test_set_persists_to_chunk_file():
    d = Path(tempfile.mkdtemp(prefix="ctx-"))
    c = ContextDict(d)
    c["doc"] = "hello chunk"
    assert (d / "doc.md").read_text() == "hello chunk"
    c["doc"] = "rewritten"
    assert (d / "doc.md").read_text() == "rewritten"
    del c["doc"]
    assert not (d / "doc.md").exists() and "doc" not in c
    print("ok  persist           set writes the file, del removes it")


def test_values_are_chunks():
    c = ContextDict(Path(tempfile.mkdtemp(prefix="ctx-")))
    c["n"] = 42                       # coerced: a slot IS text
    assert c["n"] == "42" and c.path("n").read_text() == "42"
    print("ok  chunks            values coerce to str; the file is authoritative")


def test_adopts_existing_chunks():
    d = Path(tempfile.mkdtemp(prefix="ctx-"))
    (d / "prior.md").write_text("from an earlier session")
    c = ContextDict(d)
    assert c["prior"] == "from an earlier session"
    print("ok  adoption          a prior session's chunk files load as slots")


def test_path_is_for_neurons_and_missing_slot_raises():
    c = ContextDict(Path(tempfile.mkdtemp(prefix="ctx-")))
    c["a"] = "x"
    assert c.path("a").name == "a.md"
    try:
        c.path("nope")
    except KeyError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("missing slot must raise, not invent a path")
    print("ok  path              slot file for Neuron(content=...); missing raises")


def test_slot_names_are_sanitized():
    assert _slot_file("weird name/../x") == "weird_name_.._x.md" or "/" not in _slot_file("weird name/../x")
    c = ContextDict(Path(tempfile.mkdtemp(prefix="ctx-")))
    c["has spaces / slashes"] = "v"
    files = list(c.dir.iterdir())
    assert len(files) == 1 and "/" not in files[0].name
    print("ok  sanitize          slot names cannot escape the context dir")


def test_slots_listing_previews_not_dumps():
    c = ContextDict(Path(tempfile.mkdtemp(prefix="ctx-")))
    c["big"] = "A" * 100000
    out = c.slots()
    assert "100000" in out and len(out) < 500
    print("ok  listing           slots() shows sizes and previews, never the chunk")


if __name__ == "__main__":
    test_set_persists_to_chunk_file()
    test_values_are_chunks()
    test_adopts_existing_chunks()
    test_path_is_for_neurons_and_missing_slot_raises()
    test_slot_names_are_sanitized()
    test_slots_listing_previews_not_dumps()
    print("\nall context tests passed")
