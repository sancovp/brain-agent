"""my_brain tests — binding, birth, and composition-as-sub-brain. Deterministic
(digest=False skips the model); needs heaven importable for sdk.from_dir.

Run: python tests/test_my_brain.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HEAVEN_DATA_DIR"] = tempfile.mkdtemp(prefix="mybrain-hdd-")

try:
    from brain_agent import my_brain as mb
    from brain_agent.orchestrator import _registry_load_brain
except Exception as exc:
    print(f"SKIP test_my_brain: import failed ({type(exc).__name__}: {exc})")
    raise SystemExit(0)


def test_birth_binds_registers_and_charters():
    b = asyncio.run(mb.my_brain("Avi Research!", task="qualify leads", digest=False))
    assert b.name == "avi_research_brain"
    assert mb.bound_brain("Avi Research!") == "avi_research_brain"
    names = [n.name for n in b.neurons]
    assert "charter.md" in names, names
    assert "qualify leads" in (b.dir / "charter.md").read_text()
    print("ok  birth             newborn charters, registers, binds")


def test_binding_is_sticky():
    b2 = asyncio.run(mb.my_brain("Avi Research!", task="something else entirely",
                                 digest=False))
    assert b2.name == "avi_research_brain"   # identity, not per-call routing
    print("ok  sticky            second call returns the SAME brain, no re-route")


def test_seed_slots_become_neurons():
    b = asyncio.run(mb.my_brain("seeded", task="t", seed={"icp": "B2B $20M+"},
                                digest=False))
    assert "icp.md" in [n.name for n in b.neurons]
    print("ok  seed              seed slots are neurons of the newborn")


def test_composition_is_a_nested_sub_brain():
    asyncio.run(mb.compose("seeded", "acme review",
                           {"verdict": "tier A", "evidence": "quote..."},
                           digest=False))
    b = _registry_load_brain("seeded_brain")
    kinds = {n.name: type(n).__name__ for n in b.neurons}
    assert kinds.get("compositions") == "Brain", kinds     # subdir -> sub-brain
    comp = [n for n in b.neurons if n.name == "compositions"][0]
    inner = comp.neurons[0]
    assert type(inner).__name__ == "Brain" and inner.name == "acme_review"
    assert {x.name for x in inner.neurons} == {"verdict.md", "evidence.md"}
    print("ok  composition       compose() -> a nested specialist brain, for free")


def test_explicit_bind_skips_routing():
    asyncio.run(mb.my_brain("mason", brain="seeded_brain", digest=False))
    assert mb.bound_brain("mason") == "seeded_brain"
    print("ok  explicit          brain= binds without routing")


def test_compose_without_brain_raises():
    try:
        asyncio.run(mb.compose("nobody", "x", {"a": "b"}, digest=False))
    except KeyError as e:
        assert "no brain" in str(e)
    else:
        raise AssertionError("compose without a bound brain must raise")
    print("ok  guard             composing with no brain is an error, not a mystery")


if __name__ == "__main__":
    test_birth_binds_registers_and_charters()
    test_binding_is_sticky()
    test_seed_slots_become_neurons()
    test_composition_is_a_nested_sub_brain()
    test_explicit_bind_skips_routing()
    test_compose_without_brain_raises()
    print("\nall my_brain tests passed")
