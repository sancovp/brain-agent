"""Bandit ledger tests — deterministic, no model, no heaven.

Run: python tests/test_bandit.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["BRAIN_BANDIT_LEDGER"] = str(Path(tempfile.mkdtemp(prefix="bandit-")) / "ledger.json")

from brain_agent import bandit  # noqa: E402


def test_no_record_means_gauge_stands_alone():
    assert bandit.blend(8, "newborn") == 0.8
    assert bandit.blend(0, "newborn") == 0.0
    print("ok  newborn           unpulled arm is judged on its face alone")


def test_record_tempers_the_gauge():
    for _ in range(10):
        bandit.record("liar", 0.0, source="judge", note="confident, wrong")
    for _ in range(10):
        bandit.record("workhorse", 1.0, source="judge")
    # Same gauge claim, opposite records — the record must separate them.
    assert bandit.blend(9, "workhorse") > bandit.blend(9, "liar")
    # A liar with a perfect gauge must fall below a modest honest arm.
    assert bandit.blend(10, "liar") < bandit.blend(7, "workhorse")
    print("ok  tempering         a confident record-breaker loses to an honest record")


def test_optimism_bonus_shrinks_with_pulls():
    bandit.record("thin", 0.5, source="judge")
    thin = bandit.blend(5, "thin")
    for _ in range(50):
        bandit.record("thick", 0.5, source="judge")
    thick = bandit.blend(5, "thick")
    assert thin > thick, (thin, thick)  # same mean, thinner record explores more
    print("ok  exploration       UCB bonus decays as the record thickens")


def test_reward_is_clamped_and_logged():
    r = bandit.record("clamp", 7.0, source="caller")   # out of range -> 1.0
    assert r["mean"] == 1.0
    s = bandit.stats("clamp")
    assert s["pulls"] == 1 and s["mean"] == 1.0
    assert bandit.stats()["clamp"]["pulls"] == 1
    print("ok  ledger            rewards clamp to [0,1]; stats read back")


if __name__ == "__main__":
    test_no_record_means_gauge_stands_alone()
    test_record_tempers_the_gauge()
    test_optimism_bonus_shrinks_with_pulls()
    test_reward_is_clamped_and_logged()
    print("\nall bandit tests passed")
