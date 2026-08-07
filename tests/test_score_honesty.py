"""_score honesty: a parse failure is not a verdict of zero.

Run: python tests/test_score_honesty.py   (needs heaven importable)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from brain_agent.hierarchical import _score
except Exception as exc:
    print(f"SKIP: heaven not importable ({type(exc).__name__})")
    raise SystemExit(0)


def test_real_zero_is_not_a_parse_failure():
    v = _score('{"score": 0, "reasoning": "irrelevant"}')
    assert v["score"] == 0 and v["parse_failed"] is False
    print("ok  real_zero         a genuine 0 is a verdict, flagged as parsed")


def test_garbage_is_a_parse_failure_not_a_verdict():
    for raw in ("total nonsense", "", '{"score": "not json...', '{broken'):
        v = _score(raw)
        assert v["score"] == 0 and v["parse_failed"] is True, raw
    print("ok  garbage           unparseable replies carry parse_failed=True")


def test_normal_scores_unaffected():
    assert _score('{"score": 8, "reasoning": "yes"}') == {
        "score": 8, "reasoning": "yes", "parse_failed": False}
    v = _score('prefix noise {"related_to": true} suffix')
    assert v["score"] == 8 and v["parse_failed"] is False
    print("ok  compat            existing score paths unchanged")


if __name__ == "__main__":
    test_real_zero_is_not_a_parse_failure()
    test_garbage_is_a_parse_failure_not_a_verdict()
    test_normal_scores_unaffected()
    print("\nall score-honesty tests passed")
