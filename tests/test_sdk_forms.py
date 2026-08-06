"""Call-shape tests for the SDK — the forms an agent actually reaches for.

Every one of these pins a shape that failed live and made an agent abandon the
primitive. No model is called; these are pure input-normalization checks.

Needs heaven importable (sdk imports the heaven call path at module load).
Run: python tests/test_sdk_forms.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from brain_agent.sdk import Synthesizer, Neuron, NEURON_FRAME, _messages
except Exception as exc:                       # heaven not installed here
    print(f"SKIP test_sdk_forms: sdk import failed ({type(exc).__name__}: {exc})")
    raise SystemExit(0)


def test_synthesizer_accepts_natural_shapes():
    n = Synthesizer._normalize
    assert n({"legal": "a", "money": "b"}) == [("legal", "a"), ("money", "b")]
    assert n([("legal", "a")]) == [("legal", "a")]
    assert n(["a", "b"]) == [("source_0", "a"), ("source_1", "b")]
    assert n([["legal", "a"]]) == [("legal", "a")]
    assert n([]) == []
    print("ok  synth_shapes      dict / list[str] / list[tuple] / list[list] all normalize")


def test_neuron_content_is_always_grounded():
    """A terse custom lens must not let the model answer from general
    knowledge — that failed silently, which is the worst way to fail."""
    msgs = _messages("fee $48,000", "describe", "Report only amounts.")
    system = msgs[0].content
    assert NEURON_FRAME.strip() in system
    assert "<neuron content>" in system and "fee $48,000" in system
    print("ok  neuron_grounding  every lens, however terse, is framed by its content")


def test_neuron_lens_is_carried_into_messages():
    a = Neuron(content="x", name="a", prompt="LENS-A")
    b = Neuron(content="x", name="b", prompt="LENS-B")
    assert "LENS-A" in a.instruct_msgs("q")[0].content
    assert "LENS-B" in b.instruct_msgs("q")[0].content
    assert "LENS-A" not in b.instruct_msgs("q")[0].content
    print("ok  lens_isolation    two lenses over one chunk stay independent")


if __name__ == "__main__":
    test_synthesizer_accepts_natural_shapes()
    test_neuron_content_is_always_grounded()
    test_neuron_lens_is_carried_into_messages()
    print("\nall sdk-form tests passed")
