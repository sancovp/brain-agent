"""RLM shell tests.

The provider round-trip is stubbed (`_batch`), so these prove the MECHANISM:
the root writes code, the code runs in a persistent namespace, sub-calls are
launched from that code, and the answer comes back out of `Final` — never
through the root's context window.

Run: PYTHONNOUSERSITE=1 python tests/test_shell.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain_agent import hierarchical, shell as shell_mod, sdk
from brain_agent.shell import BrainShell, PyShell


class FakeResp:
    def __init__(self, text):
        self.content = text


def script_model(turns):
    """Replace heaven's _batch with a scripted model. Each root call pops the
    next scripted reply; neuron/fanout batches answer per-message."""
    state = {"i": 0, "neuron_calls": 0, "seen_by_root": []}

    async def fake_batch(message_lists, **kw):
        # A neuron/fanout batch: many messages at once, or a non-root system prompt.
        first = message_lists[0][0].content if message_lists and message_lists[0] else ""
        if not first.startswith("You are the root"):
            state["neuron_calls"] += len(message_lists)
            out = []
            for ml in message_lists:
                body = ml[0].content
                out.append(FakeResp("FOUND: acct 44-9 mismatch" if "44-9" in body else "NOT_FOUND"))
            return out
        # A root call: record what the root was allowed to see, return next script line.
        state["seen_by_root"].append("\n".join(m.content for m in message_lists[0]))
        i = state["i"]
        state["i"] += 1
        return [FakeResp(turns[i] if i < len(turns) else "```python\nFinal = 'gave up'\n```")]

    return fake_batch, state


CORPUS = "\n\n".join(
    [f"ledger entry {i}: acct {i}-0 ok" for i in range(40)]
    + ["ledger entry 99: acct 44-9 MISMATCH against invoice 7"]
    + [f"ledger entry {i}: acct {i}-1 ok" for i in range(40, 80)]
)


def test_root_loop_writes_code_and_returns_via_final():
    turns = [
        # 1. the root inspects P without ever printing it
        "```python\nchunks = P.split('\\n\\n')\nprint('chunks:', len(chunks))\n```",
        # 2. it launches N sub-calls from code IT wrote
        "```python\nouts = await fanout(chunks, 'find any mismatch')\n"
        "hits = [o for o in outs if o.startswith('FOUND')]\nprint('hits:', len(hits))\n```",
        # 3. the answer leaves via a variable, not the context window
        "```python\nFinal = 'MISMATCH: ' + hits[0] + ' (' + 'x' * 5000 + ')'\n```",
    ]
    fake, state = script_model(turns)
    hierarchical._batch = fake
    sdk._batch = fake

    sh = BrainShell(P=CORPUS)
    answer = asyncio.run(sh.run("which ledger entries disagree with an invoice?"))

    assert answer.startswith("MISMATCH: FOUND: acct 44-9"), answer
    # unbounded output: Final is returned verbatim, past any root output window
    assert len(answer) > 5000, len(answer)
    # symbolic recursion: 81 sub-calls launched from model-authored code
    assert state["neuron_calls"] == 81, state["neuron_calls"]
    # symbolic handle: the corpus never entered the root's context
    root_saw = "\n".join(state["seen_by_root"])
    assert "44-9 MISMATCH against invoice 7" not in root_saw
    assert "ledger entry 20" not in root_saw
    assert f"P is a str of {len(CORPUS)} chars" in root_saw
    print(f"ok  root_loop        sub_calls={state['neuron_calls']} answer={len(answer)}ch "
          f"root_context={len(root_saw)}ch corpus={len(CORPUS)}ch")


def test_stdout_is_truncated_to_metadata():
    turns = ["```python\nprint('z' * 50000)\n```", "```python\nFinal = 'done'\n```"]
    fake, state = script_model(turns)
    hierarchical._batch = fake
    sdk._batch = fake
    sh = BrainShell(P="tiny")
    asyncio.run(sh.run("t"))
    assert len(state["seen_by_root"][-1]) < 20000, len(state["seen_by_root"][-1])
    assert "truncated" in state["seen_by_root"][-1]
    print("ok  stdout_truncated  root never ate the 50k print")


def test_namespace_persists_and_policy_blocks():
    sh = PyShell()
    asyncio.run(sh.run("acc = []"))
    asyncio.run(sh.run("acc.append(1)"))
    out = asyncio.run(sh.run("print(len(acc))"))
    assert out.strip() == "1", out
    blocked = asyncio.run(sh.run("import shutil; shutil.rmtree('/tmp/x')"))
    assert blocked.startswith("blocked by policy"), blocked
    err = asyncio.run(sh.run("1/0"))
    assert "ZeroDivisionError" in err
    print("ok  namespace+policy  state persists, denylist holds, tracebacks captured")


def test_sub_rlm_nests_and_caps_depth():
    fake, state = script_model(["```python\nFinal = 'child answer'\n```"])
    hierarchical._batch = fake
    sdk._batch = fake
    parent = BrainShell(P="", max_depth=2)
    child_out = asyncio.run(parent.shell.ns["sub_rlm"]("do a subtask", P="sub corpus"))
    assert child_out == "child answer", child_out

    deep = BrainShell(P="", depth=2, max_depth=2)
    leaf = asyncio.run(deep.shell.ns["sub_rlm"]("too deep", P="x"))
    assert leaf in ("NOT_FOUND", "") or "NOT_FOUND" in leaf, leaf
    print("ok  sub_rlm           nested shell answers; depth cap degrades to sub_llm")


def test_agent_can_compose_a_brain_at_runtime():
    """The SDK is composable: neurons, a synthesizer and a custom router built
    from values in the shell, with no directory anywhere."""
    fake, _ = script_model([])
    hierarchical._batch = fake
    sdk._batch = fake
    sh = PyShell()
    sh.ns.update({"Neuron": sdk.Neuron, "Brain": sdk.Brain,
                  "Synthesizer": sdk.Synthesizer, "asyncio": asyncio})
    out = asyncio.run(sh.run(
        "ns = [Neuron(content=c, name=f'n{i}') for i, c in enumerate(['a', 'acct 44-9', 'c'])]\n"
        "def my_router(q, votes, neurons): return [i for i, n in enumerate(neurons) if '44-9' in str(n.content)]\n"
        "b = Brain(neurons=ns, router=my_router, name='adhoc')\n"
        "print('built', b.name, len(b.neurons), 'router picks', my_router(None, None, ns))\n"))
    assert "built adhoc 3 router picks [1]" in out, out
    print("ok  compose_at_runtime neurons+brain+custom router built in the shell")


if __name__ == "__main__":
    test_namespace_persists_and_policy_blocks()
    test_agent_can_compose_a_brain_at_runtime()
    test_root_loop_writes_code_and_returns_via_final()
    test_stdout_is_truncated_to_metadata()
    test_sub_rlm_nests_and_caps_depth()
    print("\nall shell tests passed (provider stubbed — no real model call)")
