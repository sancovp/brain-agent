"""Orchestrator tests — the parts that must hold without a model.

The orchestrator's contract is: one persistent runtime context, a real tool (not
a parsed code fence), and no turn cap. Those are checkable directly.

Needs heaven importable for the tool-class build. Run: python tests/test_orchestrator.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("BRAIN_KERNEL_DIR", tempfile.mkdtemp(prefix="orch-kernels-"))

from brain_agent import orchestrator as orch  # noqa: E402
from brain_agent.kernel import Kernel  # noqa: E402

KERNEL = "test-orch"


def test_no_turn_cap_by_default():
    """The agent decides when it is done. The only bound is a runaway backstop,
    and it must be far above any real run."""
    assert orch.UNBOUNDED_TOOL_CALLS >= 1000, orch.UNBOUNDED_TOOL_CALLS
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    assert orch.Orchestrator.__dataclass_fields__["max_tool_calls"].default >= 1000
    print(f"ok  no_turn_cap       backstop only, at {orch.UNBOUNDED_TOOL_CALLS} tool calls")


def test_shell_is_a_real_tool_not_a_parsed_fence():
    from brain_agent.shell import PyShell  # noqa: F401
    import brain_agent.shell as shell_mod
    assert not hasattr(shell_mod, "_extract_code"), "the fence parser must be gone"
    assert not hasattr(shell_mod, "BrainShell"), "the capped driver loop must be gone"
    ShellTool = orch._build_tool_classes()
    assert ShellTool.name == "ShellTool" and ShellTool.is_async is True
    tool = ShellTool.create()          # heaven must accept the schema
    assert tool is not None
    print("ok  real_tool          ShellTool builds; no fence parser, no capped loop")


def test_runtime_context_persists_across_tool_calls():
    orch.set_current_kernel(KERNEL)
    Kernel(KERNEL).shutdown()
    asyncio.run(orch.shell_func("made_here = [1, 2, 3]", kernel=KERNEL))
    out = asyncio.run(orch.shell_func("print(sum(made_here))", kernel=KERNEL))
    assert out.strip() == "6", out
    print("ok  context_persists   values survive between independent tool calls")


def test_large_output_is_truncated_but_the_value_is_not():
    """Truncation bounds the MODEL's context, never its data."""
    asyncio.run(orch.shell_func("big = 'x' * 50000", kernel=KERNEL))
    seen = asyncio.run(orch.shell_func("print(big)", kernel=KERNEL))
    assert len(seen) < orch.TOOL_OUTPUT_LIMIT + 300, len(seen)
    assert "withheld" in seen
    still = asyncio.run(orch.shell_func("print(len(big))", kernel=KERNEL))
    assert still.strip() == "50000", still
    print("ok  truncation         model sees 4k; the variable is still 50,000 chars")


def test_empty_output_is_explained_not_silent():
    out = asyncio.run(orch.shell_func("x = 1", kernel=KERNEL))
    assert "no output" in out.lower(), out
    print("ok  empty_output       a silent cell says so instead of returning ''")


def test_final_text_handles_heavens_dict_return():
    """heaven returns {"history": History(...)} — treating that as an object
    made run() return a 39,000-char repr of the whole conversation."""
    class Msg:
        def __init__(self, content): self.content = content
    class AIMessage(Msg): pass
    class HumanMessage(Msg): pass

    class History:
        messages = [HumanMessage("do the thing"),
                    AIMessage([{"type": "tool_use", "input": {}}]),
                    AIMessage([{"type": "text", "text": "the answer"}])]
    assert orch._final_text({"history": History()}) == "the answer"
    assert orch._final_text("already a string") == "already a string"
    print("ok  final_text         digs the answer out of heaven's dict return")


if __name__ == "__main__":
    test_no_turn_cap_by_default()
    test_shell_is_a_real_tool_not_a_parsed_fence()
    test_runtime_context_persists_across_tool_calls()
    test_large_output_is_truncated_but_the_value_is_not()
    test_empty_output_is_explained_not_silent()
    test_final_text_handles_heavens_dict_return()
    Kernel(KERNEL).shutdown()
    print("\nall orchestrator tests passed")
