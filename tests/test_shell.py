"""Shell + kernel tests.

These test the SHELL, which is the part that must be true regardless of any
model: a persistent namespace that OUTLIVES the caller.

Note what is deliberately NOT here: the old stubbed BrainShell root-loop tests.
They monkeypatched `_batch` in the test process, which is meaningless now that
the namespace lives in a separate kernel process — the stub could never reach
it. A root-loop test needs a live model; anything else is theatre.

Run: python tests/test_shell.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("BRAIN_KERNEL_DIR", tempfile.mkdtemp(prefix="brain-kernels-"))

import asyncio  # noqa: E402
from brain_agent.shell import PyShell  # noqa: E402
from brain_agent.kernel import Kernel  # noqa: E402

KERNEL = "test-rlm"


def test_pyshell_mechanics():
    sh = PyShell()
    assert asyncio.run(sh.run("acc = []")) == ""
    asyncio.run(sh.run("acc.append(1)"))
    assert asyncio.run(sh.run("print(len(acc))")).strip() == "1"
    assert "await ok" in asyncio.run(sh.run(
        "import asyncio\nawait asyncio.sleep(0)\nprint('await ok')"))
    assert asyncio.run(sh.run("import shutil; shutil.rmtree('/x')")).startswith("blocked by policy")
    assert "ZeroDivisionError" in asyncio.run(sh.run("1/0"))
    print("ok  pyshell           persist, top-level await, policy, traceback")


def test_print_survives_patched_builtins():
    """heaven_base rebinds builtins.print to something that reaches neither
    sys.stdout nor fd 1. The shell binds its own print so agent output is
    always captured."""
    import builtins
    original = builtins.print
    builtins.print = lambda *a, **k: None          # simulate the patch
    try:
        sh = PyShell()
        out = asyncio.run(sh.run("print('still captured')"))
        assert out.strip() == "still captured", repr(out)
    finally:
        builtins.print = original
    print("ok  patched_print     agent output captured despite a hostile builtins.print")


def _cli(*args):
    """Drive the kernel from a genuinely separate PROCESS — the whole point."""
    r = subprocess.run([sys.executable, "-m", "brain_agent.kernel", *args],
                       capture_output=True, text=True, cwd=str(REPO),
                       env={**os.environ, "PYTHONPATH": str(REPO) + os.pathsep
                            + os.environ.get("PYTHONPATH", "")})
    return r.stdout


def test_kernel_state_outlives_the_caller():
    """Each _cli call is a separate OS process. State must survive all of them."""
    Kernel(KERNEL).shutdown()
    _cli("exec", "--name", KERNEL, "import json; CORPUS = 'acct 44-9 MISMATCH ' * 500")
    out = _cli("exec", "--name", KERNEL, "print('read back:', len(CORPUS))")
    assert "read back: 9500" in out, out
    _cli("exec", "--name", KERNEL,
         "def chunk(s, n=200): return [s[i:i+n] for i in range(0, len(s), n)]\n"
         "class Neuron:\n    def __init__(s, c): s.c = c")
    out = _cli("exec", "--name", KERNEL,
               "ns = [Neuron(c) for c in chunk(CORPUS)]\n"
               "hits = [i for i, n in enumerate(ns) if '44-9' in n.c]\n"
               "print(json.dumps({'neurons': len(ns), 'hits': len(hits)}))")
    assert '"neurons": 48' in out and '"hits": 48' in out, out
    live = Kernel(KERNEL).vars()
    for name in ("CORPUS", "Neuron", "chunk", "ns", "hits", "json"):
        assert name in live, (name, live)
    print(f"ok  kernel_persists    4 separate processes, one namespace: {sorted(live)}")


def test_getvar_pulls_large_final():
    """Final leaves via the namespace, so it is not bounded by any context or
    stdout window."""
    k = Kernel(KERNEL)
    assert k.getvar("Final") is None
    _cli("exec", "--name", KERNEL, "Final = 'X' * 200000")
    val = k.getvar("Final")
    assert val is not None and len(val) == 200000, len(val or "")
    print(f"ok  final_via_var     pulled {len(val):,} chars out of the kernel")
    k.shutdown()


if __name__ == "__main__":
    test_pyshell_mechanics()
    test_print_survives_patched_builtins()
    test_kernel_state_outlives_the_caller()
    test_getvar_pulls_large_final()
    print("\nall shell/kernel tests passed (no model involved — by design)")
