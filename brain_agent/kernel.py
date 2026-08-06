"""The kernel — a PyShell that OUTLIVES the caller.

PyShell alone is in-process: state persists across `run()` calls only while that
process lives. That is useless for an agent, because every agent turn is a
separate invocation — the shell must still be there on the next turn, next tool
call, next process.

So the namespace lives in a daemon. Clients connect, send a cell, get stdout
back, and disconnect. The namespace stays.

    python -m brain_agent.kernel start --name rlm      # daemon, survives callers
    python -m brain_agent.kernel exec  --name rlm 'x = 1'
    python -m brain_agent.kernel exec  --name rlm 'print(x)'   # -> 1 (new process)

From Python (this is what the agent's tool calls):

    from brain_agent.kernel import Kernel
    k = Kernel("rlm")
    k.start()                 # no-op if already running
    k.exec("import json")     # state accumulates across processes
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from .shell import PyShell, default_policy

RUNTIME_DIR = Path(os.environ.get("BRAIN_KERNEL_DIR",
                                  Path(tempfile.gettempdir()) / "brain-kernels"))


def socket_path(name: str) -> Path:
    return RUNTIME_DIR / f"{name}.sock"


# ── server ───────────────────────────────────────────────────────────────────

async def serve(name: str, policy=default_policy, idle_timeout: Optional[float] = None):
    """Hold ONE namespace and serve cells over a unix socket until killed."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    path = socket_path(name)
    if path.exists():
        path.unlink()
    shell = PyShell(policy=policy)
    shell.ns["__kernel_name__"] = name
    last = {"t": time.time()}

    async def handle(reader, writer):
        try:
            raw = await reader.readline()
            if not raw:
                return
            req = json.loads(raw.decode("utf-8"))
            last["t"] = time.time()
            if req.get("op") == "ping":
                resp = {"ok": True, "pid": os.getpid(),
                        "vars": sorted(k for k in shell.ns if not k.startswith("__"))}
            elif req.get("op") == "getvar":
                key = req.get("name", "")
                present = key in shell.ns and shell.ns[key] is not None
                val = shell.ns.get(key)
                resp = {"ok": True, "set": present,
                        "value": val if isinstance(val, str) else
                                 (None if val is None else str(val))}
            elif req.get("op") == "shutdown":
                resp = {"ok": True}
                writer.write((json.dumps(resp) + "\n").encode()); await writer.drain()
                writer.close()
                asyncio.get_running_loop().stop()
                return
            else:
                out = await shell.run(req.get("code", ""))
                resp = {"ok": True, "stdout": out,
                        "vars": sorted(k for k in shell.ns if not k.startswith("__"))}
        except Exception as exc:                        # never kill the kernel
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        writer.write((json.dumps(resp) + "\n").encode())
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handle, path=str(path))
    os.chmod(path, 0o600)
    async with server:
        if idle_timeout:
            while time.time() - last["t"] < idle_timeout:
                await asyncio.sleep(1)
        else:
            await server.serve_forever()


# ── client ───────────────────────────────────────────────────────────────────

class Kernel:
    """Client handle to a named kernel. Cheap to construct; the state is remote."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.path = socket_path(name)

    def alive(self) -> bool:
        if not self.path.exists():
            return False
        try:
            self._request({"op": "ping"}, timeout=2)
            return True
        except OSError:
            return False

    def start(self, wait: float = 10.0) -> "Kernel":
        """Spawn the daemon if it is not already up. Idempotent."""
        if self.alive():
            return self
        if self.path.exists():
            self.path.unlink()
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        log = open(RUNTIME_DIR / f"{self.name}.log", "ab")
        subprocess.Popen(
            [sys.executable, "-m", "brain_agent.kernel", "start", "--name", self.name],
            stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            start_new_session=True,      # survives the caller's process group
        )
        deadline = time.time() + wait
        while time.time() < deadline:
            if self.alive():
                return self
            time.sleep(0.1)
        raise RuntimeError(f"kernel {self.name!r} did not come up; see "
                           f"{RUNTIME_DIR / (self.name + '.log')}")

    def _request(self, payload: dict, timeout: float = 300) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(self.path))
            s.sendall((json.dumps(payload) + "\n").encode())
            chunks = []
            while not (chunks and chunks[-1].endswith(b"\n")):
                b = s.recv(65536)
                if not b:
                    break
                chunks.append(b)
        return json.loads(b"".join(chunks).decode("utf-8"))

    def exec(self, code: str, timeout: float = 300) -> str:
        r = self._request({"op": "exec", "code": code}, timeout=timeout)
        if not r.get("ok"):
            return f"kernel error: {r.get('error')}"
        return r.get("stdout", "")

    def vars(self) -> list:
        return self._request({"op": "ping"}).get("vars", [])

    def getvar(self, name: str, timeout: float = 60) -> Optional[str]:
        """Pull a variable's value out of the remote namespace. Returns None if
        unset — this is how the loop retrieves `Final` without it ever passing
        through the root's context window."""
        r = self._request({"op": "getvar", "name": name}, timeout=timeout)
        return r.get("value") if r.get("ok") and r.get("set") else None

    def shutdown(self) -> None:
        try:
            self._request({"op": "shutdown"}, timeout=5)
        except OSError:
            pass
        if self.path.exists():
            self.path.unlink(missing_ok=True)


# ── cli ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.stdout.write((__doc__ or "") + "\n")
        return 0
    op, argv = argv[0], argv[1:]
    name = "default"
    if "--name" in argv:
        i = argv.index("--name")
        name = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if op == "start":
        try:
            asyncio.run(serve(name))
        except (KeyboardInterrupt, RuntimeError):
            pass
        return 0
    k = Kernel(name)
    if op == "exec":
        code = argv[0] if argv else sys.stdin.read()
        k.start()                      # idempotent: attach if up, spawn if not
        sys.stdout.write(k.exec(code))
    elif op == "ping":
        # not print(): heaven_base patches builtins.print and it never reaches fd 1
        sys.stdout.write(json.dumps({"alive": k.alive(), "vars": k.vars() if k.alive() else []}) + "\n")
    elif op == "shutdown":
        k.shutdown()
        sys.stdout.write("stopped\n")
    else:
        sys.stdout.write(f"unknown op {op!r}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
