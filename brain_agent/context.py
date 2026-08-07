"""CONTEXT — the state dict at the center of the RLM.

A dict whose slots are chunks of context, each persisted to a file the moment it
is set. The agent engineers this dict with full Python: slice chunks into slots,
pipe vars between them, point neurons and brains at the slot FILES, write
results back into new slots, and keep going until FINAL holds the answer.

The persistence is the point, twice over:
  * neurons read files (heaven's path= rendering), so a slot is directly
    callable context;
  * a directory of slot files IS a brain (from_dir / DigestBrain), so the
    working context can be cognized, judged, and digested like any corpus.

    CONTEXT["contract_a"] = chunk          # persisted to <dir>/contract_a.md
    n = Neuron(content=CONTEXT.path("contract_a"), prompt="Report only risk.")
    CONTEXT["risk_a"] = await n("review")  # result becomes context too
    b = CONTEXT.brain()                    # the whole dict as a brain
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_SLOT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slot_file(name: str) -> str:
    clean = _SLOT_RE.sub("_", str(name)).strip("._") or "slot"
    return clean if "." in clean else f"{clean}.md"


class ContextDict(dict):
    """A dict of context slots, each mirrored to a chunk file on write.

    Values are coerced to str on set — a slot IS a chunk of context, and the
    file is the authoritative copy (neurons read the file, not the memory).
    Deleting a slot deletes its file. Non-str values you don't want flattened
    belong in plain shell variables, not in CONTEXT.
    """

    def __init__(self, directory):
        super().__init__()
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        # Adopt chunks already on disk (a prior session's context).
        for f in sorted(self.dir.iterdir()):
            if f.is_file() and not f.name.startswith((".", "_")):
                super().__setitem__(f.stem, f.read_text(errors="replace"))

    # ── the dict, persisted ──────────────────────────────────────────────────

    def __setitem__(self, name, value) -> None:
        text = value if isinstance(value, str) else str(value)
        super().__setitem__(str(name), text)
        (self.dir / _slot_file(name)).write_text(text, errors="replace")

    def __delitem__(self, name) -> None:
        super().__delitem__(str(name))
        f = self.dir / _slot_file(name)
        if f.exists():
            f.unlink()

    def update(self, *args, **kw) -> None:          # keep persistence on update()
        for k, v in dict(*args, **kw).items():
            self[k] = v

    # ── the slots as callable context ────────────────────────────────────────

    def path(self, name) -> Path:
        """The slot's FILE — what a Neuron(content=...) should be given."""
        f = self.dir / _slot_file(name)
        if not f.exists():
            raise KeyError(f"no slot {name!r}; slots: {sorted(self)}")
        return f

    def slots(self) -> str:
        """One line per slot: name, size, preview. Print this, not the chunks."""
        if not self:
            return "(CONTEXT is empty)"
        lines = []
        for k in sorted(self):
            v = self[k]
            preview = " ".join(v[:80].split())
            lines.append(f"{k:<24} {len(v):>8} chars  {preview}")
        return "\n".join(lines)

    def brain(self, router=None, name: Optional[str] = None):
        """The whole CONTEXT as a brain — every slot a neuron."""
        from .sdk import from_dir
        return from_dir(self.dir, router=router, name=name or "CONTEXT")
