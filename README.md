# Brain Agent

Neural-inspired knowledge retrieval system built on heaven-base.

## Overview

Brain Agent provides a sophisticated system for organizing and querying knowledge from document collections. It uses a "neural" metaphor where documents become "neurons" that can be activated based on relevance to queries.

## Components

- **BrainAgent**: Main agent class for knowledge retrieval
- **CognizeTool**: Identifies relevant neurons for a query
- **InstructTool**: Generates instructions from activated neurons
- **QueryBrainTool**: Simple interface for querying registered brains
- **SynthesizerReplicant**: Alternative replicant-based interface

## Installation

```bash
# Install heaven-base first
pip install git+https://github.com/sancovp/heaven-base.git@v1.2.0

# Install brain-agent
pip install -e .
```

## Usage

```python
from brain_agent import BrainAgent, register_brain

# Register a brain (document collection)
register_brain(
    directory="/path/to/documents",
    brain_name="my_knowledge_base",
    chunk_size=-1  # whole files
)

# Create and query brain agent
agent = BrainAgent()
result = await agent.query("brain=my_knowledge_base query=What is machine learning?")
```

## Dependencies

- heaven-base>=1.2.0 (for core agent framework)
- langchain-core (for message types)
- Various LLM providers (OpenAI, Google, etc.)

## License

Private - All rights reserved.
## Hierarchical Brains (v0.2.0)

Brains whose neurons are brains. A `Brain` implements the same 2-stage Neuron
protocol its own neurons use — **cognize** (cheap relevance vote off the brain's
`_digest.md`) and **instruct** (a full recursive cognize→instruct→synthesize
pass) — so hierarchies nest to arbitrary depth. Digests build bottom-up
(`build_digests`): the level touching raw files gets an LLM fold; levels above
concatenate child digests verbatim, so distinctive vocabulary survives to the
root.

```python
from brain_agent.hierarchical import Brain, build_digests
await build_digests(Path("corpus"))            # the pyramid is just directories
answer = await Brain(Path("corpus")).query("...")
```

## HTTP server (judge + fill)

`brain-agent-http` serves the brain over HTTP for coordinate/configuration
engines: the caller owns addressing (which parts exist, which slots are empty);
the server owns judgment and generation.

- `POST /judge` — {parts, rule} → one verdict per part (complies / violates /
  not_applicable) with a **verbatim witness quote**, machine-verified to appear
  in the source (`witness_verified`). Exhaustive: every part judged, none
  skipped. Returns a `global_section` flag (no violations anywhere).
- `POST /fill` — {slot_label, siblings, n, brain_root?} → candidate spectrum
  completions; if `brain_root` is given the proposals are grounded in a
  witnessed brain synthesis (`grounded` flag reports honestly).
- `POST /brains/build`, `POST /brains/query`, `POST /neuron/cognize`,
  `POST /neuron/instruct`, `GET /health`.

Runs entirely on **heaven** (`UnifiedChat` + `HeavenAgentConfig`) — heaven owns
model routing and auth, exactly as `brain_agent/tools.py` does. Model is
`HBRAIN_MODEL` (default `MiniMax-M2.7-highspeed`, provider `HBRAIN_PROVIDER`,
default `ANTHROPIC` — heaven auto-routes MiniMax-\* to `MINIMAX_API_KEY`). No raw
provider SDK.

## RLM — Recursive Language Model (v0.3.0)

`RLM` owns a GROWING corpus end-to-end: live ingestion, incremental pyramid
maintenance, and query/judge over unbounded context. The context window is
replaced by a filesystem pyramid — each LLM call sees O(node) tokens while the
corpus is unbounded.

```python
from brain_agent.rlm import RLM

r = RLM("corpus_root", session="my_session")
r.ingest_message("user", "...")        # boundary rule: user text starts an iteration
r.ingest_message("assistant", "...")
await r.reindex()                        # folds ONLY dirty branches — O(changed)
res = await r.query("...")              # witnessed synthesis + descent refs
report = await r.judge("<rule>")        # exhaustive incidence row over all parts
```

HTTP: `/rlm/ingest`, `/rlm/query`, `/rlm/judge`, `/rlm/flush` (stateful
sessions keyed by root+session on the same `brain-agent-http` server).

Verified on a real 329-message agent transcript: 22 iterations / 3 phases
auto-folded, incremental growth re-folds exactly the dirty branch, needle
query answered with a verbatim source-cited quote.

## The shell — RLM as the paper defines it (v0.5.0)

The hierarchical brain above descends by a **fixed** Python program: the model
only ever emits a relevance score, and `hierarchical.py` decides what to open.
The RLM paper (arXiv:2512.24601 §2) requires the inverse — a symbolic handle to
the corpus, **symbolic recursion** (the model invokes sub-calls from code *it*
writes), and the answer returned from a variable rather than a finish action.
That is what `shell.py` + `kernel.py` + `sdk.py` add.

```python
from brain_agent import BrainShell

sh = BrainShell(P=corpus, kernel="rlm")        # corpus is a VARIABLE, not a prompt
answer = await sh.run("which ledger entries are disputed?")
```

The root model never sees the corpus. It writes Python against `P`, and only a
bounded stdout preview comes back each turn — so root context scales with the
number of turns, not with corpus size.

### The SDK, preloaded in the shell

The agent composes these at runtime; nothing walks a directory unless it asks.

| | |
|---|---|
| `Neuron(content, prompt=...)` | one chunk + one prompt |
| `Synthesizer(prompt=...)` | folds `[(name, text), ...]` into one answer |
| `Brain(neurons=[...], synthesizer=..., router=...)` | N neurons + 1 synthesizer; **a Brain is usable as a neuron**, so hierarchies nest |
| `await brain.vote(q)` | raw relevance scores — route them yourself |
| `await fanout(chunks, q)` | N parallel sub-calls |
| `await sub_rlm(task, P=...)` | a nested shell agent on its own kernel (a sub-RLM) |
| `from_dir(path)` | build a Brain from a directory tree |

The router is a plain callable `(query, votes, neurons) -> [index]`. The old
hardcoded traversal survives as `top_k_router()` — an explicit default, no
longer the mechanism.

### The kernel — state that outlives the caller

A shell is only interactive if it is still there on the **next** turn. Every
agent turn is a separate invocation, so the namespace lives in a daemon:

```bash
python -m brain_agent.kernel exec --name rlm 'CORPUS = open("big.txt").read()'
python -m brain_agent.kernel exec --name rlm 'print(len(CORPUS))'   # new process
python -m brain_agent.kernel ping --name rlm                        # live vars
```

Variables, imports, functions, and any brains the agent built persist across
processes. `Kernel` and `PyShell` import with the **standard library only** —
no heaven, no langchain — so the shell works on a box with no model wired up.

### Verified

Shell/kernel (no model): namespace persistence across four separate OS
processes; output captured even against a patched `builtins.print`
(`heaven_base` rebinds it to something that reaches neither `sys.stdout` nor
fd 1); a 200,000-char `Final` pulled out of the kernel.

Live (MiniMax-M2.7-highspeed via heaven): all ten SDK primitives; `sub_rlm`
recursion onto a child kernel; and an end-to-end run over a 60,709-char corpus
in which the model wrote seven cells, abandoned a first decomposition that
produced 1,188 false positives, recovered, and returned the correct answer.

## Call-graph tracing (v0.6.0)

Every agent call is recorded as a node — root turns, cells, brains, votes,
neurons, fanouts, sub-LLM calls, synthesizers, and `sub_rlm` delegations.
Recording is off unless `BRAIN_TRACE_DIR` is set, and needs only the standard
library; `networkx` is required only to *load* a graph (`pip install
brain-agent[graph]`).

```bash
BRAIN_TRACE_DIR=/tmp/run1 python my_rlm_script.py
python -m brain_agent.trace /tmp/run1
```

```
root     Exactly one ledger entry is disputed. Which account,  10.51s
  turn     turn 0                                                 4.39s
  turn     turn 2                                                 2.69s
  cell     import re                                              0.00s
  cell     Final = {                                              0.00s
```

```python
from brain_agent import load_run
import networkx as nx
G = load_run("/tmp/run1")
nx.is_tree(G.to_undirected())        # True — one caller per call
G.nodes["tr:4"]["elapsed_s"], G.nodes["tr:4"]["kind"]
```

The shape is a tree, but it is a DiGraph because it spans **processes**: a
`sub_rlm` child runs in its own kernel and writes its own file. The parent's
`sub_rlm` node carries the child's kernel name, and `load_run()` stitches on
that edge (marked `cross_process=True`). Nesting inside a process is automatic
via a contextvar, and the cell request carries the caller's span id so
kernel-side work hangs under the client's tree rather than orphaning.

Nodes carry `kind`, `label`, `elapsed_s`, `ok`/`error`, plus extras: a `vote`
node records the scores it produced, a `brain` node how many neurons opened.

### Parallelism

- `fanout(chunks, q)` and `Brain.vote` issue **N model calls in one batch** —
  measured 2.6× faster than the same six calls sequentially.
- `Brain.query` runs opened files and sub-brains concurrently.
- `sub_rlm` awaits its child, but `asyncio.gather` (exposed in the shell as
  `gather`) runs several children at once — each on its **own** kernel.
