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

## The orchestrator — RLM as the paper defines it (v0.8.0)

The hierarchical brain above descends by a **fixed** Python program: the model
only emits a relevance score and `hierarchical.py` decides what to open. The RLM
paper (arXiv:2512.24601 §2) requires the inverse — a symbolic handle to the
data, **symbolic recursion** (the model invokes sub-calls from code *it* writes),
and the answer returned from a variable rather than a finish action.

A brain is N neurons + 1 synthesizer. In an RLM the synthesizer stops being a
one-shot fold and becomes the thing that **runs the turn**: it builds brain
systems and calls them however it wants, as many at once as it wants, until it
is satisfied.

```python
from brain_agent import Orchestrator

orch = Orchestrator(kernel="work")
orch.bind(contracts=big_string)          # anything, by name, into the context
answer = await orch.run("review these from three independent perspectives")
```

- **One persistent runtime context.** Neurons, brains and whole systems it
  builds are live Python objects in a kernel; results pipe between them as
  values, never re-serialized through the model.
- **The shell is a real tool** (`ShellTool`), called through heaven's native
  tool loop. The model's output formatting is not load-bearing.
- **No turn cap.** It decides when the answer is good enough. The only bound is
  a runaway backstop (`BRAIN_MAX_TOOL_CALLS`, default 1000).
- **Recursion needs no special case**, because a Brain is usable as a neuron.

A real run, from the call graph: the orchestrator built three lensed brains, ran
them concurrently inside one shell call, then built a *fourth* brain whose
neurons were those three analyses:

```
orchestrator Review these contracts from three independent expert  166.89s
  shell    import asyncio                                           43.91s
    brain    Legal Risk Brain            opened=3/3  scores=[10, 9, 9]
    brain    Financial Exposure Brain    opened=3/3  scores=[10, 9, 9]
    brain    Renewal Timeline Brain      opened=3/3  scores=[8, 9, 6]
  shell    from brain_agent import Neuron, Synthesizer, Compose     61.68s
    brain    Priority Synthesizer        opened=3/3  scores=[10, 9, 10]
```

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

## Lenses (v0.7.0)

A neuron's `prompt=` is its **lens** — what it reports and what it ignores. Give
the same content different lenses to get readings that cannot contaminate each
other, then fold them with a `Synthesizer` whose prompt controls the form:

```python
lenses = {"legal":    "Report ONLY legal risk. Ignore money and dates.",
          "money":    "Report ONLY amounts and payment terms.",
          "timeline": "Report ONLY dates and durations."}
ns = [Neuron(content=doc, name=k, prompt=v) for k, v in lenses.items()]
reads = await fanout(ns, "review this contract")     # each keeps its own lens
report = await Synthesizer(prompt="Return JSON with keys legal, money, timeline.")(
    dict(zip(lenses, reads)))
```

`cognize_prompt=` is the separate routing lens (how a neuron scores its own
relevance). Lens *separation quality* tracks lens prompt quality: one-line
lenses ground correctly but partition weakly; a lens that names what to ignore
partitions cleanly.

## CONTEXT — the state dict (v0.10.0)

The synthesizer agent's working memory is `CONTEXT`, a state dict in its shell.
Each slot is a chunk of context, persisted to a file the moment it is set — and
neurons read files, so a slot is directly callable context. A directory of slot
files is a brain, so `CONTEXT.brain()` turns the working context itself into
one, and `register_brain(CONTEXT.dir, name)` makes it permanent.

```python
CONTEXT["contract_a"] = chunk               # persisted to <ctx>/contract_a.md
n = Neuron(content=CONTEXT.path("contract_a"), prompt="Report only risk.")
CONTEXT["risk_a"] = await n("review")       # results become context too
FINAL = report                              # ends the work; returned unbounded
```

The instructed loop: get material into slots → make vars and pipe them through
neurons/brains reading the slot files → write results back as new slots →
reslice and relens until satisfied → set `FINAL`. Slots survive kernel restarts
(chunk files re-adopt on boot), `bind()` places values into slots, and `FINAL`
is retrieved from the kernel so the answer is not bounded by the model's reply.
