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
