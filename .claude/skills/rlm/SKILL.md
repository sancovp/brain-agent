---
name: rlm
description: Run the brain-agent RLM synthesizer on a task — an agent with a persistent Python shell, a CONTEXT state dict of chunk slots, and the full brain SDK. It routes across your registered brains (route/reward bandit), builds new specialist brains on the fly, and returns FINAL. Use for any task over context too large or too structured for one window — analysis over big corpora, multi-lens reviews, judged sweeps.
---

# RLM — the brain-manager agent as a callable script

One invocation = one task handed to the synthesizer agent. It manages its own
work: engineers CONTEXT slots, tiles/lenses the material, routes to existing
registered brains or designs new ones, and sets FINAL.

```bash
python -m brain_agent.orchestrator "TASK" \
  [--kernel NAME]            # persistent runtime context (default: rlm)
  [--reset]                  # fresh kernel + CONTEXT
  [--bind slot=/path/file]   # load a file into a CONTEXT slot (repeatable)
```

Requires: `MINIMAX_API_KEY` (or heaven-routed provider), `HEAVEN_DATA_DIR`
(brain registry + bandit ledger), optionally `BRAIN_CONTEXT_DIR`,
`BRAIN_TRACE_DIR` (call-graph), `BRAIN_KERNEL_DIR`.

The kernel persists between invocations: calling again with the same
`--kernel` resumes with all slots, vars, and brains intact. The bandit ledger
(`route`/`reward`) persists across ALL kernels — routing improves globally.
