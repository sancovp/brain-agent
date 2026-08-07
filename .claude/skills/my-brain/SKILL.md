---
name: my-brain
description: Give any agent ITS brain. First call routes the agent's purpose over the registered brains (bandit-tempered) and binds the best fit, or births a new specialized brain (charter + seeds, digested, registered). Later calls return the same bound brain. Compositions the agent produces become nested sub-brains — specialization accrues. Use when an agent needs persistent domain knowledge it will query repeatedly.
---

# my-brain — the optimal brain configurator, as a callable skill

```bash
# first call: route-or-birth, then bind
python -m brain_agent.my_brain AGENT "what this agent is for" [--seed name=/path/file]

# every later call: the same brain, optionally queried
python -m brain_agent.my_brain AGENT --query "QUESTION"
```

From Python (or the RLM shell, where these are preloaded):

```python
b = await my_brain("research-dept", task="qualify leads against the ICP")
answer = await b.query("does ACME fit tier A?")
await compose("research-dept", "acme_review", {"verdict": text, "evidence": quotes})
# -> the agent's brain now contains compositions/acme_review/ — a nested
#    specialist brain. Subdirectories ARE sub-brains; nothing else needed.
```

Requires `HEAVEN_DATA_DIR` (registry, bindings ledger, newborn brain dirs) and
a heaven-routed model key for routing/digesting. Bindings persist in
`$HEAVEN_DATA_DIR/registry/agent_brains.json`.
