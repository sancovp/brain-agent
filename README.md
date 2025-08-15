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