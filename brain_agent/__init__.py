"""Brain Agent — hierarchical brains, RLM, and the judge/fill server.

The hierarchical/RLM/server layer is dependency-light (anthropic + fastapi +
uvicorn). The LEGACY heaven-based agent (BrainAgent, the *Tool classes, the
replicants) needs the optional 'heaven' extra: pip install brain-agent[heaven].
Its symbols are imported lazily and are simply absent if heaven isn't present,
so 'from brain_agent.rlm import RLM' never drags in the heaven stack.
"""

# ── dependency-light core (always available) ──────────────────────────────────
from .config import BrainConfig
from .hierarchical import Brain, FileNeuron, build_digests
from .rlm import RLM, RLMResult

__all__ = [
    "BrainConfig",
    "Brain", "FileNeuron", "build_digests",
    "RLM", "RLMResult",
]

# ── legacy heaven-based agent (optional; requires the 'heaven' extra) ─────────
try:
    from .brain_agent import BrainAgent, register_brain, get_brain_config
    from .tools import CognizeTool, InstructTool
    from .query_brain_tool import QueryBrainTool
    from .replicants import SynthesizerReplicant, BrainAgentReplicant
    from .manager_tools import BrainManagerTool, ModesAndPersonasManagerTool
    __all__ += [
        "BrainAgent", "register_brain", "get_brain_config",
        "CognizeTool", "InstructTool", "QueryBrainTool",
        "SynthesizerReplicant", "BrainAgentReplicant",
        "BrainManagerTool", "ModesAndPersonasManagerTool",
    ]
except ImportError:
    # heaven-framework not installed — the legacy agent is unavailable, but the
    # hierarchical/RLM/server layer works fine without it.
    pass
