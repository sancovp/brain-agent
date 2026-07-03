"""Brain Agent — neural-inspired knowledge synthesis on HEAVEN.

Everything runs on heaven (UnifiedChat + HeavenAgentConfig). There is no
provider-SDK path — heaven owns model routing and auth.

  BrainAgent / CognizeTool / InstructTool   the canonical flat brain (replicant)
  Brain / build_digests                     recursive brains-whose-neurons-are-brains
  RLM                                        recursive language model over a growing corpus
"""

# ── canonical heaven brain (the substrate everything extends) ─────────────────
from .config import BrainConfig
from .brain_agent import BrainAgent, register_brain, get_brain_config
from .tools import CognizeTool, InstructTool

# ── hierarchical brains + RLM (extend the canonical brain to depth N) ─────────
from .hierarchical import Brain, FileNeuron, build_digests
from .rlm import RLM, RLMResult

__all__ = [
    "BrainConfig",
    "BrainAgent", "register_brain", "get_brain_config",
    "CognizeTool", "InstructTool",
    "Brain", "FileNeuron", "build_digests",
    "RLM", "RLMResult",
]
