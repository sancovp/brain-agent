"""
Brain Agent - Neural-inspired knowledge retrieval system for heaven-base.
"""

from .config import BrainConfig
from .brain_agent import BrainAgent, register_brain, get_brain_config
from .tools import CognizeTool, InstructTool
from .query_brain_tool import QueryBrainTool
from .replicants import SynthesizerReplicant, BrainAgentReplicant
from .manager_tools import BrainManagerTool, ModesAndPersonasManagerTool

__all__ = [
    "BrainConfig",
    "BrainAgent", 
    "register_brain",
    "get_brain_config",
    "CognizeTool",
    "InstructTool",
    "QueryBrainTool",
    "SynthesizerReplicant",
    "BrainAgentReplicant",
    "BrainManagerTool",
    "ModesAndPersonasManagerTool"
]