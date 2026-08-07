"""Brain Agent — neural-inspired knowledge synthesis on HEAVEN.

  BrainAgent / CognizeTool / InstructTool   the canonical flat brain (replicant)
  Brain / build_digests                     recursive brains-whose-neurons-are-brains
  RLM                                       recursive language model over a growing corpus
  Neuron / Synthesizer / ComposedBrain      the SDK a shell agent composes at runtime
  Orchestrator / PyShell / Kernel           an agent with one persistent runtime
                                            context in which it builds brain systems

Imports are LAZY (PEP 562). Everything that talks to a model needs heaven, but
the shell and the kernel do not — and eagerly importing the package used to drag
heaven in regardless, so `from brain_agent.kernel import Kernel` failed on any
box without a working heaven install. Now each name pulls in only its own
module: PyShell and Kernel import with the standard library alone.
"""
import importlib
from typing import TYPE_CHECKING

__version__ = "0.13.0"

_EXPORTS = {
    # canonical heaven brain (the substrate everything extends) — needs heaven
    "BrainConfig": "config",
    "BrainAgent": "brain_agent", "register_brain": "brain_agent",
    "get_brain_config": "brain_agent",
    "CognizeTool": "tools", "InstructTool": "tools",
    # hierarchical brains + RLM — need heaven
    "Brain": "hierarchical", "FileNeuron": "hierarchical",
    "build_digests": "hierarchical",
    "RLM": "rlm", "RLMResult": "rlm",
    # the composable SDK — needs heaven only when a primitive is CALLED
    "Neuron": "sdk", "Synthesizer": "sdk", "fanout": "sdk", "sub_llm": "sdk",
    "from_dir": "sdk", "open_all_router": "sdk", "top_k_router": "sdk",
    "threshold_router": "sdk",
    # the shell + kernel — stdlib only
    "PyShell": "shell", "default_policy": "shell",
    "Orchestrator": "orchestrator", "shell_func": "orchestrator",
    "ContextDict": "context",
    "record": "bandit", "blend": "bandit",
    "my_brain": "my_brain", "compose": "my_brain", "bound_brain": "my_brain",
    "ActivationGraph": "neuro", "Membrane": "neuro",
    "Kernel": "kernel", "serve": "kernel",
    # call-graph tracing — stdlib to record, networkx only to load
    "load_run": "trace", "text_tree": "trace", "load_records": "trace",
}

# `Brain` is the directory-walking brain; the composable one is ComposedBrain.
_ALIASES = {"ComposedBrain": ("sdk", "Brain")}

# Submodules reachable as attributes. Without this, lazy __getattr__ shadows the
# normal submodule lookup and `from brain_agent import trace` raises.
_SUBMODULES = ("config", "brain_agent", "tools", "hierarchical", "rlm",
               "sdk", "shell", "kernel", "trace", "orchestrator", "context", "bandit", "my_brain", "neuro")

__all__ = sorted(list(_EXPORTS) + list(_ALIASES) + list(_SUBMODULES) + ["__version__"])


def __getattr__(name):
    if name in _SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name in _ALIASES:
        mod, attr = _ALIASES[name]
    elif name in _EXPORTS:
        mod, attr = _EXPORTS[name], name
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), attr)
    globals()[name] = value          # cache: pay the import once
    return value


def __dir__():
    return __all__


if TYPE_CHECKING:  # editors/type-checkers still see the real names
    from .config import BrainConfig
    from .brain_agent import BrainAgent, register_brain, get_brain_config
    from .tools import CognizeTool, InstructTool
    from .hierarchical import Brain, FileNeuron, build_digests
    from .rlm import RLM, RLMResult
    from .sdk import (Neuron, Synthesizer, Brain as ComposedBrain, fanout,
                      sub_llm, from_dir, open_all_router, top_k_router,
                      threshold_router)
    from .shell import PyShell, default_policy
    from .orchestrator import Orchestrator
    from .context import ContextDict
    from .kernel import Kernel, serve
    from .trace import load_run, text_tree, load_records
