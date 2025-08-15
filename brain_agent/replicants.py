import os
import json
from typing import List, Dict

from heaven_base import BaseHeavenAgentReplicant, HeavenAgentConfig, UnifiedChat, ProviderEnum, History

from .config import BrainConfig

class SynthesizerReplicant(BaseHeavenAgentReplicant):
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # Dummy config for initialization
            dummy_cfg = HeavenAgentConfig(
                name="SynthesizerReplicant",
                system_prompt="",
                tools=[],
                provider=ProviderEnum.OPENAI,
                model="gpt-4o",
                temperature=0.0,
            )
            cls._instance = cls(dummy_cfg, UnifiedChat(), History(messages=[]))
        return cls._instance

    def __init__(self, config: HeavenAgentConfig, unified_chat: UnifiedChat, history: History):
        super().__init__(config, unified_chat, history)
        self.neuron_paths: List[str] = []

    def load_brain(self, brain_cfg: BrainConfig):
        self.neuron_paths = []
        for root, _, files in os.walk(brain_cfg.directory):
            for fn in files:
                full = os.path.join(root, fn)
                self.neuron_paths.append(full)

    def cognize(self, context: str) -> List[str]:
        related: List[str] = []
        for path in self.neuron_paths:
            neuron_cfg = HeavenAgentConfig(
                name=f"Neuron[{os.path.basename(path)}]",
                system_prompt=(
                    "You are a NeuronAgent. Respond in JSON: {\"is_related\": true/false}."
                ),
                tools=[],
                provider=ProviderEnum.OPENAI,
                model=self.config.model,
                temperature=0.0,
                prompt_suffix_blocks=[f"path={path}"]
            )
            neuron = BaseHeavenAgentReplicant(neuron_cfg, UnifiedChat(), History(messages=[]))
            raw = neuron.run(f"cognize: {context}")
            try:
                data = json.loads(raw)
                if data.get("is_related"):
                    related.append(path)
            except Exception:
                pass
        return related

    def rerank(self, context: str, related: List[str]) -> List[str]:
        # Simple pass-through or implement ThinkTool reranking later
        return related

    def instruct(self, context: str, ranked: List[str]) -> Dict[str, str]:
        instructions: Dict[str, str] = {}
        for path in ranked:
            neuron_cfg = HeavenAgentConfig(
                name=f"Neuron[{os.path.basename(path)}]",
                system_prompt=(
                    "You are a NeuronAgent. Respond in JSON: {\"instructions\": \"...\"}."
                ),
                tools=[],
                provider=ProviderEnum.OPENAI,
                model=self.config.model,
                temperature=0.0,
                prompt_suffix_blocks=[f"path={path}"]
            )
            neuron = BaseHeavenAgentReplicant(neuron_cfg, UnifiedChat(), History(messages=[]))
            raw = neuron.run(f"instruct: {context}")
            try:
                data = json.loads(raw)
                instructions[path] = data.get("instructions", "")
            except Exception:
                instructions[path] = ""
        return instructions

class BrainAgentReplicant(BaseHeavenAgentReplicant):
    def __init__(self, brain_cfg: BrainConfig):
        # Dummy config to satisfy BaseHeavenAgentReplicant
        dummy_cfg = HeavenAgentConfig(
            name="BrainAgentReplicant",
            system_prompt="",
            tools=[],
            provider=ProviderEnum.OPENAI,
            model="gpt-4o",
            temperature=0.0,
        )
        super().__init__(dummy_cfg, UnifiedChat(), History(messages=[]))
        self.synth = SynthesizerReplicant.get_instance()
        self.synth.load_brain(brain_cfg)

    def run(self, user_query: str) -> str:
        # Compose agent-space prompt
        prompt = f"agent goal=Brain query={user_query}, iterations=5"
        return self.synth.run(prompt)
