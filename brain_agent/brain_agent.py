"""
Brain Agent Implementation - A neural-inspired knowledge retrieval system.

This module implements the BrainAgent class that extends BaseHeavenAgentReplicant,
along with the necessary tools and utilities for document-based knowledge retrieval.
"""

import os
import json
import uuid
from typing import List, Dict, Any, Optional, Type

from heaven_base import BaseHeavenAgentReplicant, HeavenAgentConfig, BaseHeavenTool, UnifiedChat, ProviderEnum
from heaven_base.tools.registry_tool import registry_util_func
from .config import BrainConfig

# Import tools after they are defined to avoid circular imports
from .tools import CognizeTool, InstructTool

# System prompt for brain agent
BRAIN_AGENT_SYSTEM_PROMPT = """You are BrainAgent, a neural-inspired knowledge retrieval system.

When given a query:
1. Use CognizeTool to identify relevant neurons in the specified brain
2. Analyze the relevance results to determine which neurons are most important
3. Use InstructTool to generate instructions from the most relevant neurons
4. Synthesize all instructions into a coherent response

Always return your instructions in a dedicated fence format:

```instructions
Your synthesized instructions here
```
block.

Focus on providing actionable, concrete guidance based on the knowledge contained in the activated neurons.
"""

class BrainAgent(BaseHeavenAgentReplicant):
    """
    Neural-inspired knowledge retrieval agent that activates relevant "neurons" (documents)
    for a given query and synthesizes instructions from them.
    """
    
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        """Return the default configuration for BrainAgent."""
        return HeavenAgentConfig(
            name="BrainAgent",
            system_prompt=BRAIN_AGENT_SYSTEM_PROMPT,
            tools=[CognizeTool, InstructTool],
            # provider=ProviderEnum.GOOGLE,
            # model="gemini-2.5-pro-preview-06-05",
            # model="gemini-2.5-flash-preview-05-20",
            provider=ProviderEnum.OPENAI,
            model="gpt-4.1-mini",
            temperature=0.3,
            additional_kws=["instructions"],
            additional_kw_instructions="instructions: Use this kw when writing Synthesized instructions from relevant neurons, after using InstructTool. All of the `instructions` kws you produce will be concatenated at the end of the process by another program, so you can output multiple `instructions` blocks AFTER the InstructTool phase -- as many `instructions` blocks as required. Only use XML tag blocks. For example, <instructions>{{your text}}</instructions>",
            prompt_suffix_blocks=[]
        )

    def __init__(self, 
                 config: Optional[HeavenAgentConfig] = None,
                 chat: Optional[UnifiedChat] = None,
                 history_id: Optional[str] = None,
                 orchestrator: bool = False,
                 system_prompt_suffix: Optional[str] = None,
                 additional_tools: Optional[List[Type[BaseHeavenTool]]] = None,
                 remove_agents_config_tools: bool = False,
                 duo_enabled: bool = False,
                 run_on_langchain: bool = True,
                 adk: bool = False,
                 use_uni_api: bool = False):
        """
        Initialize a BrainAgent.
        
        Note: The brain name should be passed in the query itself, not stored on the agent.
        The agent is stateless and can query different brains.
        """
        # Just call parent - it will use get_default_config() if config is None
        super().__init__(
            config=config,
            chat=chat,
            history_id=history_id,
            orchestrator=orchestrator,
            system_prompt_suffix=system_prompt_suffix,
            additional_tools=additional_tools,
            remove_agents_config_tools=remove_agents_config_tools,
            duo_enabled=duo_enabled,
            run_on_langchain=run_on_langchain,
            adk=adk,
            use_uni_api=use_uni_api
        )
    # def get_brain_instructions(self) -> str:
    #     extracts = self.history.agent_status.extracted_content or {}
    #     ordered  = [
    #         extracts[key] for key in sorted(extracts)
    #         if key.startswith("instructions")
    #     ]
    #     return "\n\n".join(ordered)  # final concatenated guidance
    def get_brain_instructions(self) -> str:
        extracts = self.history.agent_status.extracted_content or {}

        unique = []
        seen = set()

        for key in sorted(extracts):
            if key.startswith("instructions"):
                txt = extracts[key]
                if txt not in seen:
                    unique.append(txt)
                    seen.add(txt)

        return "\n\n".join(unique)



    async def query(self, query_text: str) -> str:
        """
        Query the brain with the given text.
        
        Args:
            query_text: The query to process (should include brain name)
            
        Returns:
            The instructions generated from relevant neurons
        """
        goal = f"""agent goal=Process brain query: {query_text}\n\nUse CognizeTool to get a map of which neurons in the brain are related to this query, then rank them using your best estimation of which ones are most relevant according to the reasoning strings you get back from CognizeTool. Then, call only those most relevant neurons with the InstructTool with the same query as before. Then write however many 'instructions' XML-tag blocks are required. They can be as long as they need to be and you can write multiple instructions blocks  over multiple turns (one per iteration once you start, for as many iterations as are left after you receive the InstructTool output), iterations=8"""
        
        result = await self.run(goal)
        history_id = result['history_id']
        history_id_str = f"history_id: {history_id}\n\n"
        # If the agent has extracted instructions, return those
        if hasattr(self, 'history') and self.history and self.history.agent_status:
            
            result = self.get_brain_instructions()
            history_id_str += result
        # Otherwise return the raw result
        return str(history_id_str)

def get_brain_config(brain_name: str) -> BrainConfig:

    entry = registry_util_func(operation="get",

                               registry_name="brain_configs",

                               key=brain_name)

    if isinstance(entry, dict) and entry.get("value_dict"):

        return BrainConfig(**entry["value_dict"])

    raise KeyError(f"Brain '{brain_name}' not found")

def register_brain(directory: str, brain_name: str, chunk_size: int = -1) -> None:
    """Register a new brain in the brain_configs registry."""
    try:
        if not os.path.isdir(directory):
            raise ValueError(f"Directory does not exist: {directory}")
        
        # Convert to relative path if within HEAVEN_DATA_DIR
        storage_directory = directory
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'heaven-base'))
            from heaven_base.utils.get_env_value import EnvConfigUtil
            heaven_data_dir = EnvConfigUtil.get_heaven_data_dir()
            
            # If directory is within HEAVEN_DATA_DIR, store as relative path
            if directory.startswith(heaven_data_dir + os.sep):
                storage_directory = os.path.relpath(directory, heaven_data_dir)
                print(f"DEBUG: Storing brain directory as relative path: {storage_directory}")
        except Exception as e:
            print(f"DEBUG: Could not resolve HEAVEN_DATA_DIR, storing absolute path: {e}")
        
        # List registries
        result = registry_util_func(operation="list_registries")
        
        if "brain_configs" not in result:
            result = registry_util_func(
                operation="create_registry",
                registry_name="brain_configs"
            )
            if "Error:" in result:
                raise RuntimeError(f"Failed to create registry: {result}")
        
        # Add the brain
        result = registry_util_func(
            operation="add",
            registry_name="brain_configs",
            key=brain_name,
            value_dict={
                "directory": storage_directory,
                "brain_name": brain_name,
                "chunk_size": chunk_size
            }
        )
        
        if "added to registry" in result:
            print(f"Brain '{brain_name}' registered successfully with directory: {directory}")
        else:
            raise RuntimeError(f"Failed to register brain: {result}")
            
    except Exception as e:
        print(f"ERROR in register_brain: {e}")
        raise