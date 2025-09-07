"""
Brain Agent Tools - Corrected implementation using HEAVEN's prompt_suffix_blocks
Following HEAVEN's pattern: Never override _run or _arun, always use func attribute
"""

import ast
import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from heaven_base import BaseHeavenTool, ToolArgsSchema, HeavenAgentConfig, UnifiedChat, ProviderEnum
from heaven_base.tools.registry_tool import registry_util_func
from .config import BrainConfig
from . import logger

def _parse_composite_query(query: str) -> Tuple[str, Optional[str], Optional[str], str]:
    """
    Parse composite query format to extract brain, persona_id, mode_id, and actual query.
    
    Expected format:
    TargetBrain: brain_name
    PersonaID: persona_id  
    ModeID: mode_id
    Query: actual_query
    
    Returns: (brain_name, persona_id, mode_id, actual_query)
    """
    brain_name = None
    persona_id = None
    mode_id = None
    actual_query = query
    
    # Parse TargetBrain
    brain_match = re.search(r'TargetBrain:\s*([^\n]+)', query)
    if brain_match:
        brain_name = brain_match.group(1).strip()
    
    # Parse PersonaID
    persona_match = re.search(r'PersonaID:\s*([^\n]+)', query)
    if persona_match:
        persona_id = persona_match.group(1).strip()
    
    # Parse ModeID
    mode_match = re.search(r'ModeID:\s*([^\n]+)', query)
    if mode_match:
        mode_id = mode_match.group(1).strip()
    
    # Parse Query (everything after "Query:")
    query_match = re.search(r'Query:\s*(.*)', query, re.DOTALL)
    if query_match:
        actual_query = query_match.group(1).strip()
    
    return brain_name, persona_id, mode_id, actual_query

def _build_enhanced_prompt_suffix_blocks(neuron_path: str, persona_id: Optional[str], mode_id: Optional[str]) -> List[str]:
    """Build prompt_suffix_blocks with neuron path and persona/mode registry lookups."""
    blocks = []
    
    # Handle different neuron types
    if neuron_path.startswith("registry_key:"):
        # Format: registry_key:registry_name:key
        _, registry_name, key = neuron_path.split(":", 2)
        blocks.append(f'registry_heaven_variable={{"registry_name": "{registry_name}", "key": "{key}"}}')
    elif neuron_path.startswith("registry_entire:"):
        # Format: registry_entire:registry_name
        _, registry_name = neuron_path.split(":", 1)
        blocks.append(f'registry_heaven_variable={{"registry_name": "{registry_name}"}}')
    elif neuron_path.startswith("file_chunk:"):
        # Format: file_chunk:file_path:start:end
        parts = neuron_path.split(":", 3)
        file_path, start, end = parts[1], int(parts[2]), int(parts[3])
        blocks.append(f"file_chunk={file_path}:{start}:{end}")
    else:
        # Regular file path
        blocks.append(f"path={neuron_path}")
    
    if persona_id:
        blocks.append(f'registry_heaven_variable={{"registry_name": "brain_personas_registry", "key": "{persona_id}"}}')
    
    if mode_id:
        blocks.append(f'registry_heaven_variable={{"registry_name": "brain_modes_registry", "key": "{mode_id}"}}')
    
    return blocks

# Filter functions remain the same
def _should_include_file(file_path: str) -> bool:
    """Determine if a file should be included as a neuron."""
    basename = os.path.basename(file_path)
    if basename.startswith('.'):
        return False
    if basename.endswith('.pyc'):
        return False
    if '__pycache__' in file_path:
        return False
    return True

def _load_neurons(brain_cfg: BrainConfig) -> List[str]:
    """Load neurons based on brain configuration."""
    neurons = []
    
    if brain_cfg.neuron_source_type == "directory":
        # Original file-based loading
        directory = brain_cfg.neuron_source
        logger.debug_print(f"Loading neurons from directory: {directory}")
        for root, _, files in os.walk(directory):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if _should_include_file(file_path):
                    neurons.append(file_path)
                    
    elif brain_cfg.neuron_source_type == "registry_keys":
        # Each registry key becomes one neuron
        registry_name = brain_cfg.neuron_source
        logger.debug_print(f"Loading neurons from registry keys: {registry_name}")
        result = registry_util_func("get_all", registry_name=registry_name)
        
        if "Items in registry" in result:
            try:
                # Parse registry data (Python dict format)
                start_idx = result.find("{")
                if start_idx != -1:
                    dict_str = result[start_idx:]
                    # Use ast.literal_eval to safely parse Python dict format
                    registry_data = ast.literal_eval(dict_str)
                    
                    # Each key becomes a neuron identifier
                    for key in registry_data.keys():
                        neurons.append(f"registry_key:{registry_name}:{key}")
                        
            except Exception as e:
                logger.debug_print(f"Failed to parse registry data: {e}")
                
    elif brain_cfg.neuron_source_type == "entire_registry":
        # Entire registry as one neuron
        registry_name = brain_cfg.neuron_source
        logger.debug_print(f"Loading entire registry as neuron: {registry_name}")
        neurons.append(f"registry_entire:{registry_name}")
        
    elif brain_cfg.neuron_source_type == "file":
        # Single file, potentially chunked
        file_path = brain_cfg.neuron_source
        logger.debug_print(f"Loading neurons from file: {file_path}")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Chunk if needed
                if len(content) > brain_cfg.chunk_max:
                    # Simple chunking by character count
                    chunk_size = brain_cfg.chunk_max
                    for i in range(0, len(content), chunk_size):
                        chunk_end = min(i + chunk_size, len(content))
                        neurons.append(f"file_chunk:{file_path}:{i}:{chunk_end}")
                else:
                    neurons.append(file_path)
                    
            except Exception as e:
                logger.debug_print(f"Failed to read file {file_path}: {e}")
    
    logger.debug_print(f"Total neurons loaded: {len(neurons)}")
    return neurons

# CognizeTool implementation
async def cognize_func(brain: str, query: str) -> Dict[str, Any]:
    """Find relevant neurons for the given query in the specified brain."""
    # Parse composite query to extract persona/mode info
    parsed_brain, persona_id, mode_id, actual_query = _parse_composite_query(query)
    
    # Use parsed brain name if available, otherwise fall back to parameter
    brain_name = parsed_brain if parsed_brain else brain
    
    # Get brain config from registry
    brain_entry = registry_util_func(
        operation="get",
        registry_name="brain_configs",
        key=brain_name
    )
    
    # Check if brain was found
    if "not found" in brain_entry:
        raise ValueError(f"Brain '{brain_name}' not found in registry")
    # Parse the dict from the string response
    try:
        import ast
        dict_str = brain_entry.split(": ", 1)[1]
        brain_cfg_dict = ast.literal_eval(dict_str)
    except:
        raise ValueError(f"Failed to parse brain config for '{brain_name}'")

    brain_cfg = BrainConfig(**brain_cfg_dict)
    
    # Resolve directory path relative to HEAVEN_DATA_DIR
    try:
        # Import heaven-base to get HEAVEN_DATA_DIR
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'heaven-base'))
        from heaven_base.utils.get_env_value import EnvConfigUtil
        heaven_data_dir = EnvConfigUtil.get_heaven_data_dir()
        
        # If directory is relative, resolve it to HEAVEN_DATA_DIR
        if not os.path.isabs(brain_cfg.directory):
            resolved_directory = os.path.join(heaven_data_dir, brain_cfg.directory)
        else:
            resolved_directory = brain_cfg.directory
    except Exception as e:
        logger.debug_print(f"Could not resolve HEAVEN_DATA_DIR, using directory as-is: {e}")
        resolved_directory = brain_cfg.directory
    
    # Load all neurons based on brain config
    neuron_paths = _load_neurons(brain_cfg)
    if not neuron_paths:
        return {"relevant_neurons": [], "reasoning": {}}
        
    # Create unified chat
    unified_chat = UnifiedChat.create(
        provider=ProviderEnum.GOOGLE,
        model="gemini-2.0-flash",
        temperature=0.2,
        max_tokens=500
    )
    
    # Prepare message lists for batch processing
    message_lists = []
    for path in neuron_paths:
        # Build enhanced prompt_suffix_blocks with persona/mode registry lookups
        prompt_suffix_blocks = _build_enhanced_prompt_suffix_blocks(path, persona_id, mode_id)
        
        # Create a HeavenAgentConfig with the neuron file and persona/mode injected
        neuron_config = HeavenAgentConfig(
            name="NeuronRelevanceAgent",
            system_prompt="You are a NeuronAgent. Determine if your neuron content is related to the query. Respond with a JSON object with two keys: 'related_to' (boolean) and 'reasoning' (string explaining why).\n\n<neuron content>",
            tools=[],
            prompt_suffix_blocks=prompt_suffix_blocks
        )
        
        # Get the rendered system prompt with file contents and persona/mode
        rendered_prompt = neuron_config.get_system_prompt()
        
        final_prompt = rendered_prompt + "</neuron content>"
        messages = [
            SystemMessage(content=final_prompt),
            HumanMessage(content=f"Query: {actual_query}")
        ]
        message_lists.append(messages)
        
    # Process all neurons in parallel
    import time
    start_time = time.time()
    logger.debug_print(f"🚀 Starting batch processing of {len(message_lists)} neurons...")
    responses = await unified_chat.abatch(message_lists)
    batch_time = time.time() - start_time
    logger.debug_print(f"✅ Batch completed in {batch_time:.2f} seconds for {len(responses)} responses")
    for i, response in enumerate(responses):
        logger.debug_print(f"Response {i}: {response}")
        logger.debug_print(f"Response content: {response.content[:200]}...")  # First 200 chars

    # Process responses
    relevant_neurons = []
    reasoning_map = {}

    for i, response in enumerate(responses):
        neuron_path = neuron_paths[i]
        try:
            # Catch it
            content = response.content
            logger.debug_print(f"Parsing content for {neuron_path}: {content[:100]}...")
            # Try to parse JSON response
           

            # Strip markdown code blocks if present
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3].strip()  # Remove ```json and ```
            elif content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()  # Remove ``` and ```

           
            data = json.loads(content)
            
            # Check if neuron is related
            if data.get("related_to", False):
                relevant_neurons.append(neuron_path)
                reasoning_map[neuron_path] = data.get("reasoning", "No reasoning provided")
                logger.debug_print(f"Neuron {neuron_path} is RELATED")
            else:
                logger.debug_print(f"Neuron {neuron_path} is NOT RELATED")
        except (json.JSONDecodeError, AttributeError) as e:
            # Skip malformed responses
            logger.debug_print(f"Failed to parse response for {neuron_path}: {e}")
            continue
            
    return {
        "relevant_neurons": relevant_neurons,
        "reasoning": reasoning_map
    }

class CognizeToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'brain': {
            'name': 'brain',
            'type': 'str',
            'description': 'Registered brain name',
            'required': True
        },
        'query': {
            'name': 'query',
            'type': 'str',
            'description': 'Query to find relevant neurons for',
            'required': True
        }
    }

class CognizeTool(BaseHeavenTool):
    name = "CognizeTool"
    description = "Run Brain inference on a query to return the cluster of related neurons to then map"
    args_schema = CognizeToolArgsSchema
    func = cognize_func  # Reference the function
    is_async = True

# InstructTool implementation
async def instruct_func(brain: str, query: str, neurons: List[str], reasoning: Dict[str, str]) -> Dict[str, Any]:
    """Generate instructions from the given relevant neurons."""
    if not neurons:
        return {
            "instructions": {},
            "combined": "No relevant neurons found for this query."
        }
    
    # Parse composite query to extract persona/mode info
    parsed_brain, persona_id, mode_id, actual_query = _parse_composite_query(query)
        
    # Create unified chat
    unified_chat = UnifiedChat.create(
        provider=ProviderEnum.GOOGLE,
        model="gemini-2.0-flash",
        temperature=0.2
    )
    
    # Prepare message lists for batch processing
    message_lists = []
    for path in neurons:
        neuron_reasoning = reasoning.get(path, "This neuron was deemed relevant to your query.")
        
        # Build enhanced prompt_suffix_blocks with persona/mode registry lookups
        prompt_suffix_blocks = _build_enhanced_prompt_suffix_blocks(path, persona_id, mode_id)
        
        # Create a HeavenAgentConfig with the neuron file and persona/mode injected
        neuron_config = HeavenAgentConfig(
            name="NeuronInstructionAgent",
            system_prompt="You are a NeuronAgent. Generate instructions based on your neuron content and the query.\n\n<neuron content>",
            tools=[],
            prompt_suffix_blocks=prompt_suffix_blocks
        )
        
        # Get the rendered system prompt with file contents and persona/mode
        rendered_prompt = neuron_config.get_system_prompt()
        final_prompt = rendered_prompt + "</neuron content>"
        
        # Create messages for this neuron
        messages = [
            SystemMessage(content=final_prompt),
            HumanMessage(content=f"""Query: {actual_query}\n\n
            How does this query relate to the content you are the Neuron for? Focus on instructions: what guidance would you give for implementing or addressing this Query based on your neuron content?\n\nReasoning for why you're being asked: {neuron_reasoning}\n\nDont hesitate to make clarifications based on my reasoning. I want you to be completely honest. I want to make sure that we do a superb job and get this right. Respond with clear, actionable instructions in a JSON object with an 'instructions' key.""")
        ]
        message_lists.append(messages)
        
    # Process all neurons in parallel
    import time
    start_time = time.time()
    logger.debug_print(f"🚀 Starting InstructTool batch processing of {len(message_lists)} neurons...")
    responses = await unified_chat.abatch(message_lists)
    batch_time = time.time() - start_time
    logger.debug_print(f"✅ InstructTool batch completed in {batch_time:.2f} seconds for {len(responses)} responses")
    
    # Process responses
    instruction_map = {}
    combined_instructions = []
    
    for i, response in enumerate(responses):
        neuron_path = neurons[i]
        try:
            # Try to parse JSON response
           
            # Catch it
            content = response.content
            logger.debug_print(f"Parsing content for {neuron_path}: {content[:100]}...")
            # Try to parse JSON response
           

            # Strip markdown code blocks if present
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3].strip()  # Remove ```json and ```
            elif content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()  # Remove ``` and ```

           
          
            data = json.loads(content)
            
            # Get instructions from response
            instructions = data.get("instructions", "")
            if instructions:
                instruction_map[neuron_path] = instructions
                neuron_name = os.path.basename(neuron_path)
                combined_instructions.append(f"From {neuron_name}:\n{instructions}")
        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, use the raw content as instructions
            instruction_map[neuron_path] = response.content
            neuron_name = os.path.basename(neuron_path)
            combined_instructions.append(f"From {neuron_name}:\n{response.content}")
            
    # # Combine all instructions
    # combined = "\n\n".join(combined_instructions)
    
    return {
        "instructions": instruction_map
    }

class InstructToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'brain': {
            'name': 'brain',
            'type': 'str',
            'description': 'Registered brain name',
            'required': True
        },
        'query': {
            'name': 'query',
            'type': 'str',
            'description': 'Original query to generate instructions for',
            'required': True
        },
        'neurons': {
            'name': 'neurons',
            'type': 'list',
            'description': 'List of relevant neuron file paths',
            'required': True
        },
        'reasoning': {
            'name': 'reasoning',
            'type': 'dict',
            'description': 'Dict mapping neuron paths to reasoning',
            'required': True
        }
    }
    
class InstructTool(BaseHeavenTool):
    name = "InstructTool"
    description = "Generate instructions from relevant neurons based on a query"
    args_schema = InstructToolArgsSchema
    func = instruct_func  # Reference the function
    is_async = True
