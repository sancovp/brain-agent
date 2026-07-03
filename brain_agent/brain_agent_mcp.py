#!/usr/bin/env python3
"""
Brain-Agent MCP Server
Exposes brain management and querying functionality through MCP protocol
"""

import logging
from typing import Optional, List
from fastmcp import FastMCP

# Import brain-agent functions
from .manager_tools import brain_manager_func, modes_and_personas_manager_func
from .query_brain_tool import query_brain_func

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP
mcp = FastMCP("Brain-Agent")

@mcp.tool()
def manage_brain(
    operation: str,
    brain_id: Optional[str] = None,
    name: Optional[str] = None,
    knowledge_source: Optional[str] = None,
    base_system_prompt: Optional[str] = None,
    allowed_personas: Optional[List[str]] = None,
    allowed_modes: Optional[List[str]] = None,
    neuron_source_type: Optional[str] = None,
    neuron_source: Optional[str] = None,
    chunk_max: Optional[int] = None
) -> str:
    """
    Manage brain configurations - create, read, update, delete brains.
    
    Args:
        operation: CRUD operation (add, get, update, delete, list_keys, get_all)
        brain_id: Unique brain identifier
        name: Human-readable brain name
        knowledge_source: Identifier for knowledge source
        base_system_prompt: Base system prompt for the brain
        allowed_personas: List of compatible persona IDs
        allowed_modes: List of compatible mode IDs
        neuron_source_type: How to load neurons ('registry_keys', 'entire_registry', 'directory', 'file')
        neuron_source: Registry name, directory path, or file path
        chunk_max: Maximum characters per neuron chunk
        
    Returns:
        Result of the brain management operation
    """
    return brain_manager_func(
        operation=operation,
        brain_id=brain_id,
        name=name,
        knowledge_source=knowledge_source,
        base_system_prompt=base_system_prompt,
        allowed_personas=allowed_personas,
        allowed_modes=allowed_modes,
        neuron_source_type=neuron_source_type,
        neuron_source=neuron_source,
        chunk_max=chunk_max
    )

@mcp.tool()
def manage_persona_or_mode(
    entity_type: str,
    operation: str,
    entity_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    prompt_block: Optional[str] = None
) -> str:
    """
    Manage personas and modes - create, read, update, delete personas/modes.
    
    Args:
        entity_type: Type of entity ('persona' or 'mode')
        operation: CRUD operation (add, get, update, delete, list_keys, get_all)
        entity_id: Unique entity identifier
        name: Human-readable entity name
        description: Description of the persona or mode
        prompt_block: Prompt block text for the persona or mode
        
    Returns:
        Result of the persona/mode management operation
    """
    return modes_and_personas_manager_func(
        entity_type=entity_type,
        operation=operation,
        entity_id=entity_id,
        name=name,
        description=description,
        prompt_block=prompt_block
    )

@mcp.tool()
async def query_brain(
    brain: str,
    query: str,
    persona_id: Optional[str] = None,
    persona_str: Optional[str] = None,
    mode_id: Optional[str] = None,
    mode_str: Optional[str] = None
) -> str:
    """
    Query a brain and get intelligent responses.
    
    Args:
        brain: Brain ID to query
        query: Question or prompt to ask the brain
        persona_id: ID of registered persona to use
        persona_str: Custom persona description
        mode_id: ID of registered mode to use
        mode_str: Custom mode description
        
    Returns:
        Brain's response to the query
    """
    return await query_brain_func(
        brain=brain,
        query=query,
        persona_id=persona_id,
        persona_str=persona_str,
        mode_id=mode_id,
        mode_str=mode_str
    )

def main():
    """Entry point for brain-agent-server console script"""
    # Initialize brain registries on startup
    try:
        from .seed_brain_registries import main as seed_main
        seed_main()
        logger.info("✅ Brain registries initialized")
    except Exception as e:
        logger.warning(f"Registry seeding failed: {e}")
    
    mcp.run()

if __name__ == "__main__":
    main()