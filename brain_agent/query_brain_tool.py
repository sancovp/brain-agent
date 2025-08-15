
"""
Query Brain Tool - Interface for other agents to access brain knowledge.

This tool allows any agent to query a registered brain by name and get
synthesized instructions back.
"""

import re
from typing import Dict, Any, Optional

from heaven_base import BaseHeavenTool, ToolArgsSchema
from heaven_base.tools.registry_tool import registry_util_func
from .brain_agent import BrainAgent


async def query_brain_func(brain: str, query: str, persona_id: Optional[str] = None, persona_str: Optional[str] = None, mode_id: Optional[str] = None, mode_str: Optional[str] = None) -> str:
    """Query a brain and get instructions."""
    # Build composite prompt string
    persona_text = ""
    if persona_id:
        persona_text = f"PersonaID: {persona_id}"
    elif persona_str:
        persona_text = f"Persona: {persona_str}"

    mode_text = ""
    if mode_id:
        mode_text = f"ModeID: {mode_id}"
    elif mode_str:
        mode_text = f"Mode: {mode_str}"

    prompt_parts = [f"TargetBrain: {brain}"]
    if persona_text:
        prompt_parts.append(persona_text)
    if mode_text:
        prompt_parts.append(mode_text)
    prompt_parts.append(f"Query: {query}")
    composite_prompt = "\n\n".join(prompt_parts)

    # Verify brain exists
    brain_entry = registry_util_func(
        operation="get",
        registry_name="brain_configs",  # Fix registry name
        key=brain
    )
    if "not found" in brain_entry:
        return f"Error: Brain '{brain}' not found in registry."
        
    # Create and run brain agent
    brain_agent = BrainAgent()
    result = await brain_agent.query(composite_prompt)
    
    # Get the properly extracted content
    if hasattr(brain_agent, 'history') and brain_agent.history and brain_agent.history.agent_status:
        extracts = brain_agent.history.agent_status.extracted_content or {}
        if "instructions" in extracts:
            return extracts["instructions"]
    
    # Fallback to raw result if no extraction
    return result


class QueryBrainToolArgsSchema(ToolArgsSchema):
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
            'description': 'Query to ask the brain',
            'required': True
        },
        'persona_id': {
            'name': 'persona_id',
            'type': 'str',
            'description': 'Optional persona registry id to apply',
            'required': False
        },
        'persona_str': {
            'name': 'persona_str',
            'type': 'str',
            'description': 'Custom persona prompt block (ignored if persona_id given)',
            'required': False
        },
        'mode_id': {
            'name': 'mode_id',
            'type': 'str',
            'description': 'Optional mode registry id to apply',
            'required': False
        },
        'mode_str': {
            'name': 'mode_str',
            'type': 'str',
            'description': 'Custom mode prompt block (ignored if mode_id given)',
            'required': False
        }
    }

class QueryBrainTool(BaseHeavenTool):
    name = "QueryBrainTool"
    description = "Query a registered brain to get synthesized knowledge and instructions. Strategy and Help: Sequences should be performed to chain the context and aggregate a new context holistically. For example 1. QueryBrainTool with query \"I want to build a discord chatbot with HEAVEN... How do I do that?\", get the response and synthesize the broader and deeper context of what the new understanding -> 2.  Feed it back into BrainAgent. Note that BrainAgent is stateless so that this exact type of context chaining can be taken advantage of: reconfigure the presentation of the information so that the second query reads something like \"I'm building a discord chatbot using HEAVEN like this...<rest of information>\" + \"What should the code look like for each aspect you are aware of?\" (This particular way of phrasing and framing (\"for each aspect you are aware of\") is prompt engineered like that because of the fact that BrainAgent is stateless and the neurons are stateless and the query is prompting them at the same time. Always speak to BrainAgent like a person that has a mechanical memory)."
    args_schema = QueryBrainToolArgsSchema
    func = query_brain_func  # Reference the function here
    is_async = True

            
#         return result