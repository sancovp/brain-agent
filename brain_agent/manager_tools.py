
import json
from typing import Optional, List, Dict, Any

from heaven_base import ToolArgsSchema
from heaven_base.tools.registry_tool import registry_util_func

# --- BrainManagerTool ---

def brain_manager_func(
    operation: str,
    brain_id: Optional[str] = None,
    name: Optional[str] = None,
    knowledge_source: Optional[str] = None,
    base_system_prompt: Optional[str] = None,
    allowed_personas: Optional[List[str]] = None,
    allowed_modes: Optional[List[str]] = None
) -> str:
    """Manages CRUD operations for the brain_configs_registry."""
    registry_name = "brain_configs"
    
    # The dictionary is assembled inside the function from flat arguments.
    value_dict = {}
    if name is not None:
        value_dict['name'] = name
    if knowledge_source is not None:
        value_dict['knowledge_source'] = knowledge_source
    if base_system_prompt is not None:
        value_dict['base_system_prompt'] = base_system_prompt
    if allowed_personas is not None:
        value_dict['allowed_personas'] = allowed_personas
    if allowed_modes is not None:
        value_dict['allowed_modes'] = allowed_modes

    return registry_util_func(
        registry_name=registry_name,
        operation=operation,
        key=brain_id,
        value_dict=value_dict if value_dict else None
    )

class BrainManagerToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'operation': {
            'name': 'operation', 'type': 'str', 'required': True,
            'description': "CRUD operation: add, get, update, delete, list_keys, get_all"
        },
        'brain_id': {
            'name': 'brain_id', 'type': 'str', 'required': False,
            'description': "Unique ID for the brain (used for get, update, delete)."
        },
        'name': {
            'name': 'name', 'type': 'str', 'required': False,
            'description': "Human-readable name for the brain."
        },
        'knowledge_source': {
            'name': 'knowledge_source', 'type': 'str', 'required': False,
            'description': "Identifier for the brain's knowledge source."
        },
        'base_system_prompt': {
            'name': 'base_system_prompt', 'type': 'str', 'required': False,
            'description': "The base system prompt for the brain."
        },
        'allowed_personas': {
            'name': 'allowed_personas', 'type': 'List[str]', 'required': False,
            'description': "List of persona IDs compatible with this brain."
        },
        'allowed_modes': {
            'name': 'allowed_modes', 'type': 'List[str]', 'required': False,
            'description': "List of mode IDs compatible with this brain."
        }
    }

class BrainManagerTool:
    name = "BrainManagerTool"
    description = "Exposes all CRUD operations for the brain_configs_registry."
    args_schema = BrainManagerToolArgsSchema
    func = brain_manager_func

# --- ModesAndPersonasManagerTool ---

def modes_and_personas_manager_func(
    entity_type: str,
    operation: str,
    entity_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    prompt_block: Optional[str] = None
) -> str:
    """Manages CRUD operations for persona and mode registries."""
    if entity_type.lower() == 'persona':
        registry_name = "brain_personas_registry"
    elif entity_type.lower() == 'mode':
        registry_name = "brain_modes_registry"
    else:
        return "Error: entity_type must be 'persona' or 'mode'."

    value_dict = {}
    if name is not None:
        value_dict['name'] = name
    if description is not None:
        value_dict['description'] = description
    if prompt_block is not None:
        value_dict['prompt_block'] = prompt_block
        
    # For public-facing list/get, we must strip sensitive data.
    if operation == 'get_all':
         full_result_str = registry_util_func(registry_name=registry_name, operation='get_all')
         try:
             full_result = json.loads(full_result_str)
             public_result = {
                 key: {'name': value.get('name'), 'description': value.get('description')}
                 for key, value in full_result.items()
             }
             return json.dumps(public_result, indent=2)
         except (json.JSONDecodeError, AttributeError):
             return full_result_str # Return raw if parsing fails
    
    if operation == 'get':
        full_result_str = registry_util_func(registry_name=registry_name, operation='get', key=entity_id)
        try:
            full_result = json.loads(full_result_str)
            # Return only public-facing data
            return json.dumps({
                'id': entity_id,
                'name': full_result.get('name'),
                'description': full_result.get('description')
            }, indent=2)
        except (json.JSONDecodeError, AttributeError):
            return full_result_str # Return raw if parsing fails

    # For add, update, delete, list_keys, call the util func directly
    return registry_util_func(
        registry_name=registry_name,
        operation=operation,
        key=entity_id,
        value_dict=value_dict if value_dict else None
    )

class ModesAndPersonasManagerToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'entity_type': {
            'name': 'entity_type', 'type': 'str', 'required': True,
            'description': "The type of entity to manage: 'persona' or 'mode'."
        },
        'operation': {
            'name': 'operation', 'type': 'str', 'required': True,
            'description': "CRUD operation: add, get, update, delete, list_keys, get_all"
        },
        'entity_id': {
            'name': 'entity_id', 'type': 'str', 'required': False,
            'description': "Unique ID for the persona or mode."
        },
        'name': {
            'name': 'name', 'type': 'str', 'required': False,
            'description': "Human-readable name for the entity."
        },
        'description': {
            'name': 'description', 'type': 'str', 'required': False,
            'description': "A brief description of the entity."
        },
        'prompt_block': {
            'name': 'prompt_block', 'type': 'str', 'required': False,
            'description': "The prompt text. Only used for 'add' and 'update' operations."
        }
    }

class ModesAndPersonasManagerTool:
    name = "ModesAndPersonasManagerTool"
    description = "Exposes all CRUD operations for the brain_personas and brain_modes registries."
    args_schema = ModesAndPersonasManagerToolArgsSchema
    func = modes_and_personas_manager_func
