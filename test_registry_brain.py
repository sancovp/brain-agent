#!/usr/bin/env python3
"""
Test script for registry-based brain functionality.
Tests creating a brain that loads neurons from STARLOG registry keys.
"""

import os
import sys
import json
import asyncio

# Add brain-agent to path
sys.path.insert(0, os.path.dirname(__file__))

from brain_agent.brain_agent import BrainAgent, register_brain, get_brain_config
from brain_agent.manager_tools import brain_manager_func
from heaven_base.tools.registry_tool import registry_util_func

def test_create_registry_brain():
    """Test creating a brain that uses registry keys as neurons."""
    
    # Create registry if it doesn't exist
    registry_util_func("create_registry", registry_name="test_rules_registry")
    
    # Add some sample rules
    sample_rules = {
        "rule_1": {
            "rule": "Always use type hints in Python functions",
            "category": "coding",
            "priority": 8
        },
        "rule_2": {
            "rule": "Write unit tests for all public functions",
            "category": "testing", 
            "priority": 9
        },
        "rule_3": {
            "rule": "Use descriptive variable names",
            "category": "coding",
            "priority": 7
        }
    }
    
    for rule_id, rule_data in sample_rules.items():
        registry_util_func(
            "add",
            registry_name="test_rules_registry", 
            key=rule_id,
            value_dict=rule_data
        )
    
    # Create a brain that uses registry keys as neurons
    result = brain_manager_func(
        operation="add",
        brain_id="test_registry_brain",
        name="Test Registry Brain",
        neuron_source_type="registry_keys",
        neuron_source="test_rules_registry",
        chunk_max=30000
    )
    print(result)
    
    return "test_registry_brain"

async def test_query_registry_brain(brain_name):
    """Test querying the registry-based brain."""
    
    # Create brain agent
    agent = BrainAgent()
    
    # Query the brain
    query = f"TargetBrain: {brain_name}\nQuery: What coding standards should I follow when writing Python functions?"
    
    result = await agent.query(query)
    
    print(result)
    
    return result

async def main():
    """Run the full test."""
    
    # Test creating registry brain
    brain_name = test_create_registry_brain()
    
    # Test querying the brain
    await test_query_registry_brain(brain_name)

if __name__ == "__main__":
    # Set environment for heaven base
    os.environ["HEAVEN_DATA_DIR"] = "/tmp/heaven_data"
    
    asyncio.run(main())