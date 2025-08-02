#!/usr/bin/env python3
"""
Run Brain Agent Script

This script demonstrates how to register a brain and query it.
It's also a convenient way to test the brain agent functionality.

Usage:
    python run_brain_agent.py register <directory> <brain_name>
    python run_brain_agent.py query <brain_name> <query_text>
    python run_brain_agent.py list
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add core to path to enable imports using HEAVEN_DATA_DIR structure
try:
    # Try to use heaven-base's HEAVEN_DATA_DIR system
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'heaven-base'))
    from heaven_base.utils.get_env_value import EnvConfigUtil
    heaven_data_dir = EnvConfigUtil.get_heaven_data_dir()
    core_path = os.path.join(heaven_data_dir, '..', 'core')
    sys.path.append(core_path)
except (ImportError, Exception):
    # Fallback for development
    sys.path.append('/home/GOD/core')

from computer_use_demo.codebase_analyzer_system.brain_agent import BrainAgent, register_brain
from computer_use_demo.tools.base.tools.registry_tool import registry_util_func


def list_brains():
    """List all registered brains."""
    result = registry_util_func(
        operation="list_keys",
        registry_name="brain_configs"
    )
    
    print(f"Listing brains: {result}")


def register_brain_cmd(directory, brain_name, chunk_size=-1):
    """Register a new brain."""
    # Ensure directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory does not exist: {directory}")
        sys.exit(1)
        
    # Register the brain
    register_brain(directory, brain_name, chunk_size)
    
    print(f"Directory: {directory}")



def query_brain_cmd(brain_name, query_text):
    """Query a brain and print the result."""
    # Check if brain exists
    brain_exists = registry_util_func(
        operation="get", 
        registry_name="brain_configs",
        key=brain_name
    )
    # Check if the string indicates the brain was found
    if "not found" in brain_exists:
        print(f"Error: Brain '{brain_name}' not found.")
        sys.exit(1)
    
    # Create brain agent
    agent = BrainAgent()
    
    # Query the brain
    print(f"Querying brain '{brain_name}'...")
    print(f"Query: {query_text}")
    print("-" * 80)
    
    # Fix: Run async query with asyncio
    # prepare the query
    query_with_brain = f"""Query the following brain with the following query text. This is the exact brain name:\n\n<target_brain>{brain_name}</target_brain>\n\nAnd this is the query to use: <query_text>{query_text}<query_text>"""
    result = asyncio.run(agent.query(query_with_brain))
    
    # print("-" * 80)
    # print("RESULT:")
    # print(result)
    # print("-" * 80)
    return result
    


def print_usage():
    """Print usage instructions."""
    print("Usage:")
    print("  python run_brain_agent.py register <directory> <brain_name> [chunk_size]")
    print("  python run_brain_agent.py query <brain_name> <query_text>")
    print("  python run_brain_agent.py list")
    print("")
    print("Examples:")
    print("  python run_brain_agent.py register ./docs my_docs_brain")
    print("  python run_brain_agent.py query my_docs_brain \"How do I implement X?\"")
    print("  python run_brain_agent.py list")


if __name__ == "__main__":
    # Ensure at least one argument is provided
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
        
    command = sys.argv[1].lower()
    
    if command == "list":
        list_brains()
    elif command == "register":
        if len(sys.argv) < 4:
            print("Error: 'register' command requires directory and brain_name arguments.")
            print_usage()
            sys.exit(1)
            
        directory = sys.argv[2]
        brain_name = sys.argv[3]
        chunk_size = int(sys.argv[4]) if len(sys.argv) > 4 else -1
        
        register_brain_cmd(directory, brain_name, chunk_size)
    elif command == "query":
        if len(sys.argv) < 4:
            print("Error: 'query' command requires brain_name and query_text arguments.")
            print_usage()
            sys.exit(1)
            
        brain_name = sys.argv[2]
        query_text = sys.argv[3]
        
        query_brain_cmd(brain_name, query_text)
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)