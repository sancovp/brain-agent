#!/usr/bin/env python3
"""
Test persona/mode integration with brain agent
"""

import sys
import asyncio
import os

# Add brain-agent to Python path (assuming it's in HEAVEN_DATA_DIR structure)
try:
    from heaven_base.utils.get_env_value import EnvConfigUtil
    heaven_data_dir = EnvConfigUtil.get_heaven_data_dir()
    brain_agent_path = os.path.join(heaven_data_dir, '..', 'brain-agent')  # Assumes brain-agent is sibling to heaven-data
    sys.path.insert(0, brain_agent_path)
except ImportError:
    # Fallback for development
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain_agent.query_brain_tool import query_brain_func

async def test_persona_mode():
    """Test persona/mode integration"""
    
    # Test basic query without persona/mode
    print("=== Testing Basic Query ===")
    try:
        result1 = await query_brain_func(
            brain="diplomacy_brain",
            query="What is the Juggernaut strategy?"
        )
        print(f"Basic query result: {result1[:200]}...")
    except Exception as e:
        print(f"Basic query error: {e}")
    
    # Test query with persona and mode
    print("\n=== Testing Persona/Mode Query ===")
    try:
        result2 = await query_brain_func(
            brain="diplomacy_brain", 
            query="What is the Juggernaut strategy?",
            persona_id="senior_scientist",
            mode_id="summarize"
        )
        print(f"Enhanced query result: {result2[:200]}...")
    except Exception as e:
        print(f"Enhanced query error: {e}")

if __name__ == "__main__":
    asyncio.run(test_persona_mode())