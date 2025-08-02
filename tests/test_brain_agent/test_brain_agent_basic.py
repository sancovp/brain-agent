#!/usr/bin/env python3
"""
Basic test for Brain Agent system in heaven-base
"""

import sys
import os
import asyncio

# Add brain-agent and heaven-base to Python path using HEAVEN_DATA_DIR structure
try:
    from heaven_base.utils.get_env_value import EnvConfigUtil
    heaven_data_dir = EnvConfigUtil.get_heaven_data_dir()
    # Assumes brain-agent and heaven-base are siblings to heaven-data directory
    brain_agent_path = os.path.join(heaven_data_dir, '..', 'brain-agent')
    heaven_base_path = os.path.join(heaven_data_dir, '..', 'heaven-base')
    sys.path.insert(0, brain_agent_path)
    sys.path.insert(0, heaven_base_path)
except ImportError:
    # Fallback for development - use relative paths
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, current_dir)
    sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'heaven-base'))

from brain_agent import (
    BrainConfig,
    BrainAgent,
    register_brain,
    QueryBrainTool,
    CognizeTool,
    InstructTool
)


def test_imports():
    """Test that all brain agent components can be imported"""
    print("✅ All imports successful")
    print(f"  - BrainConfig: {BrainConfig}")
    print(f"  - BrainAgent: {BrainAgent}")
    print(f"  - QueryBrainTool: {QueryBrainTool}")
    print(f"  - CognizeTool: {CognizeTool}")
    print(f"  - InstructTool: {InstructTool}")
    

def test_brain_config():
    """Test BrainConfig creation"""
    config = BrainConfig(
        directory="/tmp/test_brain",
        brain_name="test_brain",
        chunk_size=-1
    )
    print(f"\n✅ BrainConfig created: {config}")
    return config


async def test_brain_agent():
    """Test BrainAgent creation and basic functionality"""
    try:
        # Create brain agent
        agent = BrainAgent()
        print(f"\n✅ BrainAgent created successfully")
        print(f"  - Agent name: {agent.config.name}")
        print(f"  - Tools: {[tool.name for tool in agent.config.tools]}")
        
        return True
    except Exception as e:
        print(f"\n❌ BrainAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("🧪 Testing Brain Agent System in heaven-base")
    print("=" * 60)
    
    # Test imports
    test_imports()
    
    # Test BrainConfig
    test_brain_config()
    
    # Test BrainAgent
    success = await test_brain_agent()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
        

if __name__ == "__main__":
    asyncio.run(main())