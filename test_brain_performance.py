#!/usr/bin/env python3
"""
Test brain-agent batch processing performance
"""

import asyncio
import time
from brain_agent.brain_agent import BrainAgent

async def test_brain_performance():
    """Test brain query performance with timing."""
    print("🧠 Testing Brain-Agent Batch Performance")
    print("=" * 50)
    
    # Create brain agent
    agent = BrainAgent()
    
    # Test query that should hit multiple neurons
    test_query = """TargetBrain: test_registry_brain
Query: What are the project coding standards and conventions?"""
    
    print(f"\n📊 Query: {test_query[:50]}...")
    print(f"🔍 Target brain: test_registry_brain")
    
    # Time the full query
    start_time = time.time()
    result = await agent.query(test_query)
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total query time: {total_time:.2f} seconds")
    print(f"\n📝 Full result:")
    print(result)
    
    print("\n" + "=" * 50)
    print("🎯 Performance Analysis:")
    print(f"- Total time: {total_time:.2f}s")
    print(f"- Check HEAVEN_AGENT_DEBUG=1 logs for batch timing details")
    print(f"- If batch time << total time: LLM provider is the bottleneck")
    print(f"- If batch time ≈ total time: Processing is the bottleneck")

if __name__ == "__main__":
    asyncio.run(test_brain_performance())