#!/usr/bin/env python3
"""
Demo script for Reasoning-as-a-Service (RaaS).
Demonstrates the core functionality without requiring the full FastAPI server.
"""

import asyncio
import json
from src.adapters.model_adapter import MockModelAdapter
from src.core.monitor import CoTMonitor
from src.agents.reasoning_agent import ReasoningAgent
from src.core.models import ReasoningRequest


async def demo():
    """Run a demonstration of the RaaS system."""
    print("🚀 Reasoning-as-a-Service (RaaS) Demo")
    print("=" * 50)
    
    # Initialize components
    print("📦 Initializing components...")
    adapter = MockModelAdapter()
    monitor = CoTMonitor(adapter)
    agent = ReasoningAgent(adapter, monitor, n_candidates=2)
    
    # Test request
    request = ReasoningRequest(
        input="What is the capital of France?",
        context={"domain": "geography"},
        request_id="demo_123"
    )
    
    print(f"🤔 Processing request: {request.input}")
    print(f"📋 Context: {request.context}")
    print()
    
    # Process in dryrun mode
    print("🔄 Processing in dryrun mode...")
    response = await agent.reason(request, mode="dryrun")
    
    print("✅ Results:")
    print(f"   Answer: {response.answer}")
    print(f"   Reasoning steps: {len(response.reasoning_trace)}")
    for i, step in enumerate(response.reasoning_trace, 1):
        print(f"     Step {i}: {step}")
    print()
    print(f"📊 Scores:")
    print(f"   Faithfulness: {response.faithfulness_score:.3f}")
    print(f"   Coherence: {response.coherence_score:.3f}")
    print(f"   Risk Flag: {response.risk_flag}")
    print(f"   Monitor Explanation: {response.monitor_explanation}")
    print()
    
    # Test perturbation mode
    print("🔬 Testing perturbation mode...")
    perturb_request = ReasoningRequest(
        input="Solve 2 + 2",
        context={"domain": "math"}
    )
    
    perturb_response = await agent.reason(perturb_request, mode="perturb")
    
    print("✅ Perturbation Results:")
    print(f"   Original Answer: {perturb_response.perturbation.original_answer}")
    print(f"   Causal Influence Score: {perturb_response.perturbation.causal_influence_score:.3f}")
    print(f"   Number of Perturbations: {len(perturb_response.perturbation.perturbed_answers)}")
    
    # Show some perturbation results
    for i, perturb in enumerate(perturb_response.perturbation.perturbed_answers[:3]):
        print(f"     Perturbation {i+1}: Removed steps {perturb['removed_steps']}, "
              f"Answer: '{perturb['new_answer']}', Changed: {perturb['changed']}")
    
    print()
    print("🎉 Demo completed successfully!")
    print("💡 To run the full API server, use: uvicorn src.main:app --reload")


if __name__ == "__main__":
    asyncio.run(demo())
