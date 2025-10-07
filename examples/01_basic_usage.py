"""
LLM-Use - Basic Usage Example
Simple introduction to intelligent LLM routing
"""

import asyncio
from llm_use import SmartRouter, ResilientLLMClient

async def main():
    print("🚀 LLM-Use - Basic Usage Example\n")
    
    # 1. Initialize the router
    print("📌 Step 1: Initialize router")
    router = SmartRouter(verbose=True)
    
    # 2. Create resilient client
    print("📌 Step 2: Create client")
    client = ResilientLLMClient(router)
    
    # 3. Simple chat examples
    prompts = [
        "What is 2+2?",  # Simple → cheap model
        "Explain quantum computing in detail",  # Complex → powerful model
        "Write a Python function to reverse a string",  # Coding → specialized model
    ]
    
    print("\n📌 Step 3: Send requests (auto-routing)\n")
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"💬 Prompt: {prompt}")
        print(f"{'='*60}")
        
        # Get response (automatic routing)
        response = await client.chat(prompt)
        
        print(f"\n🤖 Response:\n{response}")
        
        # Show routing decision
        routing = router.route(prompt)
        print(f"\n📊 Routed to: {routing.model_name}")
        print(f"💰 Cost: ${routing.estimated_cost:.6f}")
    
    # 4. Show statistics
    print("\n\n📊 SESSION STATISTICS")
    print("="*60)
    stats = router.get_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Total cost: ${stats['total_cost']:.4f}")
    print(f"\nModel usage:")
    for model, count in stats['model_usage'].items():
        print(f"  • {router.models[model].name}: {count}x")

if __name__ == "__main__":
    # Check if API keys are set
    import os
    
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    
    if missing:
        print("⚠️  Missing API keys:")
        for key in missing:
            print(f"   export {key}=your_key_here")
        print("\n💡 Set at least one API key to continue")
    else:
        asyncio.run(main())
