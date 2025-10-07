"""
LLM-Use - Advanced Features Example
Demonstrates streaming, A/B testing, caching, and benchmarking
"""

import asyncio
import time
from llm_use import (
    SmartRouter, 
    ResilientLLMClient,
    ProductionABTestManager,
    ProductionBenchmarker
)

async def demo_streaming():
    """Demo 1: Streaming responses"""
    print("\n" + "="*70)
    print("🌊 DEMO 1: STREAMING RESPONSES")
    print("="*70)
    
    router = SmartRouter(verbose=False)
    client = ResilientLLMClient(router)
    
    prompt = "Write a short poem about AI"
    print(f"\n💬 Prompt: {prompt}")
    print("🤖 Streaming response:\n")
    
    # Stream response
    response_generator = await client.chat(prompt, stream=True)
    
    async for chunk in response_generator:
        print(chunk, end="", flush=True)
    
    print("\n\n✅ Streaming complete!")

async def demo_cost_control():
    """Demo 2: Cost control"""
    print("\n" + "="*70)
    print("💰 DEMO 2: COST CONTROL")
    print("="*70)
    
    router = SmartRouter(verbose=False)
    client = ResilientLLMClient(router)
    
    prompt = "Explain machine learning in detail with examples"
    
    # Without cost limit
    print("\n📊 Without cost limit:")
    routing1 = router.route(prompt)
    print(f"  Selected: {routing1.model_name}")
    print(f"  Estimated cost: ${routing1.estimated_cost:.6f}")
    
    # With cost limit
    print("\n📊 With $0.001 cost limit:")
    response = await client.chat(prompt, max_cost=0.001)
    routing2 = router.route(prompt, prefer_local=False)
    print(f"  Selected: {routing2.model_name}")
    print(f"  Estimated cost: ${routing2.estimated_cost:.6f}")
    print(f"\n💡 Saved: ${(routing1.estimated_cost - routing2.estimated_cost):.6f}")

async def demo_caching():
    """Demo 3: Response caching"""
    print("\n" + "="*70)
    print("💾 DEMO 3: RESPONSE CACHING")
    print("="*70)
    
    router = SmartRouter(verbose=False)
    client = ResilientLLMClient(router, cache_ttl=3600)
    
    prompt = "What is the capital of France?"
    
    # First request (not cached)
    print("\n📥 First request (not cached):")
    start = time.time()
    response1 = await client.chat(prompt)
    time1 = time.time() - start
    print(f"  Response: {response1}")
    print(f"  Time: {time1:.3f}s")
    
    # Second request (cached)
    print("\n⚡ Second request (cached):")
    start = time.time()
    response2 = await client.chat(prompt)
    time2 = time.time() - start
    print(f"  Response: {response2}")
    print(f"  Time: {time2:.3f}s")
    print(f"\n💡 Speedup: {time1/time2:.1f}x faster!")

async def demo_ab_testing():
    """Demo 4: A/B Testing"""
    print("\n" + "="*70)
    print("🧪 DEMO 4: A/B TESTING")
    print("="*70)
    
    router = SmartRouter(verbose=False)
    client = ResilientLLMClient(router)
    
    # Create A/B test manager
    ab_manager = ProductionABTestManager(min_sample_size=5)
    client.set_ab_test_manager(ab_manager)
    
    # Create test
    test_id = ab_manager.create_test(
        name="GPT-3.5 vs Claude Haiku",
        model_a="gpt-3.5-turbo",
        model_b="claude-3-haiku",
        allocation_ratio=0.5
    )
    
    print(f"\n✅ Created A/B test: {test_id[:8]}...")
    print("📊 Running 10 test requests...\n")
    
    # Run test requests
    prompts = [
        "Explain photosynthesis",
        "What is recursion?",
        "Translate 'hello' to Spanish",
        "Calculate 15 * 23",
        "What is DNA?",
    ] * 2  # 10 total requests
    
    for i, prompt in enumerate(prompts, 1):
        user_id = f"user_{i % 5}"  # Simulate 5 different users
        
        # This will automatically select variant based on A/B test
        response = await client.chat(
            prompt,
            ab_test_id=test_id,
            user_id=user_id
        )
        
        print(f"  {i}/10: User {user_id[-1]} → {response[:50]}...")
    
    # Analyze results
    print("\n📊 Analyzing results...")
    results = ab_manager.analyze_test(test_id)
    
    if results.get('status') == 'insufficient_data':
        print(f"⚠️  Need more samples: {results['samples_a']} + {results['samples_b']} / {results['required']}")
    else:
        print("\n✅ A/B Test Results:")
        print(f"  Samples A: {results['samples_a']}")
        print(f"  Samples B: {results['samples_b']}")
        print(f"  Winner: {results['winner']}")
        print(f"  Statistical power: {results['statistical_power']:.2%}")

async def demo_benchmarking():
    """Demo 5: Model benchmarking"""
    print("\n" + "="*70)
    print("🔬 DEMO 5: MODEL BENCHMARKING")
    print("="*70)
    
    router = SmartRouter(verbose=False)
    client = ResilientLLMClient(router)
    
    # Quick benchmark
    print("\n⏱️  Running quick benchmarks (3 tests per model)...")
    benchmarker = ProductionBenchmarker(comprehensive=False)
    
    available_models = router._get_available_models()
    
    # Benchmark first 2 available models
    test_models = list(available_models.items())[:2]
    
    results = []
    for model_id, config in test_models:
        print(f"\n  Testing {config.name}...")
        result = await benchmarker.benchmark_model(
            model_id,
            config.provider,
            client,
            comprehensive=False
        )
        results.append(result)
    
    # Show comparison
    print("\n📊 Benchmark Results:")
    print("-" * 70)
    print(f"{'Model':<25} {'Latency':<12} {'Quality':<10} {'Cost':<10}")
    print("-" * 70)
    
    for result in results:
        if result['status'] == 'success':
            metrics = result['metrics']
            print(f"{result['model_id']:<25} "
                  f"{metrics['avg_latency']:<12.3f}s "
                  f"{metrics['avg_quality']:<10.2f} "
                  f"${metrics['total_cost']:<10.6f}")

async def demo_fallback_resilience():
    """Demo 6: Fallback and resilience"""
    print("\n" + "="*70)
    print("🛡️  DEMO 6: FALLBACK & RESILIENCE")
    print("="*70)
    
    router = SmartRouter(verbose=False)
    client = ResilientLLMClient(router, max_retries=3)
    
    prompt = "What is artificial intelligence?"
    
    print(f"\n💬 Prompt: {prompt}")
    print("\n🔄 Client features:")
    print("  • Automatic retries (max 3)")
    print("  • Exponential backoff")
    print("  • Circuit breaker pattern")
    print("  • Intelligent fallback chain")
    
    # Build fallback chain
    routing = router.route(prompt)
    fallback_chain = client._build_fallback_chain(routing.model_id, routing.complexity)
    
    print(f"\n📋 Fallback chain for '{routing.model_name}':")
    for i, model_id in enumerate(fallback_chain, 1):
        print(f"  {i}. {router.models[model_id].name}")
    
    print("\n💡 If primary fails → tries next in chain automatically")

async def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("🚀 LLM-USE - ADVANCED FEATURES DEMO")
    print("="*70)
    
    demos = [
        ("Streaming", demo_streaming),
        ("Cost Control", demo_cost_control),
        ("Caching", demo_caching),
        ("A/B Testing", demo_ab_testing),
        ("Benchmarking", demo_benchmarking),
        ("Fallback & Resilience", demo_fallback_resilience),
    ]
    
    print("\n📚 Available demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\n▶️  Running all demos...\n")
    
    for name, demo_func in demos:
        try:
            await demo_func()
            await asyncio.sleep(1)  # Pause between demos
        except Exception as e:
            print(f"\n❌ Demo '{name}' failed: {e}")
            continue
    
    print("\n" + "="*70)
    print("✅ ALL DEMOS COMPLETED!")
    print("="*70)
    print("\n💡 Next steps:")
    print("  • Check the main code for more features")
    print("  • Read the documentation")
    print("  • Start the API server: python llm_use.py server")
    print("  • Run benchmarks: python llm_use.py benchmark")