"""
Demo script showing how to use the non-blocking background summarization feature.

This demonstrates how conversations can continue immediately while summaries
are generated in the background.
"""

import os
import asyncio
from pathlib import Path
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from agent_memory import MemoryOps


async def demo_nonblocking_conversation():
    """Demonstrate non-blocking conversation with background summarization."""
    
    print("=" * 80)
    print("NON-BLOCKING MEMORY SUMMARIZATION DEMO")
    print("=" * 80)
    
    # Setup LLM
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.6,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    # Initialize memory system
    memory_ops = MemoryOps(
        username="demo_user",
        llm=llm,
        memory_dir="./demo_memory",
        rate_limit_delay=1.0,
        summary_interval=3  # Summarize every 3 turns for this demo
    )
    
    print("\n✅ Memory system initialized (background mode enabled)")
    print("📊 Summary interval: Every 3 turns")
    print("-" * 80)
    
    # Simulate a conversation
    conversations = [
        ("Hi! Can you help me with quadratic equations?", 
         "Of course! I'd be happy to help with quadratic equations. What would you like to know?"),
        
        ("How do I solve x² + 5x + 6 = 0?",
         "Great question! You can factor this equation. Find two numbers that multiply to 6 and add to 5: those are 2 and 3. So (x+2)(x+3)=0, giving x=-2 or x=-3."),
        
        ("What about equations that don't factor easily?",
         "For those, use the quadratic formula: x = (-b ± √(b²-4ac)) / 2a. This works for any quadratic equation in the form ax² + bx + c = 0."),
    ]
    
    print("\n🔄 Processing conversation turns (non-blocking):\n")
    
    import time
    for i, (user_msg, bot_msg) in enumerate(conversations, 1):
        start = time.time()
        
        # Process message with background summarization (NON-BLOCKING!)
        result = await memory_ops.process_message(
            message=user_msg,
            bot_response=bot_msg,
            background_summary=True  # Enable background mode
        )
        
        elapsed = time.time() - start
        pending = result.get('background_tasks', 0)
        is_summary_turn = result.get('is_summary_turn', False)
        
        print(f"Turn {i}:")
        print(f"  User: {user_msg[:60]}...")
        print(f"  Bot: {bot_msg[:60]}...")
        print(f"  ⚡ Processed in {elapsed:.2f}s (non-blocking!)")
        if is_summary_turn:
            print(f"  📊 Summary created for this turn (background)")
        print(f"  🔄 Background tasks running: {pending}")
        print()
    
    # Check pending summaries
    pending_count = memory_ops.get_pending_summaries_count()
    print("-" * 80)
    print(f"✅ All {len(conversations)} turns processed immediately!")
    print(f"🔄 {pending_count} summary tasks still running in background")
    print()
    
    # Option 1: Continue without waiting (conversation can proceed)
    print("💡 Option 1: Continue conversation (summaries complete later)")
    print("   The user can continue chatting while summaries generate!")
    print()
    
    # Option 2: Wait for summaries to complete (if needed)
    print("💡 Option 2: Wait for summaries (when needed)")
    print("   Waiting for background tasks to complete...")
    await memory_ops.wait_for_summaries(timeout=60)
    print("   ✅ All summaries completed!")
    print()
    
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\n📊 Key Benefits:")
    print("  ✅ Immediate response: User doesn't wait for summarization")
    print("  ✅ Background processing: Summaries generated asynchronously")
    print("  ✅ Better UX: No blocking delays in conversation flow")
    print()
    print("📝 Usage in your code:")
    print("  result = await memory_ops.process_message(")
    print("      message=user_msg,")
    print("      bot_response=bot_msg,")
    print("      background_summary=True  # Enable non-blocking mode")
    print("  )")
    print()


async def demo_blocking_vs_nonblocking():
    """Compare blocking vs non-blocking modes side by side."""
    
    print("\n" + "=" * 80)
    print("BLOCKING vs NON-BLOCKING COMPARISON")
    print("=" * 80)
    
    import time
    
    # Sample conversation
    user_msg = "Can you explain Newton's laws?"
    bot_msg = "Newton's three laws describe how objects move. The first law states that an object in motion stays in motion unless acted upon by a force."
    
    # Setup
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.6,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    # Test blocking mode
    print("\n🔴 BLOCKING MODE:")
    memory_ops_blocking = MemoryOps(
        username="demo_blocking",
        llm=llm,
        memory_dir="./demo_memory_blocking",
        rate_limit_delay=0.5,
        summary_interval=1  # Summarize every turn for comparison
    )
    
    start = time.time()
    result = await memory_ops_blocking.process_message(
        message=user_msg,
        bot_response=bot_msg,
        background_summary=False  # Blocking mode
    )
    blocking_time = time.time() - start
    print(f"  ⏱️  Time: {blocking_time:.2f}s (waited for summary)")
    print(f"  📊 Summary created: {result.get('is_summary_turn', False)}")
    
    # Test non-blocking mode
    print("\n🟢 NON-BLOCKING MODE:")
    memory_ops_nonblocking = MemoryOps(
        username="demo_nonblocking",
        llm=llm,
        memory_dir="./demo_memory_nonblocking",
        rate_limit_delay=0.5,
        summary_interval=1  # Summarize every turn for comparison
    )
    
    start = time.time()
    result = await memory_ops_nonblocking.process_message(
        message=user_msg,
        bot_response=bot_msg,
        background_summary=True  # Non-blocking mode
    )
    nonblocking_time = time.time() - start
    print(f"  ⏱️  Time: {nonblocking_time:.2f}s (immediate!)")
    print(f"  📊 Summary created: {result.get('is_summary_turn', False)}")
    print(f"  🔄 Background tasks: {result.get('background_tasks', 0)}")
    
    # Wait for background task
    await memory_ops_nonblocking.wait_for_summaries()
    
    print(f"\n📊 SPEEDUP: {blocking_time / nonblocking_time:.1f}x faster response!")
    print("   User experience is significantly improved!")


async def main():
    """Run all demos."""
    # Check API key
    if not os.getenv("NVIDIA_API_KEY"):
        print("\n❌ ERROR: NVIDIA_API_KEY not set!")
        print("Set it with: export NVIDIA_API_KEY='your-key-here'")
        return
    
    # Run demos
    await demo_blocking_vs_nonblocking()
    await demo_nonblocking_conversation()


if __name__ == "__main__":
    asyncio.run(main())

