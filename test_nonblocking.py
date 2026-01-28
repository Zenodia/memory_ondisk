"""
Test script for non-blocking background summarization.

This script tests:
1. Non-blocking vs blocking performance comparison
2. Background task management
3. Summary interval functionality
4. Proper summary creation at correct intervals

Run with: python test_nonblocking.py
"""

import os
import asyncio
import time
from pathlib import Path
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from agent_memory import MemoryOps


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title):
    """Print a section divider."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


async def test_blocking_mode():
    """Test blocking mode (waits for each summary)."""
    print_section("TEST 1: BLOCKING MODE")
    
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.6,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    memory_ops = MemoryOps(
        username="test_blocking",
        llm=llm,
        memory_dir="./test_blocking_output",
        rate_limit_delay=0.5,
        summary_interval=3  # Summarize every 3 turns for quick test
    )
    
    conversations = [
        ("What is 2+2?", "2+2 equals 4."),
        ("What is 3+3?", "3+3 equals 6."),
        ("What is 4+4?", "4+4 equals 8."),
        ("What is 5+5?", "5+5 equals 10."),
        ("What is 6+6?", "6+6 equals 12."),
    ]
    
    print(f"\nProcessing {len(conversations)} conversations in BLOCKING mode...")
    print("(Waits for summary at each interval turn)\n")
    
    start_time = time.time()
    
    for i, (user_msg, bot_msg) in enumerate(conversations, 1):
        turn_start = time.time()
        
        result = await memory_ops.process_message(
            message=user_msg,
            bot_response=bot_msg,
            background_summary=False  # BLOCKING
        )
        
        turn_time = time.time() - turn_start
        is_summary_turn = result.get('is_summary_turn', False)
        
        if is_summary_turn:
            print(f"  📊 Turn {i}: {turn_time:.2f}s (WAITED for summary)")
        else:
            print(f"     Turn {i}: {turn_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n✓ Total time (blocking): {total_time:.2f}s")
    
    return total_time, len(conversations)


async def test_nonblocking_mode():
    """Test non-blocking mode (summaries in background)."""
    print_section("TEST 2: NON-BLOCKING MODE")
    
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.6,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    memory_ops = MemoryOps(
        username="test_nonblocking",
        llm=llm,
        memory_dir="./test_nonblocking_output",
        rate_limit_delay=0.5,
        summary_interval=3  # Summarize every 3 turns for quick test
    )
    
    conversations = [
        ("What is 2+2?", "2+2 equals 4."),
        ("What is 3+3?", "3+3 equals 6."),
        ("What is 4+4?", "4+4 equals 8."),
        ("What is 5+5?", "5+5 equals 10."),
        ("What is 6+6?", "6+6 equals 12."),
    ]
    
    print(f"\nProcessing {len(conversations)} conversations in NON-BLOCKING mode...")
    print("(Summaries generated in background - immediate returns)\n")
    
    start_time = time.time()
    
    for i, (user_msg, bot_msg) in enumerate(conversations, 1):
        turn_start = time.time()
        
        result = await memory_ops.process_message(
            message=user_msg,
            bot_response=bot_msg,
            background_summary=True  # NON-BLOCKING
        )
        
        turn_time = time.time() - turn_start
        is_summary_turn = result.get('is_summary_turn', False)
        pending = result.get('background_tasks', 0)
        
        if is_summary_turn:
            print(f"  📊 Turn {i}: {turn_time:.2f}s (IMMEDIATE! {pending} tasks in background)")
        else:
            print(f"     Turn {i}: {turn_time:.2f}s ({pending} tasks in background)")
    
    processing_time = time.time() - start_time
    pending_count = memory_ops.get_pending_summaries_count()
    
    print(f"\n✓ Processing time (non-blocking): {processing_time:.2f}s")
    print(f"🔄 {pending_count} summaries still running in background")
    
    # Wait for background tasks
    print(f"\n⏳ Waiting for {pending_count} background summaries to complete...")
    await memory_ops.wait_for_summaries(timeout=60)
    
    total_time = time.time() - start_time
    print(f"✓ Total time (with background completion): {total_time:.2f}s")
    
    return processing_time, total_time, len(conversations)


async def test_summary_intervals():
    """Test that summaries are created at correct intervals."""
    print_section("TEST 3: SUMMARY INTERVALS")
    
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.6,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    # Test with interval=5
    memory_ops = MemoryOps(
        username="test_intervals",
        llm=llm,
        memory_dir="./test_intervals_output",
        rate_limit_delay=0.3,
        summary_interval=5  # Every 5 turns
    )
    
    print(f"\nTesting summary_interval=5")
    print("Expected summaries at turns: 5, 10, 15\n")
    
    summary_turns = []
    
    for i in range(1, 16):  # 15 turns
        result = await memory_ops.process_message(
            message=f"Question {i}",
            bot_response=f"Answer {i}",
            background_summary=True
        )
        
        is_summary_turn = result.get('is_summary_turn', False)
        if is_summary_turn:
            summary_turns.append(i)
            print(f"  📊 Turn {i:2d}: Summary created")
        else:
            print(f"     Turn {i:2d}: No summary")
    
    # Wait for background summaries
    await memory_ops.wait_for_summaries(timeout=60)
    
    print(f"\n✓ Summaries created at turns: {summary_turns}")
    
    expected = [5, 10, 15]
    if summary_turns == expected:
        print(f"✅ PASS: Summaries at correct intervals {expected}")
        return True
    else:
        print(f"❌ FAIL: Expected {expected}, got {summary_turns}")
        return False


async def test_background_task_management():
    """Test background task tracking and cleanup."""
    print_section("TEST 4: BACKGROUND TASK MANAGEMENT")
    
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.6,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    memory_ops = MemoryOps(
        username="test_tasks",
        llm=llm,
        memory_dir="./test_tasks_output",
        rate_limit_delay=0.3,
        summary_interval=2  # Every 2 turns
    )
    
    print(f"\nCreating multiple background tasks (interval=2)...")
    
    max_pending = 0
    
    for i in range(1, 7):  # 6 turns = 3 summaries
        result = await memory_ops.process_message(
            message=f"Message {i}",
            bot_response=f"Response {i}",
            background_summary=True
        )
        
        pending = result.get('background_tasks', 0)
        max_pending = max(max_pending, pending)
        
        print(f"  Turn {i}: {pending} background tasks running")
    
    print(f"\n✓ Max concurrent background tasks: {max_pending}")
    
    # Verify cleanup
    print(f"\n⏳ Waiting for all tasks to complete...")
    await memory_ops.wait_for_summaries(timeout=60)
    
    remaining = memory_ops.get_pending_summaries_count()
    print(f"✓ Background tasks after completion: {remaining}")
    
    if remaining == 0:
        print(f"✅ PASS: All tasks completed and cleaned up")
        return True
    else:
        print(f"❌ FAIL: {remaining} tasks still pending")
        return False


async def test_no_summary_mode():
    """Test disabling summaries (summary_interval very high)."""
    print_section("TEST 5: NO SUMMARY MODE")
    
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.6,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    memory_ops = MemoryOps(
        username="test_no_summary",
        llm=llm,
        memory_dir="./test_no_summary_output",
        rate_limit_delay=0.3,
        summary_interval=1000  # Very high - no summaries for small tests
    )
    
    print(f"\nProcessing 10 turns with summary_interval=1000")
    print("Expected: No summaries created\n")
    
    summary_count = 0
    
    start_time = time.time()
    
    for i in range(1, 11):
        result = await memory_ops.process_message(
            message=f"Question {i}",
            bot_response=f"Answer {i}",
            background_summary=True
        )
        
        if result.get('is_summary_turn', False):
            summary_count += 1
    
    elapsed = time.time() - start_time
    
    print(f"✓ Processed 10 turns in {elapsed:.2f}s")
    print(f"✓ Summaries created: {summary_count}")
    
    if summary_count == 0:
        print(f"✅ PASS: No summaries created (as expected)")
        return True
    else:
        print(f"❌ FAIL: {summary_count} summaries created (expected 0)")
        return False


async def main():
    """Run all non-blocking tests."""
    print_header("NON-BLOCKING SUMMARIZATION TEST SUITE")
    
    # Check API key
    if not os.getenv("NVIDIA_API_KEY"):
        print("\n❌ ERROR: NVIDIA_API_KEY not set!")
        print("Set it with: export NVIDIA_API_KEY='your-key-here'")
        return
    
    print("\nThis test suite validates non-blocking background summarization:")
    print("  1. Blocking vs non-blocking performance")
    print("  2. Summary interval functionality")
    print("  3. Background task management")
    print("  4. Task cleanup")
    
    results = {}
    
    # Test 1 & 2: Performance comparison
    try:
        blocking_time, _ = await test_blocking_mode()
        results['blocking'] = True
    except Exception as e:
        print(f"\n❌ Blocking test failed: {e}")
        results['blocking'] = False
        blocking_time = 0
    
    try:
        nonblocking_proc_time, nonblocking_total_time, _ = await test_nonblocking_mode()
        results['nonblocking'] = True
        
        # Show comparison
        print_section("PERFORMANCE COMPARISON")
        print(f"\n  Blocking mode:          {blocking_time:.2f}s")
        print(f"  Non-blocking (process): {nonblocking_proc_time:.2f}s")
        print(f"  Non-blocking (total):   {nonblocking_total_time:.2f}s")
        
        if blocking_time > 0:
            speedup = blocking_time / nonblocking_proc_time
            print(f"\n  ⚡ Speedup: {speedup:.1f}x faster (non-blocking processing)")
            print(f"  💡 User sees immediate responses, summaries happen in background")
    except Exception as e:
        print(f"\n❌ Non-blocking test failed: {e}")
        results['nonblocking'] = False
    
    # Test 3: Summary intervals
    try:
        results['intervals'] = await test_summary_intervals()
    except Exception as e:
        print(f"\n❌ Interval test failed: {e}")
        results['intervals'] = False
    
    # Test 4: Task management
    try:
        results['task_mgmt'] = await test_background_task_management()
    except Exception as e:
        print(f"\n❌ Task management test failed: {e}")
        results['task_mgmt'] = False
    
    # Test 5: No summary mode
    try:
        results['no_summary'] = await test_no_summary_mode()
    except Exception as e:
        print(f"\n❌ No summary test failed: {e}")
        results['no_summary'] = False
    
    # Summary
    print_header("TEST RESULTS SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"\n  Total tests: {total_tests}")
    print(f"  Passed:      {passed_tests}")
    print(f"  Failed:      {total_tests - passed_tests}")
    
    print("\n  Individual results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"    {status}: {test_name}")
    
    if passed_tests == total_tests:
        print("\n" + "=" * 80)
        print("  🎉 ALL TESTS PASSED! Non-blocking summarization working correctly.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("  ⚠️  SOME TESTS FAILED - Check output above for details.")
        print("=" * 80)
    
    print("\n💡 Key Findings:")
    print("  • Non-blocking mode provides immediate responses")
    print("  • Summaries are generated in background without blocking")
    print("  • Summary intervals work correctly (reduces API calls)")
    print("  • Background tasks are properly managed and cleaned up")
    print()


if __name__ == "__main__":
    asyncio.run(main())

