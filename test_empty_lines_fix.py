#!/usr/bin/env python3
"""Test to verify that empty lines don't accumulate between summary and turns."""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from agent_memory import MemoryOps, clear_user_memory

async def test_empty_lines_fix():
    """Test that empty lines don't accumulate with each turn."""
    
    test_user = "test_empty_lines_user"
    test_memory_dir = "./test_memory_empty_lines"
    
    # Clean up any existing test data
    clear_user_memory(test_user)
    if Path(test_memory_dir).exists():
        subprocess.run(['rm', '-rf', test_memory_dir], check=True)
    
    print("=" * 80)
    print("Testing empty lines fix...")
    print("=" * 80)
    
    # Create memory ops with short summary interval for testing
    memory_ops = MemoryOps(
        username=test_user,
        memory_dir=test_memory_dir,
        summary_interval=10  # Create summary every 10 turns
    )
    
    memory_file = Path(test_memory_dir) / "conversation_memory.txt"
    
    def count_empty_lines_after_summary():
        """Count consecutive empty lines between ###SUMMARY_END### and >>>TURNS_START<<<."""
        if not memory_file.exists():
            return 0
        
        with open(memory_file, 'r') as f:
            lines = f.readlines()
        
        # Find ###SUMMARY_END###
        summary_end_idx = None
        turns_start_idx = None
        
        for i, line in enumerate(lines):
            if '###SUMMARY_END###' in line:
                summary_end_idx = i
            if '>>>TURNS_START<<<' in line:
                turns_start_idx = i
                break
        
        if summary_end_idx is None or turns_start_idx is None:
            return 0
        
        # Count empty lines between them
        empty_count = 0
        for i in range(summary_end_idx + 1, turns_start_idx):
            if lines[i].strip() == '':
                empty_count += 1
            else:
                break  # Stop at first non-empty line
        
        return empty_count
    
    # Add multiple interactions to trigger summary updates
    empty_line_counts = []
    
    for turn in range(1, 35):  # 35 turns will trigger 3 summary updates (at turns 10, 20, 30)
        message = f"Test message {turn}"
        response = f"Test response {turn}"
        
        result = await memory_ops.process_message(
            message=message,
            bot_response=response,
            create_summary=True,
            background_summary=False  # Use blocking mode for testing
        )
        
        # Count empty lines after each turn
        empty_count = count_empty_lines_after_summary()
        empty_line_counts.append(empty_count)
        
        print(f"Turn {turn:2d}: {empty_count} empty line(s) after summary")
        
        # Verify empty lines don't accumulate
        if turn > 10 and empty_count > 1:
            print(f"\n❌ FAIL: Too many empty lines at turn {turn}: {empty_count}")
            print(f"   Empty line counts: {empty_line_counts}")
            return False
    
    print("\n" + "=" * 80)
    print("Empty line count history:")
    print(f"  {empty_line_counts}")
    print("=" * 80)
    
    # Verify the final count is exactly 1
    final_count = count_empty_lines_after_summary()
    if final_count == 1:
        print(f"\n✅ PASS: Exactly 1 empty line after summary (expected)")
    else:
        print(f"\n❌ FAIL: Found {final_count} empty lines, expected 1")
        return False
    
    # Verify no accumulation happened
    # After turn 10, there should be a summary, and empty lines should stabilize at 1
    summary_turn_counts = [empty_line_counts[i] for i in [9, 19, 29] if i < len(empty_line_counts)]
    print(f"\nEmpty lines at summary turns (10, 20, 30): {summary_turn_counts}")
    
    if all(count == 1 for count in summary_turn_counts):
        print("✅ PASS: Empty lines remained constant at 1 across all summary updates")
    else:
        print(f"❌ FAIL: Empty lines not constant: {summary_turn_counts}")
        return False
    
    # Clean up
    subprocess.run(['rm', '-rf', test_memory_dir], check=True)
    clear_user_memory(test_user)
    
    return True

if __name__ == "__main__":
    result = asyncio.run(test_empty_lines_fix())
    sys.exit(0 if result else 1)

