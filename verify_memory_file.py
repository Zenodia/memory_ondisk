#!/usr/bin/env python3
"""
Script to verify and diagnose memory file integrity.
Usage: python verify_memory_file.py [memory_file_path]
"""

import sys
import subprocess
from pathlib import Path
from colorama import Fore, init

init(autoreset=True)


def verify_memory_file(file_path: str):
    """Verify memory file integrity and show diagnostics."""
    
    file = Path(file_path)
    
    if not file.exists():
        print(Fore.RED + f"✗ File does not exist: {file_path}")
        return False
    
    print(Fore.CYAN + f"📄 Analyzing memory file: {file.name}")
    print(Fore.CYAN + f"   Path: {file.absolute()}")
    print()
    
    # Get file size
    file_size = file.stat().st_size
    print(Fore.CYAN + f"File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    print()
    
    # Count turns
    result = subprocess.run(
        ['grep', '-c', '<<<TURN:', str(file)],
        capture_output=True,
        text=True
    )
    turn_count = int(result.stdout.strip()) if result.returncode == 0 else 0
    
    # Get turn numbers
    result = subprocess.run(
        ['grep', '-o', '<<<TURN:[0-9]\\{4\\}>>>', str(file)],
        capture_output=True,
        text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        turn_numbers = [line.split(':')[1].split('>')[0] for line in result.stdout.strip().split('\n')]
        turn_numbers_int = [int(t) for t in turn_numbers]
        min_turn = min(turn_numbers_int) if turn_numbers_int else 0
        max_turn = max(turn_numbers_int) if turn_numbers_int else 0
    else:
        turn_numbers_int = []
        min_turn = max_turn = 0
    
    print(Fore.GREEN + f"✓ Total turns found: {turn_count}")
    if turn_count > 0:
        print(Fore.CYAN + f"  Turn range: {min_turn} to {max_turn}")
        
        # Check for missing turns
        if turn_count > 0:
            expected_turns = set(range(min_turn, max_turn + 1))
            actual_turns = set(turn_numbers_int)
            missing_turns = expected_turns - actual_turns
            
            if missing_turns:
                print(Fore.RED + f"  ✗ Missing turns: {sorted(missing_turns)}")
            else:
                print(Fore.GREEN + f"  ✓ All turns from {min_turn} to {max_turn} are present (no gaps)")
    print()
    
    # Count other elements
    result = subprocess.run(
        ['grep', '-c', '>>>USER:', str(file)],
        capture_output=True,
        text=True
    )
    user_count = int(result.stdout.strip()) if result.returncode == 0 else 0
    
    result = subprocess.run(
        ['grep', '-c', '>>>BOT:', str(file)],
        capture_output=True,
        text=True
    )
    bot_count = int(result.stdout.strip()) if result.returncode == 0 else 0
    
    result = subprocess.run(
        ['grep', '-c', '>>>SUMMARY:', str(file)],
        capture_output=True,
        text=True
    )
    summary_count = int(result.stdout.strip()) if result.returncode == 0 else 0
    
    print(Fore.CYAN + f"User messages: {user_count}")
    print(Fore.CYAN + f"Bot messages: {bot_count}")
    print(Fore.CYAN + f"Summaries: {summary_count}")
    print()
    
    # Check essential markers
    essential_markers = [
        '@@@MEMORY_LOG_START@@@',
        '@@@MEMORY_LOG_END@@@',
        '>>>TURNS_START<<<',
        '>>>TURNS_END<<<'
    ]
    
    print(Fore.CYAN + "Essential markers:")
    all_markers_present = True
    for marker in essential_markers:
        result = subprocess.run(
            ['grep', '-q', marker, str(file)],
            capture_output=True
        )
        present = (result.returncode == 0)
        all_markers_present = all_markers_present and present
        status = Fore.GREEN + "✓" if present else Fore.RED + "✗"
        print(f"  {status} {marker}")
    print()
    
    # Overall integrity check
    integrity_ok = (turn_count == user_count == bot_count and all_markers_present and len(missing_turns) == 0)
    
    if integrity_ok:
        print(Fore.GREEN + "=" * 60)
        print(Fore.GREEN + "✓ FILE INTEGRITY CHECK PASSED")
        print(Fore.GREEN + f"✓ All {turn_count} turns are present and complete")
        print(Fore.GREEN + "=" * 60)
    else:
        print(Fore.RED + "=" * 60)
        print(Fore.RED + "✗ FILE INTEGRITY ISSUES DETECTED")
        if turn_count != user_count:
            print(Fore.RED + f"  - Turn count ({turn_count}) != User message count ({user_count})")
        if turn_count != bot_count:
            print(Fore.RED + f"  - Turn count ({turn_count}) != Bot message count ({bot_count})")
        if not all_markers_present:
            print(Fore.RED + f"  - Some essential markers are missing")
        if missing_turns:
            print(Fore.RED + f"  - Missing {len(missing_turns)} turn(s)")
        print(Fore.RED + "=" * 60)
    
    print()
    print(Fore.CYAN + "To view specific turns:")
    print(Fore.YELLOW + f"  grep -A 20 '<<<TURN:0001>>>' {file}")
    print(Fore.CYAN + "To view all turn markers:")
    print(Fore.YELLOW + f"  grep '<<<TURN:' {file}")
    print(Fore.CYAN + "To view a complete turn:")
    print(Fore.YELLOW + f"  sed -n '/<<<TURN:0001>>>/,/<<<END_TURN:0001>>>/p' {file}")
    
    return integrity_ok


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default to test memory file
        file_path = "test_memory/conversation_memory.txt"
    
    success = verify_memory_file(file_path)
    sys.exit(0 if success else 1)

