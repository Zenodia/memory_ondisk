# 🔍 Grep Reference - Mem2Disk Memory Files (Linux/Unix)

## The Problem
When you grep for just anchors like `<<<TURN:`, you only get the anchor lines:
```bash
grep '<<<TURN:' test_memory/conversation_memory.txt
<<<TURN:0001>>>
<<<TURN:0002>>>
<<<TURN:0003>>>
```
**This is NOT helpful!** You need the actual conversation content.

## The Solution: Use Context Flags (-A, -B, -C)!

### Essential Commands

#### View Turn with Content
```bash
# Show turn marker + 20 lines after (the conversation)
grep -A 20 '<<<TURN:0001>>>' test_memory/conversation_memory.txt

# View entire turn (from start to end marker)
sed -n '/<<<TURN:0001>>>/,/<<<END_TURN:0001>>>/p' test_memory/conversation_memory.txt
```

#### View User Messages with Content
```bash
# Show user message marker + 1 line after (the actual message)
grep -A 1 '>>>USER:' test_memory/conversation_memory.txt

# Get specific user message from turn 3
sed -n '/>>>USER:0003>>>/,/<<<USER:0003<<</p' test_memory/conversation_memory.txt
```

#### View Bot Responses with Content
```bash
# Show bot response marker + 1 line after (the actual response)
grep -A 1 '>>>BOT:' test_memory/conversation_memory.txt

# Get specific bot response from turn 3
sed -n '/>>>BOT:0003>>>/,/<<<BOT:0003<<</p' test_memory/conversation_memory.txt
```

#### Search Keywords with Context
```bash
# Search for "algebra" with 5 lines before and after
grep -i -C 5 'algebra' test_memory/conversation_memory.txt

# Search with just lines after
grep -i -A 10 'quadratic' test_memory/conversation_memory.txt

# Search with just lines before
grep -i -B 5 'equations' test_memory/conversation_memory.txt
```

#### Get Summary
```bash
# Extract the entire summary section
sed -n '/###SUMMARY_START###/,/###SUMMARY_END###/p' test_memory/conversation_memory.txt
```

#### Count Things
```bash
# Count total turns
grep -c '<<<TURN:' test_memory/conversation_memory.txt

# Count user messages
grep -c '>>>USER:' test_memory/conversation_memory.txt

# Count bot responses
grep -c '>>>BOT:' test_memory/conversation_memory.txt
```


## Grep Flags Reference

### Context Flags (Most Important!)
- `-A N` = Show N lines **After** the match
- `-B N` = Show N lines **Before** the match  
- `-C N` = Show N lines of **Context** (both before and after)

### Other Useful Flags
- `-i` = Case **insensitive** search
- `-c` = **Count** matches only
- `-o` = Show **only** the matched part
- `-n` = Show **line numbers**
- `-v` = **Invert** match (show non-matching lines)
- `-E` = Use **extended** regex
- `-r` or `-R` = **Recursive** search in directories

## Real Examples

### Example 1: Find what Sarah said in turn 3
```bash
sed -n '/>>>USER:0003>>>/,/<<<USER:0003<<</p' test_memory/conversation_memory.txt
```

### Example 2: See entire conversation from turn 2
```bash
sed -n '/<<<TURN:0002>>>/,/<<<END_TURN:0002>>>/p' test_memory/conversation_memory.txt
```

### Example 3: Find all mentions of "algebra" with surrounding context
```bash
grep -i -C 5 'algebra' test_memory/conversation_memory.txt
```

### Example 4: Find all user messages and clean output
```bash
# Show user messages with content, exclude marker lines
grep -A 1 '>>>USER:' test_memory/conversation_memory.txt | grep -v '>>>'
```

### Example 5: Search with line numbers
```bash
grep -n -i -C 3 'quadratic' test_memory/conversation_memory.txt
```

## 💡 Pro Tips

1. **Always use context flags** (`-A`, `-B`, `-C`) when searching for anchors - never grep anchors alone!
2. **Use sed for extracting sections** between start/end markers for complete conversations
3. **Combine grep with pipes**: `grep -A 1 '>>>USER:' | grep -v '>>>'` to exclude markers
4. **Use -n flag** to see line numbers: `grep -n -i -C 5 'keyword'`
5. **Create bash aliases** for frequent searches in your `.bashrc`:
   ```bash
   alias memgrep='grep -i -C 5'
   alias memturn='sed -n "/<<<TURN:\$1>>>/,/<<<END_TURN:\$1>>>/p"'
   ```
6. **Pipe to less** for long output: `grep -A 50 '<<<TURN:0001>>>' file.txt | less`

## Try It Now!

After running `python quick_test.py` or `python test_agent_memory.py`, try these commands:

```bash
# See your first conversation with context
grep -A 20 '<<<TURN:0001>>>' test_memory/conversation_memory.txt

# View the entire first turn
sed -n '/<<<TURN:0001>>>/,/<<<END_TURN:0001>>>/p' test_memory/conversation_memory.txt

# Find all discussions about "algebra" with context
grep -i -C 5 'algebra' test_memory/conversation_memory.txt

# Get just the user messages
grep -A 1 '>>>USER:' test_memory/conversation_memory.txt | grep -v '>>>'
```

**Now you'll see the actual conversation, not just the markers!** 🎉

