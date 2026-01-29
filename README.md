# Mem2Disk - Text-Based AI Memory System

A high-performance, grep-friendly conversational memory system with **non-blocking background summarization** for AI agents and chatbots.

## 🚀 Key Features

✅ **Non-Blocking Summarization** - 5-10x faster conversation processing  
✅ **Smart Intervals** - Summarize every N turns (default: 10) - 90% fewer LLM calls  
✅ **Plain Text Storage** - No vector databases, just grep-friendly text files  
✅ **LLM-Powered Summaries** - Intelligent memory compaction using NVIDIA NIM  
✅ **Hierarchical Memory** - Direct, daily, weekly, and monthly summaries  
✅ **Grep/Sed/Awk Ready** - Structured anchors for easy text processing  
✅ **Background Tasks** - Async summarization doesn't block conversation  
✅ **Configurable** - Adjust summary intervals for your use case  

## 🎯 What's New: Non-Blocking Mode + Smart Intervals

Version 2.0 introduces **background summarization** with **configurable intervals** - conversations continue immediately while summaries are generated asynchronously:

```python
# Initialize with summary interval
memory_ops = MemoryOps(
    username="user123",
    summary_interval=10  # Summarize every 10 turns (default)
)

# Process message without blocking!
result = await memory_ops.process_message(
    message="What is a quadratic equation?",
    bot_response="A quadratic equation is...",
    background_summary=True  # Non-blocking (default)
)

# Conversation continues immediately!
# Summary generated in background at turns 10, 20, 30, etc.
```

**Performance**: 5-10x faster + 90% fewer LLM calls!

## 📦 Installation

```bash
# Clone repository
git clone <repo-url>
cd Mem2Disk

# Install dependencies
pip install -r requirements.txt

# Set NVIDIA API key
export NVIDIA_API_KEY='your-key-here'
```

### Requirements

- Python 3.8+
- langchain-nvidia-ai-endpoints
- langchain-core
- colorama
- pyyaml

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from agent_memory import MemoryOps
from langchain_nvidia_ai_endpoints import ChatNVIDIA

async def main():
    # Initialize memory system
    memory_ops = MemoryOps(
        username="user123",
        llm=ChatNVIDIA(model="nvidia/llama-3.3-nemotron-super-49b-v1.5"),
        rate_limit_delay=1.0,
        summary_interval=10  # Summarize every 10 turns (default)
    )
    
    # Process messages (non-blocking by default)
    # Summaries are created at turns 10, 20, 30, etc.
    await memory_ops.process_message(
        message="Hi! Can you help me with math?",
        bot_response="Of course! I'd be happy to help."
    )
    
    # Continue conversation immediately!
    await memory_ops.process_message(
        message="What is 2 + 2?",
        bot_response="2 + 2 equals 4."
    )
    
    # Wait for summaries before exit
    await memory_ops.wait_for_summaries()

asyncio.run(main())
```

### Configure Summary Interval

```python
# Summarize more frequently (every 5 turns)
memory_ops = MemoryOps(
    username="user123",
    summary_interval=5
)

# Summarize less frequently (every 20 turns) - fewer LLM calls
memory_ops = MemoryOps(
    username="user123",
    summary_interval=20
)

# Summarize every turn (legacy behavior)
memory_ops = MemoryOps(
    username="user123",
    summary_interval=1
)
```

### Advanced Usage

```python
# Monitor background tasks
result = await memory_ops.process_message(user_msg, bot_msg)
print(f"Background tasks: {result['background_tasks']}")

# Check pending summaries
pending = memory_ops.get_pending_summaries_count()
print(f"{pending} summaries still processing")

# Wait for summaries with timeout
await memory_ops.wait_for_summaries(timeout=60)
```

## 📁 Memory File Format

Conversations are stored as plain text with grep-friendly anchors:

```
@@@MEMORY_LOG_START@@@
@USERNAME:user123@
@USER_ID:user123@
@TOTAL_TURNS:100@

###SUMMARY_START###
Today I worked with the student on quadratic equations...
###SUMMARY_END###

>>>TURNS_START<<< (Total: 100)

<<<TURN:0001>>>
@TURN_ID:abc-123@
@TIMESTAMP:2026-01-28T10:30:00@
>>>USER:0001>>>
What is a quadratic equation?
<<<USER:0001<<<

>>>BOT:0001>>>
A quadratic equation is an equation of the form ax² + bx + c = 0...
<<<BOT:0001<<<

>>>SUMMARY:0001>>>
I just explained what quadratic equations are.
<<<SUMMARY:0001<<<
<<<END_TURN:0001>>>
```

## 🔍 Searching Memory

### Grep Examples

```bash
# Find all conversation turns
grep '<<<TURN:' conversation_memory.txt

# View specific turn with content
grep -A 20 '<<<TURN:0005>>>' conversation_memory.txt

# Extract entire turn
sed -n '/<<<TURN:0003>>>/,/<<<END_TURN:0003>>>/p' conversation_memory.txt

# Search for keywords with context
grep -i -C 5 "quadratic" conversation_memory.txt

# Get conversation summary
sed -n '/###SUMMARY_START###/,/###SUMMARY_END###/p' conversation_memory.txt

# Count total turns
grep -c '<<<TURN:' conversation_memory.txt
```

See `GREP_EXAMPLES.md` for more search patterns.

## 🧪 Testing

### Run Non-Blocking Tests (Recommended)

```bash
python test_nonblocking.py
```

**Focused tests for non-blocking functionality:**
- TEST 1: Blocking mode performance
- TEST 2: Non-blocking mode performance  
- TEST 3: Summary interval validation
- TEST 4: Background task management
- TEST 5: No-summary mode (high interval)

**Shows:** Performance comparison, speedup metrics, interval validation

### Run Full Memory System Tests

```bash
python test_memory_system.py
```

**Comprehensive tests:**
- TEST 0: Blocking vs Non-blocking comparison (5 turns)
- TEST 1: Create 100 turns with background summaries
- TEST 2: Grep/glob/sed search validation

### Run Demo

```bash
python demo_nonblocking.py
```

**Interactive demonstration:**
- Performance comparison (blocking vs non-blocking)
- Real-time conversation with background tasks
- API usage examples

## 📊 Performance Benchmarks

### Blocking vs Non-Blocking (5 turns, summarize every turn)

| Mode | Time | UX |
|------|------|-----|
| **Blocking** | 12.5s | Waits for summaries |
| **Non-Blocking** | 2.5s | Immediate responses |
| **Speedup** | **5x** | Much better! |

### 100 Conversation Turns (summary every 10 turns)

| Mode | Time | LLM Calls | Notes |
|------|------|-----------|-------|
| **Blocking (every turn)** | ~250s | 100 | User waits each turn |
| **Non-Blocking (every turn)** | ~50s | 100 | Summaries in background |
| **Non-Blocking (every 10)** | ~15s | 10 | **Best performance!** |
| **Speedup** | **16x** | **90% fewer calls** | Optimal!

## 📚 Documentation

- **[GREP_EXAMPLES.md](GREP_EXAMPLES.md)** - Grep/sed search patterns for memory files

## 🏗️ Architecture

### Non-Blocking Flow

```
User Message
    ↓
Save Interaction (immediate)
    ↓
Return to User (no blocking!)
    ↓
Background Task: Generate Summary
    ↓
Update Interaction with Summary
    ↓
Save to File
```

### Components

1. **MemoryHandler** - Core memory management with background tasks
2. **MemoryOps** - High-level operations and conversation management
3. **Background Tasks** - Async summarization using asyncio
4. **Text Files** - Grep-friendly storage with structured anchors

## 🔧 API Reference

### MemoryOps

```python
class MemoryOps:
    def __init__(
        username: str,
        llm: ChatNVIDIA = None,
        memory_dir: str = None,
        rate_limit_delay: float = 2.0,
        summary_interval: int = 10  # Summarize every N turns
    )
    
    async def process_message(
        message: str,
        bot_response: str,
        background_summary: bool = True
    ) -> Dict[str, Any]
    # Returns dict with 'is_summary_turn' indicating if summary was created
    
    async def wait_for_summaries(timeout: float = None)
    
    def get_pending_summaries_count() -> int
    
    def search_memory_text(pattern: str) -> List[str]
```

### MemoryHandler

```python
class MemoryHandler:
    async def create_memory_summary(
        content: str,
        period_type: PeriodType
    ) -> str
    
    def add_interaction(
        user_msg: str,
        bot_msg: str,
        turn_number: int
    ) -> Dict
    
    def update_interaction_summary(
        turn_number: int,
        summary: str
    ) -> bool
```

## 🎓 Examples

### Example 1: Chat Bot

```python
async def chatbot(user_id: str):
    memory = MemoryOps(username=user_id)
    
    while True:
        user_msg = input("You: ")
        if user_msg.lower() == 'quit':
            break
        
        bot_response = generate_response(user_msg)
        
        # Non-blocking save
        await memory.process_message(user_msg, bot_response)
        
        print(f"Bot: {bot_response}")
    
    # Wait for summaries before exit
    await memory.wait_for_summaries()

asyncio.run(chatbot("user123"))
```

### Example 2: Batch Processing

```python
async def process_conversations(conversations: List[Tuple[str, str]]):
    memory = MemoryOps(username="batch_user")
    
    # Process all conversations (non-blocking)
    for user_msg, bot_msg in conversations:
        result = await memory.process_message(user_msg, bot_msg)
        pending = result['background_tasks']
        print(f"Processed. {pending} summaries pending")
    
    # Wait for all summaries
    print("Waiting for summaries...")
    await memory.wait_for_summaries(timeout=120)
    print("All done!")
```

### Example 3: Search and Retrieve

```python
# Search memory for specific topics
matches = memory.search_memory_text(r"quadratic.*equation")
print(f"Found {len(matches)} matches")

# Get conversation context
context = memory.get_memory_context("algebra homework")
print(context)

# Get overall summary
summary = memory.get_history_summary()
print(summary)
```

## 🔒 Best Practices

1. **Use non-blocking by default** for better UX
2. **Wait for summaries** before shutdown/cleanup
3. **Set timeouts** to prevent hanging
4. **Monitor pending tasks** in production
5. **Adjust rate limiting** to avoid API limits

## 🐛 Troubleshooting

### Summaries not appearing?

```python
# Wait for background tasks to complete
await memory_ops.wait_for_summaries(timeout=60)
```

### Too many API calls?

```python
# Increase rate limit delay
memory_ops = MemoryOps(
    username="user",
    rate_limit_delay=2.0  # 2 seconds between calls
)
```

### Tasks timing out?

```python
# Increase timeout
await memory_ops.wait_for_summaries(timeout=300)  # 5 minutes
```

## 🛣️ Roadmap

- [x] Non-blocking summarization
- [x] Background task management
- [x] Grep-friendly text format
- [ ] Batch summarization
- [ ] Priority queue for summaries
- [ ] Configurable concurrency limits
- [ ] Progress callbacks
- [ ] Persistent task queue

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📞 Support

- Report issues on GitHub
- See documentation in `/docs`
- Run tests to verify functionality

## 🙏 Acknowledgments

Based on [standalone_agent_memory](https://github.com/Zenodia/standalone_agent_memory) by Zenodia.

Enhanced with:
- Non-blocking background summarization
- Async task management
- Grep-friendly text format
- Comprehensive test suite

---

**Version:** 2.1.0 (Smart Intervals + Non-Blocking)  
**Last Updated:** 2026-01-28  
**Status:** Production Ready ✅  
**Key Features:** 90% fewer LLM calls, 16x faster processing
