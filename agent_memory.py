import os
import json
import uuid
import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from colorama import Fore
from enum import Enum
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import asyncio
import yaml


class PeriodType(Enum):
    """Period types for hierarchical memory compaction."""
    DIRECT = "just now"
    DAILY = "today"
    WEEKLY = "this week"
    MONTHLY = "this month"


@dataclass
class PeriodMeta:
    """Metadata for each period type."""
    max_length: str
    focus: str
    time_period: str
    example: str


# Period metadata configuration
PERIOD_META: Dict[PeriodType, PeriodMeta] = {
    PeriodType.DIRECT: PeriodMeta(
        max_length="1-2 sentences",
        focus="""
- what medium did this interaction take place through (sms, phone, email, video call, etc)?
- who did you interact with?
- what did you do?
- did any new information come up that should be remembered?
- did you notice anything odd?
- what information should be remembered for the rest of the day? week? month? year?

Since tool calls might have several repeated calls, make sure to only summarize content taken based on the NEW information from the tools.
""",
        time_period="just now",
        example="""
- Book myself for 2 hours between 10-11 on Friday to learn this topic.
- I just upload an image, can you summarize the content and integrate into the study material as supplement?
"""
    ),
    PeriodType.DAILY: PeriodMeta(
        max_length="2-3 sentences",
        focus="""
- what happened today? did you talk to anyone? did you prep anything?
- who was involved in the day? did anything noticeable happen?
- what progress was made toward goals?
- what did you notice from the day?
- what information should be remembered for the rest of the week? month? year?
""",
        time_period="today",
        example="""
- I studied hard for driving test today, I even watched video on youtube to learn more about practice driving, however when I took the quiz, I don't feel confident on the answers I gave.
- Noticed that I completed the first sub topic of the lesson, I feel like I reached a milestone, although there are many more to go, I hope I won't forget about what I learned.
- I find my understanding of how to build a robust agentic system is superficial, I wonder where can I find more info to quickly grasp the key concepts?
- I feedback on the UI today, the AI assistant is behaving strangely, I asked for a youtube link related to the topic, but it gave me a video which has nothing to do with the topic. 
"""
    ),
    PeriodType.WEEKLY: PeriodMeta(
        max_length="3-4 sentences",
        focus="""
- what happened this week?
- what stood out to you? was anything odd?
- what progress was made toward your (or the student's) goals?
- did you have any lessons this week with the student?
- what difficulties have you observed in the student this week? what went bad/well?
- what information should be remembered for the rest of the month? how about year?
""",
        time_period="this week",
        example="""
- This week, I conducted three productive sessions with Sarah focusing on algebra and geometry. She demonstrated significant improvement in solving quadratic equations.
- During our Thursday session, I noticed Sarah's problem-solving speed has increased by about 40%. Based on this progress, I've decided to introduce more challenging practice problems and implement a structured review system to reinforce these concepts.
- Michael scheduled 3 study sessions this week, but his progress is minimal. After reviewing our last interaction where he expressed conflicts in scheduling with his current work for 2 more weeks, I will wait him to contact me when he is free.
"""
    ),
}


class MemoryHandler:
    """
    Enhanced Memory Handler with LLM-based fact extraction and intelligent routing.
    
    Based on: https://github.com/Zenodia/standalone_agent_memory/blob/main/MemoryManager.py
    """
    
    def __init__(
        self, 
        username: str, 
        llm: ChatNVIDIA = None,
        memory_dir: str = None,
        use_streaming: bool = False,
        rate_limit_delay: float = 2.0,  # Delay between LLM calls to avoid rate limits
        summary_interval: int = 10  # Create summaries every N turns
    ):
        """
        Initialize the Enhanced Memory Handler.
        
        Args:
            username: User ID for memory storage
            llm: ChatNVIDIA instance for LLM operations
            memory_dir: Directory to store memory files
            use_streaming: Whether to use streaming for LLM responses
            rate_limit_delay: Seconds to wait between LLM calls (default 2.0)
            summary_interval: Create summaries every N turns (default 10)
        """
        self.username = username
        self.user_id = username  # Alias for compatibility
        self.current_input = ""
        self.use_streaming = use_streaming
        self.datetime = datetime.now().strftime("%Y-%m-%d")
        self.config = None
        self.rate_limit_delay = rate_limit_delay
        self.last_llm_call_time = 0  # Track last LLM call for rate limiting
        self.turn_counter = 0  # Track conversation turns
        self.background_tasks = []  # Track background summarization tasks
        self.summary_interval = summary_interval  # Create summaries every N turns
        
        # Set up memory directory
        if memory_dir is None:
            try:
                docker_compose_path = Path("/workspace/docker-compose.yml")
                if docker_compose_path.exists():
                    with open(docker_compose_path, "r") as f:
                        yaml_data = yaml.safe_load(f)
                        mnt_folder = yaml_data["services"]["agenticta"]["volumes"][-1].split(":")[-1]
                        memory_dir = os.path.join(mnt_folder, username, "memory")
                else:
                    memory_dir = os.path.join("mnt", username, "memory")
            except Exception as e:
                print(Fore.YELLOW + f"Could not load mnt_folder from docker-compose.yml: {e}", Fore.RESET)
                memory_dir = os.path.join("mnt", username, "memory")
        
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "conversation_memory.txt"
        
        # Initialize LLM
        if llm is None:
            # Default LLM setup
            self.llm = ChatNVIDIA(
                model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                temperature=0.6,
                api_key=os.getenv("NVIDIA_API_KEY")
            )
        else:
            self.llm = llm
        
        # Memory settings (NO VECTOR STORE - pure text-based)
        self.summary = ""
        self._all_interactions = []  # Store raw interactions for text file
        
        # Create memory extraction chain with Orin-style prompt
        memory_extract_prompt = """You are Ollie, an AI tutor, creating your own memory summary from {time_period}. Write in FIRST PERSON ("I worked with...", "My student showed...").

Keep your summary to {max_length} maximum.

Here is your past memory if you'd like to incorporate any aspects of it into your response.
Do not summarize this or include it in your response – this is just background information:

— BEGIN BACKGROUND INFORMATION —
{existing_memory}
— END BACKGROUND INFORMATION —

This is the new content that you must summarize:
{content}

For your summary focus on things like:
{focus}

A good summary for this time period looks like this:
{example}

CRITICAL: Only use information explicitly stated below. Do NOT add details or infer anything.

Your summary:
"""
        
        extract_prompt_template = PromptTemplate(
            input_variables=["time_period", "max_length", "existing_memory", "content", "focus", "example"],
            template=memory_extract_prompt,
        )
        self.mem_extract_chain = (extract_prompt_template | self.llm | StrOutputParser())
        
        # Load existing memories
        self.load_memory_from_file()
        
        print(Fore.GREEN + f"✓ Text-Based Memory Handler initialized for user: {username}", Fore.RESET)
        print(Fore.CYAN + f"  Memory file: {self.memory_file}", Fore.RESET)
        print(Fore.CYAN + f"  Mode: Plain text with grep-friendly anchors (NO vector store)", Fore.RESET)
    
    async def _rate_limit_wait(self):
        """Wait to avoid rate limits between LLM calls."""
        if self.last_llm_call_time > 0:
            elapsed = time.time() - self.last_llm_call_time
            if elapsed < self.rate_limit_delay:
                wait_time = self.rate_limit_delay - elapsed
                print(Fore.YELLOW + f"Rate limiting: waiting {wait_time:.1f}s...", Fore.RESET)
                await asyncio.sleep(wait_time)
    
    def cleanup_background_tasks(self):
        """Remove completed background tasks from tracking list."""
        self.background_tasks = [task for task in self.background_tasks if not task.done()]
    
    async def wait_for_background_tasks(self, timeout: float = None):
        """
        Wait for all background summarization tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
        """
        if not self.background_tasks:
            return
        
        print(Fore.CYAN + f"Waiting for {len(self.background_tasks)} background tasks...", Fore.RESET)
        
        try:
            if timeout:
                await asyncio.wait_for(
                    asyncio.gather(*self.background_tasks, return_exceptions=True),
                    timeout=timeout
                )
            else:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            print(Fore.GREEN + "✓ All background tasks completed", Fore.RESET)
        except asyncio.TimeoutError:
            print(Fore.YELLOW + f"Warning: Some background tasks timed out after {timeout}s", Fore.RESET)
        finally:
            self.cleanup_background_tasks()
    
    async def _background_summarize_and_update(
        self,
        turn_number: int,
        content: str,
        period_type: PeriodType,
        existing_memory: str
    ):
        """
        Background task: Create summary and update interaction.
        
        Args:
            turn_number: Turn number to update
            content: Content to summarize
            period_type: Period type for summary
            existing_memory: Existing memory context
        """
        try:
            # Create the summary
            summary = await self.create_memory_summary(
                content=content,
                period_type=period_type,
                existing_memory=existing_memory
            )
            
            # Update the interaction with the summary
            if summary:
                self.update_interaction_summary(turn_number, summary)
        except Exception as e:
            print(Fore.RED + f"Error in background summarization for turn {turn_number}: {e}", Fore.RESET)
    
    async def create_memory_summary(
        self, 
        content: str, 
        period_type: PeriodType = PeriodType.DIRECT,
        existing_memory: str = "None",
        max_retries: int = 3
    ) -> str:
        """
        Create a memory summary for a given period type using LLM with retry logic.
        
        Args:
            content: The new content to summarize
            period_type: The period type (DIRECT, DAILY, WEEKLY, MONTHLY)
            existing_memory: Previous memory to incorporate (optional)
            max_retries: Number of retry attempts on rate limit
            
        Returns:
            Summary string
        """
        # Rate limiting: wait if needed
        await self._rate_limit_wait()
        
        # Get period metadata
        meta = PERIOD_META[period_type]
        
        output = ""
        inputs = {
            "time_period": meta.time_period,
            "max_length": meta.max_length,
            "existing_memory": existing_memory,
            "content": content,
            "focus": meta.focus,
            "example": meta.example
        }
        
        # Retry logic for rate limits
        for attempt in range(max_retries):
            try:
                # Use astream for streaming-compatible execution
                async for chunk in self.mem_extract_chain.astream(inputs):
                    if chunk:
                        output += str(chunk)
                
                # Update last call time on success
                self.last_llm_call_time = time.time()
                break
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    print(Fore.YELLOW + f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...", Fore.RESET)
                    await asyncio.sleep(wait_time)
                    if attempt == max_retries - 1:
                        print(Fore.RED + f"Max retries reached for memory summary. Returning empty summary.", Fore.RESET)
                        return ""
                else:
                    print(Fore.RED + f"Error creating memory summary: {e}", Fore.RESET)
                    return ""
        
        summary = output.strip()
        print(Fore.LIGHTMAGENTA_EX + f"Created {period_type.value} memory summary ({len(summary)} chars)", Fore.RESET)
        return summary
    
    def add_interaction(self, user_msg: str, bot_msg: str, turn_number: int, summary: str = "") -> Dict[str, Any]:
        """
        Add a conversation interaction to memory (plain text storage).
        
        Args:
            user_msg: User message
            bot_msg: Bot response
            turn_number: Conversation turn number
            summary: Optional summary of the interaction
            
        Returns:
            Dictionary with interaction data
        """
        interaction = {
            "turn": turn_number,
            "timestamp": datetime.now().isoformat(),
            "date": self.datetime,
            "user_id": self.user_id,
            "user_message": user_msg,
            "bot_message": bot_msg,
            "summary": summary,
            "id": str(uuid.uuid4())
        }
        
        self._all_interactions.append(interaction)
        print(Fore.GREEN + f"✓ Added interaction turn #{turn_number} to memory", Fore.RESET)
        
        # Auto-save to file
        self.save_memory_to_file()
        
        return interaction
    
    def update_interaction_summary(self, turn_number: int, summary: str) -> bool:
        """
        Update the summary for an existing interaction (called by background task).
        
        Args:
            turn_number: Turn number to update
            summary: New summary text
            
        Returns:
            True if updated successfully
        """
        for interaction in self._all_interactions:
            if interaction['turn'] == turn_number:
                interaction['summary'] = summary
                print(Fore.LIGHTMAGENTA_EX + f"✓ Updated summary for turn #{turn_number} (background)", Fore.RESET)
                # Save to file with updated summary
                self.save_memory_to_file()
                return True
        
        print(Fore.YELLOW + f"Warning: Could not find turn #{turn_number} to update", Fore.RESET)
        return False
    
    def search_text_file(self, pattern: str, case_sensitive: bool = False) -> List[str]:
        """
        Search the memory text file using regex pattern (simulates grep).
        
        Args:
            pattern: Regex pattern to search for
            case_sensitive: Whether search is case sensitive
            
        Returns:
            List of matching lines
        """
        if not self.memory_file.exists():
            return []
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            flags = 0 if case_sensitive else re.IGNORECASE
            matches = []
            
            for line in content.split('\n'):
                if re.search(pattern, line, flags):
                    matches.append(line)
            
            print(Fore.CYAN + f"Found {len(matches)} matches for pattern: {pattern}", Fore.RESET)
            return matches
            
        except Exception as e:
            print(Fore.RED + f"Error searching text file: {e}", Fore.RESET)
            return []
    
    def save_memory_to_file(self) -> bool:
        """Save all interactions to plain text file with grep-friendly anchors."""
        try:
            if not hasattr(self, '_all_interactions'):
                self._all_interactions = []
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                # Header with anchors
                f.write("@@@MEMORY_LOG_START@@@\n")
                f.write(f"@USERNAME:{self.username}@\n")
                f.write(f"@USER_ID:{self.user_id}@\n")
                f.write(f"@LAST_UPDATED:{datetime.now().isoformat()}@\n")
                f.write(f"@TOTAL_TURNS:{len(self._all_interactions)}@\n")
                f.write("=" * 80 + "\n\n")
                
                # Summary section with anchor
                if self.summary:
                    f.write("###SUMMARY_START###\n")
                    f.write(f"{self.summary}\n")
                    f.write("###SUMMARY_END###\n\n")
                
                # Interaction turns with clear anchors
                f.write(f">>>TURNS_START<<< (Total: {len(self._all_interactions)})\n")
                f.write("=" * 80 + "\n\n")
                
                for interaction in self._all_interactions:
                    turn_num = interaction['turn']
                    
                    # Turn marker - easy to grep
                    f.write(f"<<<TURN:{turn_num:04d}>>>\n")
                    f.write(f"@TURN_ID:{interaction['id']}@\n")
                    f.write(f"@TIMESTAMP:{interaction['timestamp']}@\n")
                    f.write(f"@DATE:{interaction['date']}@\n")
                    f.write(f"@USER_ID:{interaction['user_id']}@\n")
                    f.write("-" * 80 + "\n")
                    
                    # User message with marker
                    f.write(f">>>USER:{turn_num:04d}>>>\n")
                    f.write(f"{interaction['user_message']}\n")
                    f.write(f"<<<USER:{turn_num:04d}<<<\n\n")
                    
                    # Bot message with marker
                    f.write(f">>>BOT:{turn_num:04d}>>>\n")
                    f.write(f"{interaction['bot_message']}\n")
                    f.write(f"<<<BOT:{turn_num:04d}<<<\n\n")
                    
                    # Summary if available
                    if interaction.get('summary'):
                        f.write(f">>>SUMMARY:{turn_num:04d}>>>\n")
                        f.write(f"{interaction['summary']}\n")
                        f.write(f"<<<SUMMARY:{turn_num:04d}<<<\n\n")
                    
                    f.write(f"<<<END_TURN:{turn_num:04d}>>>\n")
                    f.write("=" * 80 + "\n\n")
                
                f.write(">>>TURNS_END<<<\n")
                f.write("@@@MEMORY_LOG_END@@@\n")
            
            print(Fore.GREEN + f"✓ Saved {len(self._all_interactions)} interactions to {self.memory_file.name}", Fore.RESET)
            print(Fore.CYAN + f"  Use grep '<<<TURN:' to find all turns", Fore.RESET)
            print(Fore.CYAN + f"  Use grep '>>>USER:' to find all user messages", Fore.RESET)
            return True
            
        except Exception as e:
            print(Fore.RED + f"Error saving memories to file: {e}", Fore.RESET)
            import traceback
            traceback.print_exc()
            return False
    
    def get_search_examples(self) -> str:
        """
        Return examples of bash/grep commands to search the memory text file.
        Optimized for the new anchor-based format.
        """
        examples = f"""
        === GREP/BASH SEARCH EXAMPLES FOR {self.memory_file} ===
        
        # Find all conversation turns (just markers):
        grep '<<<TURN:' {self.memory_file}
        
        # Find specific turn WITH CONTENT (shows 20 lines after):
        grep -A 20 '<<<TURN:0005>>>' {self.memory_file}
        
        # View full conversation for turn 3:
        sed -n '/<<<TURN:0003>>>/,/<<<END_TURN:0003>>>/p' {self.memory_file}
        
        # Find all user messages WITH CONTENT (1 line after):
        grep -A 1 '>>>USER:' {self.memory_file}
        
        # Find all bot responses WITH CONTENT (1 line after):
        grep -A 1 '>>>BOT:' {self.memory_file}
        
        # Find user message from turn 3 WITH CONTENT:
        sed -n '/>>>USER:0003>>>/,/<<<USER:0003<<</p' {self.memory_file}
        
        # Get total number of turns:
        grep '@TOTAL_TURNS:' {self.memory_file}
        
        # Search for keyword with context (5 lines before/after):
        grep -i -C 5 "algebra" {self.memory_file}
        
        # Search for date-specific entries with content:
        grep -A 10 '@DATE:2026-01-28@' {self.memory_file}
        
        # Get conversation summary:
        sed -n '/###SUMMARY_START###/,/###SUMMARY_END###/p' {self.memory_file}
        
        # Count total turns:
        grep -c '<<<TURN:' {self.memory_file}
        
        # Find turns containing specific word with context:
        grep -C 5 "quadratic" {self.memory_file}
        
        # Get all timestamps:
        grep '@TIMESTAMP:' {self.memory_file}
        
        # Find user ID:
        grep '@USER_ID:' {self.memory_file} | head -1
        
        # Extract turn numbers only:
        grep -o '<<<TURN:[0-9]\\{{4}}>>>' {self.memory_file}
        """
        return examples
    
    def load_memory_from_file(self) -> bool:
        """Load interactions from plain text file for returning users."""
        try:
            if not self.memory_file.exists():
                print(Fore.YELLOW + f"No existing memory file found for user {self.username} (new user)", Fore.RESET)
                self._all_interactions = []
                self.turn_counter = 0
                return False
            
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse header
            last_updated = "Unknown"
            total_turns = 0
            
            # Extract metadata using anchors
            if "@LAST_UPDATED:" in content:
                match = re.search(r'@LAST_UPDATED:([^@]+)@', content)
                if match:
                    last_updated = match.group(1)
            
            if "@TOTAL_TURNS:" in content:
                match = re.search(r'@TOTAL_TURNS:([^@]+)@', content)
                if match:
                    total_turns = int(match.group(1))
            
            # Extract summary
            summary_match = re.search(r'###SUMMARY_START###\n(.+?)\n###SUMMARY_END###', content, re.DOTALL)
            if summary_match:
                self.summary = summary_match.group(1).strip()
            else:
                self.summary = ""
            
            # Parse all turns
            interactions = []
            turn_pattern = r'<<<TURN:(\d{4})>>>(.*?)<<<END_TURN:\1>>>'
            turn_matches = re.finditer(turn_pattern, content, re.DOTALL)
            
            for match in turn_matches:
                turn_num = int(match.group(1))
                turn_content = match.group(2)
                
                # Extract turn details
                turn_id_match = re.search(r'@TURN_ID:([^@]+)@', turn_content)
                timestamp_match = re.search(r'@TIMESTAMP:([^@]+)@', turn_content)
                date_match = re.search(r'@DATE:([^@]+)@', turn_content)
                user_id_match = re.search(r'@USER_ID:([^@]+)@', turn_content)
                
                # Extract user message
                user_msg_match = re.search(rf'>>>USER:{turn_num:04d}>>>\n(.*?)\n<<<USER:{turn_num:04d}<<<', turn_content, re.DOTALL)
                user_msg = user_msg_match.group(1).strip() if user_msg_match else ""
                
                # Extract bot message
                bot_msg_match = re.search(rf'>>>BOT:{turn_num:04d}>>>\n(.*?)\n<<<BOT:{turn_num:04d}<<<', turn_content, re.DOTALL)
                bot_msg = bot_msg_match.group(1).strip() if bot_msg_match else ""
                
                # Extract summary if available
                summary_match = re.search(rf'>>>SUMMARY:{turn_num:04d}>>>\n(.*?)\n<<<SUMMARY:{turn_num:04d}<<<', turn_content, re.DOTALL)
                turn_summary = summary_match.group(1).strip() if summary_match else ""
                
                interaction = {
                    "turn": turn_num,
                    "id": turn_id_match.group(1) if turn_id_match else str(uuid.uuid4()),
                    "timestamp": timestamp_match.group(1) if timestamp_match else "unknown",
                    "date": date_match.group(1) if date_match else "unknown",
                    "user_id": user_id_match.group(1) if user_id_match else self.user_id,
                    "user_message": user_msg,
                    "bot_message": bot_msg,
                    "summary": turn_summary
                }
                
                interactions.append(interaction)
            
            self._all_interactions = interactions
            self.turn_counter = max([i['turn'] for i in interactions]) if interactions else 0
            
            print(Fore.GREEN + f"✓ Loaded {len(interactions)} interactions from file (returning user)", Fore.RESET)
            print(Fore.CYAN + f"  Last updated: {last_updated}", Fore.RESET)
            print(Fore.CYAN + f"  Total turns: {self.turn_counter}", Fore.RESET)
            if self.summary:
                print(Fore.CYAN + f"  Summary: {self.summary[:100]}...", Fore.RESET)
            
            return True
            
        except Exception as e:
            print(Fore.RED + f"Error loading memories from file: {e}", Fore.RESET)
            import traceback
            traceback.print_exc()
            self._all_interactions = []
            self.turn_counter = 0
            return False


class MemoryOps:
    """
    Enhanced Memory Operations with sophisticated conversation management.
    
    Based on: https://github.com/Zenodia/standalone_agent_memory/blob/main/utils.py
    """
    
    def __init__(
        self,
        username: str,
        llm: ChatNVIDIA = None,
        memory_dir: str = None,
        use_streaming: bool = False,
        rate_limit_delay: float = 2.0,  # Delay between LLM calls
        summary_interval: int = 10  # Create summaries every N turns
    ):
        """
        Initialize Text-Based Memory Operations.
        
        Args:
            username: User ID
            llm: Optional ChatNVIDIA instance
            memory_dir: Directory for memory files
            use_streaming: Whether to use streaming
            rate_limit_delay: Seconds to wait between LLM calls (default 2.0)
            summary_interval: Create summaries every N turns (default 10)
        """
        self.username = username
        self.memory_manager = MemoryHandler(username, llm, memory_dir, use_streaming, rate_limit_delay, summary_interval)
        self.chat_history: List[BaseMessage] = []
        self.number_of_turns = 3
        
        # Load summary from memory manager
        self.summary = self.memory_manager.summary
        
        # Initialize LLM (reuse from memory_manager)
        self.llm = self.memory_manager.llm
        
        print(Fore.GREEN + f"✓ Text-Based Memory Operations initialized for user: {username}", Fore.RESET)
        print(Fore.CYAN + f"  Rate limit delay: {rate_limit_delay}s between LLM calls", Fore.RESET)
        print(Fore.CYAN + f"  Summary interval: Every {summary_interval} turns", Fore.RESET)
    
    def check_turns(self) -> int:
        """Count user message turns in chat history."""
        return sum(1 for msg in self.chat_history if isinstance(msg, HumanMessage))
    
    def conv_items_to_list_of_strs(self, chat_history: List[BaseMessage]) -> List[str]:
        """Convert message objects to string list."""
        ls = []
        for item in chat_history:
            if isinstance(item, HumanMessage):
                ls.append("Human:" + item.content)
            elif isinstance(item, AIMessage):
                ls.append("AI:" + item.content)
            elif isinstance(item, SystemMessage):
                ls.append("System:" + item.content)
        return ls
    
    async def summarize_history(self) -> str:
        """
        Progressively summarize conversation history using LangChain LLM with streaming support.
        
        Based on: https://github.com/Zenodia/standalone_agent_memory/blob/main/utils.py
        Uses astream for streaming-compatible execution.
        """
        if not self.chat_history:
            return ""
        
        conv_summary_prompt = """You are Orin, an AI tutor, creating a memory summary from your current tutoring session. Write in FIRST PERSON ("I worked with...", "My student showed...").

Keep your summary to 2-3 sentences maximum.

Here is your past memory if you'd like to incorporate any aspects of it into your response.
Do not summarize this or include it in your response – this is just background information:

— BEGIN BACKGROUND INFORMATION —
{summary}
— END BACKGROUND INFORMATION —

This is the new content from this tutoring session that you must summarize:
{conversations}

For your summary focus on things like:
- what happened today? did you talk to anyone? did you prep anything?
- who was involved in the day? did anything noticeable happen?
- what progress was made toward goals?
- what did you notice from the session?
- what information should be remembered for the rest of the week? month? year?

CRITICAL: Only use information explicitly stated in the conversation. Do NOT add details or infer anything.

Your summary:
"""
        
        # Convert chat history to string
        chat_history_ls = self.conv_items_to_list_of_strs(self.chat_history)
        conversations_str = "\n".join(chat_history_ls)
        
        # Format prompt
        conv_summary_prompt_template = PromptTemplate(
            template=conv_summary_prompt,
            input_variables=["summary", "conversations"]
        )
        
        # Use LangChain directly (compatible with ChatNVIDIA)
        summary_chain = (conv_summary_prompt_template | self.llm | StrOutputParser())
        
        try:
            # Rate limiting: wait if needed
            await self.memory_manager._rate_limit_wait()
            
            # Use astream for streaming-compatible execution
            output = ""
            async for chunk in summary_chain.astream({"summary": self.summary, "conversations": conversations_str}):
                if chunk:
                    output += str(chunk)
            
            # Update last call time on success
            self.memory_manager.last_llm_call_time = time.time()
            
            # StrOutputParser already returns a string
            if not isinstance(output, str):
                output = str(output)
            
            self.summary = output
            self.memory_manager.summary = output
            print(Fore.CYAN + f"✓ Conversation summarized ({len(self.chat_history)} messages)", Fore.RESET)
            
            # Save summary to file
            self.memory_manager.save_memory_to_file()
            
            # Reset chat history
            self.chat_history = []
            
            return output
        except Exception as e:
            print(Fore.RED + f"Error summarizing conversation: {e}", Fore.RESET)
            import traceback
            traceback.print_exc()
            return self.summary
    
    async def process_message(
        self,
        message: str,
        bot_response: str,
        context: Optional[Dict[str, Any]] = None,
        create_summary: bool = True,
        background_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Process a message exchange and save to text file.
        Summaries are only created every N turns (configured by summary_interval).
        
        Args:
            message: User message
            bot_response: Assistant response
            context: Optional context information
            create_summary: Whether to create an LLM summary (default True)
            background_summary: Whether to create summary in background (default True)
            
        Returns:
            Dictionary with memory operation results
        """
        # Add to chat history
        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=bot_response))
        
        # Increment turn counter
        self.memory_manager.turn_counter += 1
        current_turn = self.memory_manager.turn_counter
        
        # Add interaction to memory immediately (without summary - non-blocking!)
        interaction = self.memory_manager.add_interaction(
            user_msg=message,
            bot_msg=bot_response,
            turn_number=current_turn,
            summary=""  # Will be updated by background task if this is a summary turn
        )
        
        # Create interaction summary ONLY every N turns (configured interval)
        interaction_summary = ""
        should_create_summary = create_summary and (current_turn % self.memory_manager.summary_interval == 0)
        
        if should_create_summary:
            # Get the last N turns since the last summary
            start_turn = max(1, current_turn - self.memory_manager.summary_interval + 1)
            recent_interactions = [
                inter for inter in self.memory_manager._all_interactions 
                if start_turn <= inter['turn'] <= current_turn
            ]
            
            # Build content from multiple turns
            interaction_content_parts = []
            for inter in recent_interactions:
                interaction_content_parts.append(f"Turn {inter['turn']}:")
                interaction_content_parts.append(f"User: {inter['user_message']}")
                interaction_content_parts.append(f"Assistant: {inter['bot_message']}")
                interaction_content_parts.append("")
            
            interaction_content = "\n".join(interaction_content_parts)
            
            if background_summary:
                # Launch background task to create summary (NON-BLOCKING!)
                task = asyncio.create_task(
                    self.memory_manager._background_summarize_and_update(
                        turn_number=current_turn,
                        content=interaction_content,
                        period_type=PeriodType.DIRECT,
                        existing_memory=self.memory_manager.summary
                    )
                )
                self.memory_manager.background_tasks.append(task)
                print(Fore.LIGHTCYAN_EX + f"🔄 Summary for turns {start_turn}-{current_turn} running in background...", Fore.RESET)
                interaction_summary = f"[Summary pending for turns {start_turn}-{current_turn}]"
            else:
                # Blocking mode (original behavior)
                interaction_summary = await self.memory_manager.create_memory_summary(
                    content=interaction_content,
                    period_type=PeriodType.DIRECT,
                    existing_memory=self.memory_manager.summary
                )
                # Update interaction with summary
                self.memory_manager.update_interaction_summary(current_turn, interaction_summary)
        else:
            # Not a summary turn
            next_summary_turn = ((current_turn // self.memory_manager.summary_interval) + 1) * self.memory_manager.summary_interval
            interaction_summary = f"[No summary - next summary at turn {next_summary_turn}]"
        
        # Cleanup completed background tasks
        self.memory_manager.cleanup_background_tasks()
        
        # Check if we need to summarize (uses LangChain LLM internally)
        turns = self.check_turns()
        if turns > self.number_of_turns:
            await self.summarize_history()
        
        return {
            "turn": current_turn,
            "interaction": interaction,
            "summary": interaction_summary,
            "total_turns": turns,
            "overall_summary": self.summary,
            "background_tasks": len(self.memory_manager.background_tasks),
            "is_summary_turn": should_create_summary
        }
    
    def get_memory_context(self, query: str) -> str:
        """Get formatted memory context using text search."""
        # Simple keyword search in memory file
        keywords = query.lower().split()
        matches = []
        
        for interaction in self.memory_manager._all_interactions:
            # Check if any keywords appear in user or bot messages
            text = f"{interaction['user_message']} {interaction['bot_message']} {interaction['summary']}".lower()
            if any(keyword in text for keyword in keywords):
                matches.append(interaction)
        
        if not matches:
            return ""
        
        context_parts = ["**Relevant Past Conversations:**"]
        for i, interaction in enumerate(matches[:5], 1):  # Top 5
            turn = interaction['turn']
            summary = interaction.get('summary', interaction['user_message'][:80])
            context_parts.append(f"{i}. Turn {turn}: {summary}")
        
        return "\n".join(context_parts)
    
    def get_history_summary(self) -> str:
        """Get formatted conversation summary."""
        if self.summary:
            return f"**Conversation Summary:** {self.summary}"
        return ""
    
    def search_memory_text(self, pattern: str, case_sensitive: bool = False) -> List[str]:
        """
        Search memory file using regex pattern (like grep).
        
        Args:
            pattern: Regex pattern to search for
            case_sensitive: Whether search is case sensitive
            
        Returns:
            List of matching lines
        """
        return self.memory_manager.search_text_file(pattern, case_sensitive)
    
    async def wait_for_summaries(self, timeout: float = None):
        """
        Wait for all background summary tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
        """
        await self.memory_manager.wait_for_background_tasks(timeout)
    
    def get_pending_summaries_count(self) -> int:
        """
        Get the number of background summary tasks still running.
        
        Returns:
            Number of pending background tasks
        """
        self.memory_manager.cleanup_background_tasks()
        return len(self.memory_manager.background_tasks)


# Singleton instance cache
_memory_ops_cache: Dict[str, MemoryOps] = {}


def get_memory_ops(
    username: str,
    llm: ChatNVIDIA = None,
    memory_dir: str = None,
    use_streaming: bool = False,
    rate_limit_delay: float = 2.0,
    summary_interval: int = 10
) -> MemoryOps:
    """
    Get or create a text-based MemoryOps instance for a user.
    
    Args:
        username: User ID
        llm: Optional ChatNVIDIA instance
        memory_dir: Directory for memory files
        use_streaming: Whether to use streaming
        rate_limit_delay: Seconds to wait between LLM calls (default 2.0)
        summary_interval: Create summaries every N turns (default 10)
    """
    cache_key = f"{username}_{use_streaming}_{rate_limit_delay}_{summary_interval}"
    if cache_key not in _memory_ops_cache:
        _memory_ops_cache[cache_key] = MemoryOps(username, llm, memory_dir, use_streaming, rate_limit_delay, summary_interval)
    return _memory_ops_cache[cache_key]


def clear_user_memory(username: str) -> bool:
    """Clear all memories for a user."""
    try:
        # Remove from cache
        keys_to_remove = [k for k in _memory_ops_cache.keys() if k.startswith(username)]
        for key in keys_to_remove:
            del _memory_ops_cache[key]
        
        # Delete memory file
        try:
            docker_compose_path = Path("/workspace/docker-compose.yml")
            if docker_compose_path.exists():
                with open(docker_compose_path, "r") as f:
                    yaml_data = yaml.safe_load(f)
                    mnt_folder = yaml_data["services"]["agenticta"]["volumes"][-1].split(":")[-1]
                    memory_dir = Path(mnt_folder) / username / "memory"
            else:
                memory_dir = Path("mnt") / username / "memory"
        except:
            memory_dir = Path("mnt") / username / "memory"
        
        memory_file = memory_dir / "conversation_memory.txt"
        if memory_file.exists():
            memory_file.unlink()
            print(Fore.GREEN + f"✓ Cleared memory for user: {username}", Fore.RESET)
        
        return True
    except Exception as e:
        print(Fore.RED + f"Error clearing memory: {e}", Fore.RESET)
        return False

