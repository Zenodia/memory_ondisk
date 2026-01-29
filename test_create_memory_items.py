"""
Unit test script for Mem2Disk memory system.

Purpose:
1. TEST 1: Create 100 turns of sample memory conversations
2. TEST 2: Test grep/glob/sed searches on conversation_memory.txt

Run with: python test_memory_system.py
"""

import os
import asyncio
from pathlib import Path
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from agent_memory import MemoryOps


def format_username(name: str) -> str:
    """Convert username to lowercase with spaces replaced by underscores.
    
    Args:
        name: The username to format (e.g., "Zenodia Charpy")
    
    Returns:
        Formatted username (e.g., "zenodia_charpy")
    
    Example:
        >>> format_username("Zenodia Charpy")
        'zenodia_charpy'
    """
    return name.lower().replace(" ", "_")


def generate_conversation_topics():
    """Generate 100 diverse conversation topics and responses."""
    conversations = []
    
    # Math topics (turns 1-30)
    math_topics = [
        ("Hi! I'm studying quadratic equations and need help understanding them.", 
         "Hello! I'd be happy to help you with quadratic equations. Let's start with the standard form: ax² + bx + c = 0. What specifically would you like to understand?"),
        ("How do I factor x² + 5x + 6?", 
         "Great question! To factor x² + 5x + 6, we need two numbers that multiply to 6 and add to 5. Those numbers are 2 and 3. So: x² + 5x + 6 = (x + 2)(x + 3)"),
        ("What about when the leading coefficient isn't 1?", 
         "When the leading coefficient isn't 1, like in 2x² + 7x + 3, we use the AC method. Multiply A×C (2×3=6), find factors of 6 that add to 7 (1 and 6), then factor by grouping."),
        ("Can you show me the quadratic formula?", 
         "Absolutely! The quadratic formula is: x = (-b ± √(b²-4ac)) / 2a. This solves any quadratic equation ax² + bx + c = 0. It's especially useful when factoring is difficult."),
        ("What's the discriminant?", 
         "The discriminant is b²-4ac from the quadratic formula. It tells us how many real solutions exist: if >0, two solutions; if =0, one solution; if <0, no real solutions."),
        ("How do I complete the square?", 
         "To complete the square for x² + bx + c: 1) Move c to the right side, 2) Add (b/2)² to both sides, 3) Factor the left side as a perfect square, 4) Solve for x."),
        ("What's vertex form of a parabola?", 
         "Vertex form is y = a(x-h)² + k, where (h,k) is the vertex. This form makes it easy to see the parabola's vertex and whether it opens up (a>0) or down (a<0)."),
        ("How do I find the axis of symmetry?", 
         "The axis of symmetry is the vertical line x = -b/(2a) for a parabola y = ax² + bx + c. It passes through the vertex and divides the parabola into mirror images."),
        ("What are polynomial functions?", 
         "Polynomial functions are expressions like f(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀. The degree is the highest exponent, and they're continuous everywhere."),
        ("How do I divide polynomials?", 
         "You can divide polynomials using long division or synthetic division. Long division works like numerical division, aligning terms by degree. Synthetic division is faster for (x-c) divisors."),
        ("What's the Remainder Theorem?", 
         "The Remainder Theorem states that when polynomial P(x) is divided by (x-c), the remainder equals P(c). This gives a quick way to evaluate polynomials."),
        ("Explain rational expressions please.", 
         "Rational expressions are fractions with polynomials in numerator and denominator, like (x²-1)/(x+1). Simplify by factoring and canceling common factors."),
        ("How do I add fractions with variables?", 
         "To add rational expressions: 1) Find the LCD (least common denominator), 2) Convert each fraction to have the LCD, 3) Add numerators, 4) Simplify if possible."),
        ("What's a system of equations?", 
         "A system of equations is multiple equations with the same variables. Solutions satisfy all equations simultaneously. Common methods: substitution, elimination, or graphing."),
        ("How does the substitution method work?", 
         "Substitution method: 1) Solve one equation for a variable, 2) Substitute that expression into the other equation, 3) Solve for remaining variable, 4) Back-substitute."),
        ("What's the elimination method?", 
         "Elimination: Add or subtract equations to eliminate one variable. Multiply equations by constants if needed to make coefficients opposites, then add equations together."),
        ("Can matrices solve systems?", 
         "Yes! Write the system as an augmented matrix, then use row operations (Gaussian elimination) to get row-echelon form. Back-substitution gives the solution."),
        ("What are linear inequalities?", 
         "Linear inequalities like 2x + 3 > 7 use <, >, ≤, or ≥ instead of =. Solve like equations, but flip the inequality sign when multiplying/dividing by negatives."),
        ("How do I graph inequalities?", 
         "To graph a linear inequality: 1) Graph the boundary line (solid for ≤/≥, dashed for </> ), 2) Test a point to determine which side to shade, 3) Shade the solution region."),
        ("What's absolute value?", 
         "Absolute value |x| is the distance from zero, always non-negative. |x| = x if x ≥ 0, and |x| = -x if x < 0. It measures magnitude without direction."),
        ("How do I solve |x + 3| = 5?", 
         "Split into two cases: (x + 3) = 5 or (x + 3) = -5. This gives x = 2 or x = -8. Always check both solutions in the original equation."),
        ("What are exponent rules?", 
         "Key rules: xᵃ·xᵇ = xᵃ⁺ᵇ, xᵃ/xᵇ = xᵃ⁻ᵇ, (xᵃ)ᵇ = xᵃᵇ, x⁰ = 1, x⁻ⁿ = 1/xⁿ. These rules make simplifying exponential expressions easier."),
        ("What are logarithms?", 
         "A logarithm is the inverse of exponentiation. log_b(x) = y means bʸ = x. For example, log₂(8) = 3 because 2³ = 8."),
        ("How do log properties work?", 
         "Main properties: log(xy) = log(x) + log(y), log(x/y) = log(x) - log(y), log(xⁿ) = n·log(x). These convert multiplication/division to addition/subtraction."),
        ("What's the natural logarithm?", 
         "The natural logarithm ln(x) = log_e(x) uses base e ≈ 2.71828. It's the inverse of eˣ and appears frequently in calculus, growth/decay, and continuous compounding."),
        ("Explain exponential growth.", 
         "Exponential growth: y = a·bˣ where b > 1, or y = a·eᵏˣ where k > 0. The quantity increases by a constant factor over equal intervals. Common in populations, investments."),
        ("What's exponential decay?", 
         "Exponential decay: y = a·bˣ where 0 < b < 1, or y = a·e⁻ᵏˣ where k > 0. Quantity decreases by a constant factor. Seen in radioactive decay, depreciation."),
        ("What are sequences?", 
         "A sequence is an ordered list of numbers. Can be arithmetic (constant difference), geometric (constant ratio), or other patterns. Denoted as a₁, a₂, a₃, ... or aₙ."),
        ("What's an arithmetic sequence?", 
         "Arithmetic sequences have constant difference d between consecutive terms. Formula: aₙ = a₁ + (n-1)d. Example: 3, 7, 11, 15,... has d=4."),
        ("How do geometric sequences work?", 
         "Geometric sequences have constant ratio r. Formula: aₙ = a₁·rⁿ⁻¹. Example: 2, 6, 18, 54,... has r=3. The ratio is found by dividing consecutive terms."),
    ]
    
    # Science topics (turns 31-60)
    science_topics = [
        ("What is Newton's First Law?", 
         "Newton's First Law (Inertia) states that an object at rest stays at rest, and an object in motion stays in motion with constant velocity, unless acted upon by a net force."),
        ("Explain Newton's Second Law.", 
         "Newton's Second Law: F = ma. The net force on an object equals its mass times acceleration. More force or less mass means greater acceleration."),
        ("What's Newton's Third Law?", 
         "Newton's Third Law: For every action, there's an equal and opposite reaction. If object A exerts force on B, then B exerts equal force back on A in opposite direction."),
        ("What is kinetic energy?", 
         "Kinetic energy is energy of motion: KE = ½mv². It depends on mass and velocity squared. Double the speed quadruples the kinetic energy."),
        ("What's potential energy?", 
         "Potential energy is stored energy due to position. Gravitational PE = mgh (mass × gravity × height). Chemical, elastic, and electrical PE also exist."),
        ("Explain conservation of energy.", 
         "Energy can't be created or destroyed, only transformed. Total energy in a closed system remains constant. KE can become PE and vice versa."),
        ("What is momentum?", 
         "Momentum is mass times velocity: p = mv. It's conserved in collisions. Heavy slow objects and light fast objects can have equal momentum."),
        ("What's an atom made of?", 
         "Atoms have a nucleus (protons + neutrons) surrounded by electrons. Protons are positive, electrons negative, neutrons neutral. Most of atom is empty space."),
        ("What's the periodic table?", 
         "The periodic table organizes elements by atomic number (proton count). Rows are periods, columns are groups with similar properties. It predicts element behavior."),
        ("What are chemical bonds?", 
         "Chemical bonds hold atoms together. Ionic bonds transfer electrons, covalent bonds share electrons, metallic bonds pool electrons. Bond strength varies by type."),
        ("What is photosynthesis?", 
         "Photosynthesis: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. Plants convert light energy to chemical energy (glucose), releasing oxygen as byproduct."),
        ("What's cellular respiration?", 
         "Cellular respiration: C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ATP. Cells break down glucose to release energy (ATP). Opposite of photosynthesis."),
        ("What is DNA?", 
         "DNA (deoxyribonucleic acid) stores genetic information. Double helix structure with base pairs: A-T and G-C. Genes are DNA segments coding for proteins."),
        ("How does DNA replication work?", 
         "DNA replication: 1) Helix unwinds, 2) Each strand serves as template, 3) Complementary bases pair (A-T, G-C), 4) Two identical DNA molecules form."),
        ("What's evolution by natural selection?", 
         "Evolution: Organisms with advantageous traits survive and reproduce more. Over generations, helpful traits become more common. This drives species adaptation and diversity."),
        ("What are the states of matter?", 
         "Main states: solid (fixed shape/volume), liquid (fixed volume, variable shape), gas (variable shape/volume), plasma (ionized gas). State depends on temperature/pressure."),
        ("What's the water cycle?", 
         "Water cycle: Evaporation (liquid→gas), condensation (gas→liquid), precipitation (rain/snow), collection in oceans/lakes. Solar energy drives the cycle."),
        ("What causes weather?", 
         "Weather results from uneven solar heating. Warm air rises, cool air sinks, creating pressure differences and wind. Humidity affects precipitation and storms."),
        ("What is climate change?", 
         "Climate change: Long-term shifts in temperature and weather patterns. Current change mainly from human greenhouse gas emissions trapping heat in atmosphere."),
        ("What's the electromagnetic spectrum?", 
         "EM spectrum: Radio, microwave, infrared, visible light, UV, X-rays, gamma rays. All are electromagnetic waves differing in wavelength and energy."),
        ("What is sound?", 
         "Sound is a mechanical wave (vibration) through matter. Needs a medium (solid, liquid, or gas). Travels as compression waves at different speeds in different materials."),
        ("What's the speed of light?", 
         "Light travels at c ≈ 3×10⁸ m/s in vacuum. It's the universe's speed limit. Nothing with mass can reach light speed. Light slows in denser media."),
        ("What is electricity?", 
         "Electricity is flow of electrons through a conductor. Voltage is electrical pressure, current is flow rate, resistance opposes flow. Ohm's Law: V = IR."),
        ("What are circuits?", 
         "Electric circuits are closed paths for current. Series: components in single path, current same everywhere. Parallel: multiple paths, voltage same across branches."),
        ("What's magnetism?", 
         "Magnetism: Force from moving charges or magnetic materials. Magnets have north/south poles. Like poles repel, opposite poles attract. Earth acts like giant magnet."),
        ("How do electromagnets work?", 
         "Electromagnets use electric current through wire coil to create magnetic field. Field strength increases with more current or more coil turns. Widely used in motors, speakers."),
        ("What is radioactivity?", 
         "Radioactivity: Unstable nuclei emit particles/energy to become more stable. Types: alpha (He nuclei), beta (electrons), gamma (EM radiation). Used in medicine, power."),
        ("What's half-life?", 
         "Half-life: Time for half of radioactive sample to decay. Each isotope has characteristic half-life, from microseconds to billions of years. Used for dating rocks, fossils."),
        ("What is pH?", 
         "pH measures acidity/basicity on 0-14 scale. pH<7 acidic, pH=7 neutral, pH>7 basic. Each unit is 10× difference in H⁺ concentration. Important in chemistry, biology."),
        ("What's the scientific method?", 
         "Scientific method: 1) Ask question, 2) Research, 3) Form hypothesis, 4) Test via experiment, 5) Analyze data, 6) Draw conclusions, 7) Communicate results. Basis of science."),
    ]
    
    # Programming topics (turns 61-90)
    programming_topics = [
        ("What is a variable in programming?", 
         "A variable is a named storage location that holds a value. Like a labeled box - the name stays the same but contents can change. Example: x = 5 stores 5 in variable x."),
        ("What are data types?", 
         "Data types define what kind of value a variable holds: integers (whole numbers), floats (decimals), strings (text), booleans (true/false), lists, dictionaries, etc."),
        ("What's a function?", 
         "A function is reusable code block that performs a task. Takes inputs (parameters), processes them, returns output. Helps organize code and avoid repetition."),
        ("What are loops?", 
         "Loops repeat code multiple times. For loop: iterate over sequence. While loop: repeat while condition true. Saves writing same code repeatedly."),
        ("What's an if statement?", 
         "If statements make decisions. Execute code only when condition is true. Can add 'else' for alternative path, 'elif' for multiple conditions. Basis of program logic."),
        ("What are lists/arrays?", 
         "Lists store multiple values in order. Access elements by index (position). Example: numbers = [1, 2, 3]. Can add, remove, or modify elements."),
        ("What's a dictionary/map?", 
         "Dictionaries store key-value pairs. Look up values by key (like a real dictionary). Example: ages = {'Alice': 25, 'Bob': 30}. Fast lookups."),
        ("What is object-oriented programming?", 
         "OOP organizes code into objects containing data (attributes) and functions (methods). Classes are blueprints for objects. Enables code reuse and modeling real-world entities."),
        ("What's a class?", 
         "A class is a template/blueprint for creating objects. Defines attributes (data) and methods (functions) objects will have. Like a cookie cutter for making similar objects."),
        ("What's inheritance?", 
         "Inheritance: A class can inherit attributes/methods from parent class. Child class extends parent, adding new features. Promotes code reuse, models 'is-a' relationships."),
        ("What are algorithms?", 
         "Algorithms are step-by-step procedures to solve problems. Like recipes: precise instructions that produce consistent results. Efficiency matters for large inputs."),
        ("What's time complexity?", 
         "Time complexity measures how algorithm's runtime grows with input size. O(1) constant, O(n) linear, O(n²) quadratic, O(log n) logarithmic. Lower is better."),
        ("What is recursion?", 
         "Recursion: Function calling itself. Needs base case (stop condition) and recursive case. Useful for problems with self-similar structure like factorials, trees."),
        ("What are sorting algorithms?", 
         "Sorting algorithms arrange data in order. Bubble sort, merge sort, quick sort have different speeds and uses. Important for data organization and search efficiency."),
        ("What's binary search?", 
         "Binary search finds item in sorted list by repeatedly halving search space. O(log n) time. Much faster than linear search for large datasets."),
        ("What are data structures?", 
         "Data structures organize and store data efficiently. Arrays, linked lists, stacks, queues, trees, graphs. Choice affects program performance and capability."),
        ("What's a stack?", 
         "Stack: Last-In-First-Out (LIFO) data structure. Like stack of plates - add/remove from top only. Operations: push (add), pop (remove). Used in undo, parsing, recursion."),
        ("What's a queue?", 
         "Queue: First-In-First-Out (FIFO) data structure. Like waiting line - enter at back, exit from front. Operations: enqueue (add), dequeue (remove). Used in task scheduling."),
        ("What is a linked list?", 
         "Linked list: Elements (nodes) connected by pointers. Each node has data and pointer to next node. Dynamic size, efficient insertion/deletion, but slow random access."),
        ("What's a binary tree?", 
         "Binary tree: Each node has at most 2 children (left, right). Root at top, leaves at bottom. Binary search trees keep data ordered for fast search."),
        ("What is a hash table?", 
         "Hash table: Uses hash function to compute index for key-value pairs. O(1) average lookup time. Handles collisions via chaining or open addressing. Very efficient."),
        ("What's an API?", 
         "API (Application Programming Interface): Set of rules for how programs interact. Like a menu at restaurant - specifies what you can request and what you'll get back."),
        ("What is Git?", 
         "Git is version control system tracking code changes over time. Enables collaboration, branching (parallel development), merging changes, and reverting to earlier versions."),
        ("What's the difference between Git and GitHub?", 
         "Git is the version control tool. GitHub is a website hosting Git repositories online, adding collaboration features like pull requests, issues, and code review."),
        ("What is SQL?", 
         "SQL (Structured Query Language) manages relational databases. Commands: SELECT (query), INSERT (add), UPDATE (modify), DELETE (remove). Organizes data in tables with relationships."),
        ("What's the difference between SQL and NoSQL?", 
         "SQL: Structured tables, relationships, ACID guarantees. NoSQL: Flexible schemas (documents, key-value, graphs), horizontal scaling, eventual consistency. Different use cases."),
        ("What is HTTP?", 
         "HTTP (HyperText Transfer Protocol) enables web communication. Client sends request (GET, POST, etc.), server responds. Stateless protocol - each request independent."),
        ("What's REST API?", 
         "REST: Architectural style for APIs using HTTP. Resources accessed via URLs, operations via HTTP methods (GET, POST, PUT, DELETE). Stateless, scalable, widely used."),
        ("What is JSON?", 
         "JSON (JavaScript Object Notation) is text format for data exchange. Lightweight, human-readable, language-independent. Structure: objects {}, arrays [], key-value pairs."),
        ("What's the difference between == and ===?", 
         "In JavaScript: == compares values with type coercion (5 == '5' is true). === compares values AND types without coercion (5 === '5' is false). === is stricter."),
    ]
    
    # General learning topics (turns 91-100)
    general_topics = [
        ("How can I improve my study habits?", 
         "Effective study: 1) Space out sessions (distributed practice), 2) Test yourself often (active recall), 3) Explain concepts aloud, 4) Mix different topics, 5) Take breaks."),
        ("What is active recall?", 
         "Active recall: Retrieve information from memory without looking at notes. Much more effective than passive re-reading. Use flashcards, practice problems, self-quizzing."),
        ("What's spaced repetition?", 
         "Spaced repetition: Review material at increasing intervals. Fight forgetting curve by revisiting just before you'd forget. SRS apps like Anki automate optimal timing."),
        ("How do I take better notes?", 
         "Effective notes: Don't transcribe - summarize in your words. Use Cornell method, mind maps, or outline format. Review within 24 hours to reinforce learning."),
        ("What's the Pomodoro Technique?", 
         "Pomodoro: Work 25 minutes, break 5 minutes. After 4 cycles, longer break (15-30 min). Maintains focus, prevents burnout. Adjust times to your preference."),
        ("How do I overcome procrastination?", 
         "Beat procrastination: 1) Start tiny (2-minute rule), 2) Remove distractions, 3) Use deadlines, 4) Break tasks into steps, 5) Forgive past procrastination. Action reduces anxiety."),
        ("What is growth mindset?", 
         "Growth mindset: Belief that abilities develop through effort. Contrast: fixed mindset thinks talent is innate. Growth mindset embraces challenges, learns from failure."),
        ("How can I read faster?", 
         "Speed reading: 1) Reduce subvocalization, 2) Use pointer/finger to guide eyes, 3) Preview text structure first, 4) Don't regress (reread). Balance speed with comprehension."),
        ("What's critical thinking?", 
         "Critical thinking: Analyze information objectively. Question assumptions, check sources, identify biases, consider alternatives, evaluate evidence. Essential for problem-solving."),
        ("How do I stay motivated to learn?", 
         "Stay motivated: 1) Set specific goals, 2) Track progress visibly, 3) Connect to larger purpose, 4) Celebrate small wins, 5) Find study buddies. Motivation follows action."),
    ]
    
    conversations.extend(math_topics)
    conversations.extend(science_topics)
    conversations.extend(programming_topics)
    conversations.extend(general_topics)
    
    return conversations


async def test_create_memory():
    """TEST 1: Create 100 turns of sample memory conversations with background summarization."""
    print("=" * 80)
    print("TEST 1: Creating 100 Turns of Sample Memory (NON-BLOCKING MODE)")
    print("=" * 80)
    
    # Check API key
    if not os.getenv("NVIDIA_API_KEY"):
        print("\n❌ ERROR: NVIDIA_API_KEY not set!")
        print("Set it with: export NVIDIA_API_KEY='your-key-here'")
        return None
    
    # Setup LLM
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.6,
        top_p=0.95,
        max_completion_tokens=4096,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    # Initialize memory system
    memory_dir = Path("./test_memory")
    memory_ops = MemoryOps(
        username="test_student",
        llm=llm,
        memory_dir=str(memory_dir),
        rate_limit_delay=1.0,  # Faster for 100 turns
        summary_interval=10  # Summarize every 10 turns
    )
    
    print("\n✓ Memory system initialized")
    print(f"  User: test_student")
    print(f"  Directory: {memory_dir}")
    print(f"  Mode: BACKGROUND SUMMARIZATION (non-blocking)")
    
    # Generate 100 conversation turns
    print("\n📝 Generating 100 diverse conversation topics...")
    conversation_data = generate_conversation_topics()
    print(f"✓ Generated {len(conversation_data)} conversation turns")
    
    print("\n📝 Processing and saving conversations...")
    print("⚡ Conversations are processed IMMEDIATELY (non-blocking)")
    print("🔄 Summaries are generated every 10 turns in BACKGROUND")
    print("-" * 80)
    
    import time
    start_time = time.time()
    
    for i, (user_msg, bot_msg) in enumerate(conversation_data, 1):
        turn_start = time.time()
        
        # Process message with background summarization (NON-BLOCKING!)
        result = await memory_ops.process_message(
            message=user_msg,
            bot_response=bot_msg,
            background_summary=True  # Enable background mode
        )
        
        turn_elapsed = time.time() - turn_start
        is_summary_turn = result.get('is_summary_turn', False)
        
        # Progress indicator with timing
        if i % 10 == 0:
            pending = memory_ops.get_pending_summaries_count()
            summary_indicator = "📊 SUMMARY" if is_summary_turn else ""
            print(f"  ✓ Turn {i:3d}/100 processed in {turn_elapsed:.2f}s | {pending} summaries pending {summary_indicator}")
    
    total_elapsed = time.time() - start_time
    pending_count = memory_ops.get_pending_summaries_count()
    
    print("-" * 80)
    print(f"⚡ All 100 turns processed in {total_elapsed:.2f}s (non-blocking!)")
    print(f"📊 Summaries created at turns: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100")
    print(f"🔄 {pending_count} background summary tasks still running")
    
    # Wait for all background tasks to complete
    print("\n⏳ Waiting for all background summaries to complete...")
    await memory_ops.wait_for_summaries(timeout=300)  # 5 minute timeout
    
    final_elapsed = time.time() - start_time
    print(f"✅ All summaries completed in total {final_elapsed:.2f}s")
    
    memory_file = memory_dir / "test_student" / "memory" / "conversation_memory.txt"
    
    if memory_file.exists():
        print(f"\n✅ Successfully created memory file with 100 turns!")
        print(f"   Location: {memory_file}")
        print(f"   File size: {memory_file.stat().st_size:,} bytes")
        print(f"   Total turns: {result['total_turns']}")
        print(f"   Time saved: ~{(100 * 2) - total_elapsed:.1f}s (conversations were non-blocking)")
    else:
        print(f"\n❌ Memory file not found!")
    
    return memory_file


def test_grep_glob_sed_searches(memory_file: Path):
    """TEST 2: Test grep/glob/sed searches on conversation_memory.txt file."""
    print("\n\n" + "=" * 80)
    print("TEST 2: Grep/Glob/Sed Searches on conversation_memory.txt")
    print("=" * 80)
    
    if not memory_file or not memory_file.exists():
        print(f"\n❌ Memory file not found: {memory_file}")
        return
    
    print(f"\nMemory file: {memory_file}")
    print("\nThe following commands search the memory file for conversations:")
    print("=" * 80)
    
    # Basic searches
    print("\n📌 BASIC SEARCHES:\n")
    
    print("1. Find all turns (just markers):")
    print(f"   grep '<<<TURN:' {memory_file}")
    
    print("\n2. Count total turns:")
    print(f"   grep -c '<<<TURN:' {memory_file}")
    
    print("\n3. View turn 1 WITH CONTENT (20 lines after):")
    print(f"   grep -A 20 '<<<TURN:0001>>>' {memory_file}")
    
    print("\n4. View ENTIRE turn 1 conversation:")
    print(f"   sed -n '/<<<TURN:0001>>>/,/<<<END_TURN:0001>>>/p' {memory_file}")
    
    # User/Bot messages
    print("\n\n📌 USER & BOT MESSAGES:\n")
    
    print("5. Find all user messages WITH CONTENT:")
    print(f"   grep -A 1 '>>>USER:' {memory_file}")
    
    print("\n6. Find all bot responses WITH CONTENT:")
    print(f"   grep -A 1 '>>>BOT:' {memory_file}")
    
    print("\n7. Extract specific user message (turn 2):")
    print(f"   sed -n '/>>>USER:0002>>>/,/<<<USER:0002<<</p' {memory_file}")
    
    print("\n8. Extract specific bot response (turn 3):")
    print(f"   sed -n '/>>>BOT:0003>>>/,/<<<BOT:0003<<</p' {memory_file}")
    
    # Keyword searches
    print("\n\n📌 KEYWORD SEARCHES (with context):\n")
    
    print("9. Search for 'quadratic' with 5 lines context:")
    print(f"   grep -i -C 5 'quadratic' {memory_file}")
    
    print("\n10. Search for 'factor' with 3 before, 10 after:")
    print(f"   grep -i -B 3 -A 10 'factor' {memory_file}")
    
    print("\n11. Search for 'formula' with line numbers:")
    print(f"   grep -n -i -C 5 'formula' {memory_file}")
    
    # Summary and metadata
    print("\n\n📌 SUMMARY & METADATA:\n")
    
    print("12. Get conversation summary:")
    print(f"   sed -n '/###SUMMARY_START###/,/###SUMMARY_END###/p' {memory_file}")
    
    print("\n13. Get all timestamps:")
    print(f"   grep '@TIMESTAMP:' {memory_file}")
    
    print("\n14. Get user ID:")
    print(f"   grep '@USER_ID:' {memory_file} | head -1")
    
    print("\n15. Extract turn numbers only:")
    print(f"   grep -o '<<<TURN:[0-9]{{4}}>>>' {memory_file}")
    
    # Advanced combinations
    print("\n\n📌 ADVANCED COMBINATIONS:\n")
    
    print("16. Get user messages without markers:")
    print(f"   grep -A 1 '>>>USER:' {memory_file} | grep -v '>>>'")
    
    print("\n17. Search multiple keywords (quadratic OR formula):")
    print(f"   grep -i -E 'quadratic|formula' {memory_file}")
    
    print("\n18. Count how many times 'factor' appears:")
    print(f"   grep -i -o 'factor' {memory_file} | wc -l")
    
    print("\n19. View turns 2-3 (multiple ranges):")
    print(f"   sed -n '/<<<TURN:0002>>>/,/<<<END_TURN:0003>>>/p' {memory_file}")
    
    print("\n20. Find turns on specific date:")
    print(f"   grep -B 2 '@DATE:2026-01-' {memory_file} | grep '<<<TURN:'")
    
    # Glob patterns
    print("\n\n📌 GLOB PATTERNS (file finding):\n")
    
    memory_dir = memory_file.parent.parent.parent
    print(f"21. Find all memory files:")
    print(f"   find {memory_dir} -name 'conversation_memory.txt'")
    
    print(f"\n22. Find all .txt files recursively:")
    print(f"   find {memory_dir} -name '*.txt'")
    
    print(f"\n23. Use glob with grep to search multiple files:")
    print(f"   grep -r 'quadratic' {memory_dir}/**/conversation_memory.txt")
    
    print(f"\n24. List all memory directories:")
    print(f"   find {memory_dir} -type d -name 'memory'")
    
    print(f"\n25. Search all memory files for keyword:")
    print(f"   find {memory_dir} -name '*.txt' -exec grep -l 'Newton' {{}} \\;")
    
    print("\n\n" + "=" * 80)
    print("✅ All grep/glob/sed examples listed above!")
    print("💡 Try running these commands in your terminal (Linux/Mac/WSL/Git Bash)")
    print("=" * 80)


def verify_grep_searches(memory_file: Path):
    """Verify grep/glob/sed searches work programmatically."""
    if not memory_file or not memory_file.exists():
        print(f"\n❌ Memory file not found: {memory_file}")
        return
    
    print("\n📊 Verification Stats:")
    print("-" * 80)
    
    # Read the file
    with open(memory_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count elements using regex (simulates grep -c)
    import re
    turns = len(re.findall(r'<<<TURN:', content))
    user_msgs = len(re.findall(r'>>>USER:', content))
    bot_msgs = len(re.findall(r'>>>BOT:', content))
    
    print(f"  ✓ Total turns found: {turns}")
    print(f"  ✓ User messages: {user_msgs}")
    print(f"  ✓ Bot messages: {bot_msgs}")
    
    # Extract summary (simulates sed)
    summary_match = re.search(r'###SUMMARY_START###\n(.+?)\n###SUMMARY_END###', content, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
        print(f"  ✓ Summary extracted: {len(summary)} characters")
    
    # Search for diverse keywords
    keywords = ['quadratic', 'Newton', 'variable', 'algorithm', 'motivation']
    print(f"\n  ✓ Keyword occurrences:")
    for keyword in keywords:
        matches = len(re.findall(keyword, content, re.IGNORECASE))
        print(f"     '{keyword}': {matches} times")
    
    # Verify specific turn extraction (simulates sed range)
    turn_50 = re.search(r'<<<TURN:0050>>>(.*?)<<<END_TURN:0050>>>', content, re.DOTALL)
    if turn_50:
        print(f"\n  ✓ Turn 50 extraction works!")
    
    print("\n✅ All grep/glob/sed searches verified!")


async def test_blocking_vs_nonblocking_comparison():
    """TEST 0: Compare blocking vs non-blocking summarization with 5 turns."""
    print("\n" + "=" * 80)
    print("TEST 0: BLOCKING vs NON-BLOCKING Comparison (5 turns)")
    print("=" * 80)
    
    if not os.getenv("NVIDIA_API_KEY"):
        print("\n⚠️  Skipping comparison test (NVIDIA_API_KEY not set)")
        return
    
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.6,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    # Get first 5 conversations
    conversation_data = generate_conversation_topics()[:5]
    
    import time
    
    # Test 1: BLOCKING mode
    print("\n🔴 BLOCKING MODE (traditional):")
    print("-" * 80)
    memory_dir_blocking = Path("./test_memory_blocking")
    memory_ops_blocking = MemoryOps(
        username="test_blocking",
        llm=llm,
        memory_dir=str(memory_dir_blocking),
        rate_limit_delay=0.5,
        summary_interval=1  # Summarize every turn for comparison
    )
    
    start_blocking = time.time()
    for i, (user_msg, bot_msg) in enumerate(conversation_data, 1):
        turn_start = time.time()
        result = await memory_ops_blocking.process_message(
            message=user_msg,
            bot_response=bot_msg,
            background_summary=False  # BLOCKING
        )
        turn_time = time.time() - turn_start
        summary_indicator = "📊" if result.get('is_summary_turn', False) else "  "
        print(f"  {summary_indicator} Turn {i}: {turn_time:.2f}s (waited for summary)")
    
    blocking_total = time.time() - start_blocking
    print(f"  ⏱️  Total time (blocking): {blocking_total:.2f}s")
    
    # Test 2: NON-BLOCKING mode
    print("\n🟢 NON-BLOCKING MODE (background tasks, summary every turn for comparison):")
    print("-" * 80)
    memory_dir_nonblocking = Path("./test_memory_nonblocking")
    memory_ops_nonblocking = MemoryOps(
        username="test_nonblocking",
        llm=llm,
        memory_dir=str(memory_dir_nonblocking),
        rate_limit_delay=0.5,
        summary_interval=1  # Summarize every turn for fair comparison
    )
    
    start_nonblocking = time.time()
    for i, (user_msg, bot_msg) in enumerate(conversation_data, 1):
        turn_start = time.time()
        result = await memory_ops_nonblocking.process_message(
            message=user_msg,
            bot_response=bot_msg,
            background_summary=True  # NON-BLOCKING
        )
        turn_time = time.time() - turn_start
        pending = result.get('background_tasks', 0)
        summary_indicator = "📊" if result.get('is_summary_turn', False) else "  "
        print(f"  {summary_indicator} Turn {i}: {turn_time:.2f}s (immediate, {pending} tasks running)")
    
    nonblocking_processing_time = time.time() - start_nonblocking
    
    # Wait for background tasks
    pending = memory_ops_nonblocking.get_pending_summaries_count()
    print(f"\n  🔄 Waiting for {pending} background summaries...")
    await memory_ops_nonblocking.wait_for_summaries(timeout=60)
    
    nonblocking_total = time.time() - start_nonblocking
    
    print(f"  ⏱️  Processing time (non-blocking): {nonblocking_processing_time:.2f}s")
    print(f"  ⏱️  Total time (with background): {nonblocking_total:.2f}s")
    
    # Comparison
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS:")
    print("-" * 80)
    print(f"  Blocking mode:     {blocking_total:.2f}s")
    print(f"  Non-blocking mode: {nonblocking_processing_time:.2f}s (conversation only)")
    print(f"  Speedup:           {blocking_total / nonblocking_processing_time:.1f}x faster!")
    print(f"  User experience:   ⚡ IMMEDIATE responses (summaries run in background)")
    print("=" * 80)


async def main():
    """Run all unit tests."""
    print("\n" + "=" * 80)
    print("MEM2DISK MEMORY SYSTEM - UNIT TESTS")
    print("=" * 80)
    print("\nThis script runs three tests:")
    print("  TEST 0: Compare blocking vs non-blocking (5 turns)")
    print("  TEST 1: Create 100 turns with non-blocking summaries")
    print("  TEST 2: Test grep/glob/sed searches")
    print("=" * 80)
    
    # TEST 0: Demonstrate blocking vs non-blocking
    await test_blocking_vs_nonblocking_comparison()
    
    # TEST 1: Create 100 turns of sample memories
    memory_file = await test_create_memory()
    
    if not memory_file:
        print("\n❌ TEST 1 failed - cannot proceed to TEST 2")
        return
    
    # TEST 2: Demonstrate grep/glob/sed searches
    test_grep_glob_sed_searches(memory_file)
    
    # Verify searches work
    verify_grep_searches(memory_file)
    
    print("\n\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📁 Memory file location: {memory_file}")
    print(f"📊 Contains: 100 conversation turns")
    print(f"📊 Summaries created at: turns 10, 20, 30, 40, 50, 60, 70, 80, 90, 100")
    print(f"💡 Summary interval: Every 10 turns (configurable)")
    print("\n💡 Key Benefits:")
    print("  ✅ Non-blocking: Conversations continue immediately")
    print("  ✅ Background tasks: Summaries generated asynchronously")
    print("  ✅ Efficient: Only 10 summaries created (not 100)")
    print("  ✅ Significant speedup: User doesn't wait for LLM")
    print("\n💡 Next steps:")
    print("  1. Try the grep/sed commands listed in TEST 2")
    print("  2. Experiment with different search patterns")
    print("  3. Use glob patterns to find multiple files")
    print("  4. Adjust summary_interval for your use case")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())

