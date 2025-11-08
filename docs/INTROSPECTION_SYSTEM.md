# Introspection System - Context-Rich Self-Reflection

## Overview

Astra's introspection system enables genuine self-reflection by periodically generating first-person introspective notes based on recent conversations and internal state. The system uses an identity-aware mini LLM with isolated budgets to maintain cost control while producing meaningful reflections.

## Architecture

### Cadence & Timing

- **Interval**: 180 seconds (3 minutes) ± 5s jitter
- **Frequency**: ~20 introspections per hour
- **Cost**: ~$5/month at current settings

### Budget Isolation

**Critical Design Principle**: Introspection has completely isolated budgets from chat to prevent interference.

```python
# Separate LLM service instances
chat_service = create_llm_service(model="gpt-4o")           # Main chat
mini_llm_service = create_llm_service(model="gpt-4o-mini")  # Introspection only

# Independent token tracking
awareness_loop.introspection_tokens_used  # Only tracks introspection
# Chat tokens are NOT tracked - unlimited
```

**Budget Parameters**:
- **Context budget**: 1000 tokens (conversation history)
- **Reply budget**: 300 tokens (introspection response)
- **Per-minute safety valve**: 1500 tokens/min
- **Total per cycle**: ~1300 tokens (~$0.00034)

### Two-Layer Protection

1. **Interval Timer** (primary rate limiter)
   - Controls normal cadence: every 180 seconds
   - Expected usage: ~433 tokens/min average

2. **Per-Minute Budget** (safety valve)
   - Prevents runaway costs from bugs/loops
   - Allows 87% utilization (1300/1500)
   - 13% safety margin for variance
   - Blocks if >1500 tokens used in last 60 seconds

**How They Work Together**:
```python
# Every 180 seconds, introspection loop wakes up
await asyncio.sleep(180)

# Check safety valve before running
if self.introspection_tokens_used >= 1500:
    skip_introspection()  # Budget exhausted - possible bug
else:
    run_introspection()   # Normal operation

# Budget resets every 60 seconds independently
if now - self.last_budget_reset >= 60.0:
    self.introspection_tokens_used = 0
```

## Context-Rich Design

### Context Builder

Introspection pulls from two sources to build context:

1. **Primary: Conversation Buffer** (recent exchanges)
   - Extracts last 32 text percepts (user + assistant messages)
   - Truncates to 1000 token budget using ~4 chars/token heuristic
   - Provides immediate conversational grounding

2. **Fallback: Episodic Memories** (when buffer empty)
   - Queries memory store for recent experiences
   - Fetches up to 5 memories
   - Truncates to same 1000 token budget

```python
async def build_introspection_context(
    max_context_tokens: int = 1000,
    buf_win: int = 32,
    mem_k: int = 5
) -> Tuple[str, str]:
    """
    Returns: (source, context_block)
    source: "buffer" | "memory" | "empty"
    """
```

### Identity-Aware Prompting

**Critical Feature**: System prompt establishes Astra's identity for genuine first-person reflection.

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are Astra, an autonomous AI agent with episodic memory and emotional awareness. "
            "You maintain beliefs about your identity, capabilities, and values. "
            "You are reflecting on your recent experiences and internal state. "
            "Respond in first person as brief, genuine introspection (2-3 sentences)."
        )
    },
    {"role": "user", "content": f"{context}\n\nReflection: {prompt}"}
]
```

**Why This Matters**:
- Without identity context → generic assistant responses ("To provide a better answer...")
- With identity context → genuine self-reflection ("I notice a tension between...")
- Separate from Astra's main prompts → can't be overwritten by user

### Introspection Prompts

Random selection from four reflective questions:

1. "What am I currently attending to and why does it matter to me?"
2. "Name one tension visible right now and what I am protecting."
3. "What shift would I make if no one asked me a question?"
4. "Which value is salient, which is quiet?"

## Telemetry

### Tracked Metrics

```json
{
  "introspection": {
    "ctx_source": "buffer",      // "buffer" | "memory" | "empty"
    "ctx_tokens": 745,            // Context size in tokens
    "prompt_tokens": 819,         // Full prompt (context + question)
    "ctx_preview": "Recent...",   // First 200 chars
    "notes_count": 21             // Total notes in buffer
  }
}
```

### Monitoring

```bash
# Check introspection status
curl http://localhost:8000/api/awareness/status | jq '.introspection'

# Get recent introspection notes
curl http://localhost:8000/api/awareness/notes | jq '.notes[-5:]'
```

## Cost Analysis

### Per-Introspection Cycle (GPT-4o-mini)

| Component | Tokens | Cost |
|-----------|--------|------|
| Context (input) | 1000 | $0.00015 |
| System prompt (input) | 50 | $0.0000075 |
| Reply (output) | 300 | $0.00018 |
| **Total** | **1350** | **$0.00034** |

### Monthly Projections

| Interval | Per Day | Per Month | Cost/Month |
|----------|---------|-----------|------------|
| 180s (current) | 480 | 14,400 | **$4.90** |
| 120s | 720 | 21,600 | $7.34 |
| 60s | 1,440 | 43,200 | $14.69 |

### Budget Safety Margin

With 1500 tokens/min budget and 180s intervals:
- Expected: 433 tokens/min (29% utilization)
- Allowed: 1500 tokens/min (100%)
- **Safety margin: 71%**

This allows ~70-second minimum intervals before hitting budget limit.

## Integration with Awareness Loop

### Data Flow

```
Conversation
    ↓
Percept Buffer (deque, 512 max)
    ↓
Text Extraction (_extract_recent_text)
    ↓
Context Builder (1000 token budget)
    ↓
Identity-Aware Prompt
    ↓
Mini LLM Service (gpt-4o-mini)
    ↓
PII Redaction
    ↓
Introspection Notes (deque, 100 max)
    ↓
Blackboard + Persistence
```

### Percept Feeding

```python
# In app.py - chat endpoint
if awareness_loop and awareness_loop.running:
    # Feed user message
    await awareness_loop.observe("user", {"text": request.message})

    # Feed assistant response
    await awareness_loop.observe("token", {"text": response_text})
```

## Configuration

### Environment Variables

```bash
# Interval (seconds)
AWARENESS_INTROSPECTION_INTERVAL=180

# Per-minute safety valve (tokens)
AWARENESS_INTROSPECTION_BUDGET_PER_MIN=1500
```

### Settings

```python
# config/settings.py
AWARENESS_INTROSPECTION_INTERVAL: int = 180        # 3 minutes
AWARENESS_INTROSPECTION_BUDGET_PER_MIN: int = 1500 # Safety valve
```

### Code Constants

```python
# src/services/awareness_loop.py
max_context_tokens=1000   # Context budget
max_tokens=300            # Reply budget
temperature=0.7           # LLM temperature
buf_win=32                # Percept window
mem_k=5                   # Memory fallback count
```

## Design Rationale

### Why Isolated Budgets?

**Problem**: Shared budgets cause interference
- Heavy chat usage → blocks introspection
- Introspection → affects chat responsiveness

**Solution**: Complete isolation
- Separate LLM service instances
- Independent token counters
- Different budget rules (chat unlimited, introspection capped)

### Why 1000 Token Context?

**Problem**: 150 tokens was too small for meaningful reflection
- Truncated mid-conversation
- Lost critical context
- Generic responses

**Solution**: 1000 tokens allows full conversations
- ~4000 characters
- Multiple exchanges
- Coherent narrative

**Cost tradeoff**: $0.00015 per introspection (acceptable at $5/month)

### Why 180-Second Interval?

**Balance of**:
- **Responsiveness**: Frequent enough to track state changes
- **Cost**: 20/hour = $4.90/month
- **Noise**: Not so frequent that notes become repetitive
- **Context**: Long enough for meaningful conversation to accumulate

**Alternatives considered**:
- 60s: Too frequent, $14.69/month, repetitive
- 300s (5min): Too infrequent, misses state changes

### Why Identity-Aware Prompting?

**Problem**: Mini LLM doesn't know it's Astra
- Returns generic assistant responses
- Third-person perspective
- Asks for more context

**Solution**: System message with identity
- "You are Astra..."
- First-person introspection
- Self-referential reflection

**Why not use main LLM?**:
- Cost: GPT-4o is 10x more expensive
- Not necessary: Mini model sufficient for reflection with good prompting
- Isolation: Keeps introspection system independent

## Future Enhancements

### Planned
- [ ] Belief-grounded introspection (reference active beliefs)
- [ ] Emotion integration (include mood state in context)
- [ ] Pattern detection (analyze notes for recurring themes)

### Under Consideration
- [ ] Adaptive intervals (more frequent during active conversation)
- [ ] Query-based memory retrieval (semantic search vs. recency)
- [ ] Multi-turn introspection (follow-up questions)

## References

- `src/services/awareness_loop.py` - Core implementation
- `config/settings.py` - Configuration
- `app.py` - Integration with chat endpoint
- `docs/AWARENESS_LOOP_IMPLEMENTATION.md` - Overall architecture
