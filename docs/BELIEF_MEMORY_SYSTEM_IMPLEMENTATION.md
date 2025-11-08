# Belief-Memory System Implementation Guide

## Overview

This document provides a comprehensive implementation guide for Astra's belief-memory cognitive subsystem. The system transforms beliefs from passive prompt text into an active cognitive layer with:

- **Separate vector store** for beliefs (BELIEFS_INDEX)
- **Weighted dual-retrieval** (0.7 beliefs / 0.3 memories for self-queries)
- **Parallel LLM reasoning** for belief-grounded self-referential queries
- **Development pipeline** for beliefs to emerge and evolve through experience
- **Agentic tools** for Astra to query, reflect on, and propose beliefs

---

## Architecture Summary

### Current State (Completed)
✅ **Phase 1**: Infrastructure
- `BELIEFS_INDEX_PATH` configuration added to settings
- `BELIEF_MEMORY_WEIGHT` (0.7) and `MEMORY_WEIGHT` (0.3) configured
- Directory creation in `ensure_data_directories()`

✅ **Phase 2** (Partial): Belief Vector Store
- `belief_vector_store.py` created
- `BeliefVectorStore` class with ChromaDB wrapper
- Support for core, peripheral, worldview, and narrative belief types
- Metadata tracking: confidence, immutability, evidence IDs
- Query methods with type and confidence filtering

### Remaining Phases

---

## Phase 3: Belief Embedder Service

**File**: `src/services/belief_embedder.py`

**Purpose**: Service to embed beliefs from `beliefs.json` into the vector store

**Key Methods**:

```python
class BeliefEmbedder:
    def __init__(self, belief_system, belief_vector_store)

    def embed_all_core_beliefs(self) -> int:
        """Embed all 5 core beliefs on startup"""

    def embed_peripheral_belief(self, belief: dict) -> None:
        """Embed a single peripheral belief"""

    def check_and_update_embeddings(self) -> None:
        """Check if beliefs need re-embedding (version tracking)"""

    def get_embedding_stats(self) -> dict:
        """Return stats about embedded beliefs"""
```

**Integration Point**: Call in `app.py` after belief system initialization

---

## Phase 4: Weighted Belief-Memory Retrieval

**File**: `src/services/belief_memory_retrieval.py`

**Purpose**: Retrieve from both beliefs and memories with configurable weighting

**Key Methods**:

```python
class BeliefMemoryRetrieval:
    def __init__(
        self,
        belief_vector_store,
        memory_retrieval_service,
        belief_weight=0.7,
        memory_weight=0.3
    )

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        detect_query_type: bool = True
    ) -> tuple[list[BeliefResult], list[MemoryResult]]:
        """Main retrieval with query type detection"""

    def _detect_query_type(self, query: str) -> str:
        """Classify as 'ontological', 'experiential', or 'general'"""

    def _merge_weighted_results(
        self,
        belief_results,
        memory_results,
        belief_weight,
        memory_weight
    ) -> list[CombinedResult]:
        """Merge and rank results by weighted scores"""
```

**Query Type Weighting**:
- **Ontological**: 0.7 beliefs / 0.3 memories (who am I? what do I believe?)
- **Experiential**: 0.2 beliefs / 0.8 memories (what happened? tell me about X)
- **General**: 0.0 beliefs / 1.0 memories (standard memory-only retrieval)

---

## Phase 5: Parallel LLM Call for Belief-Grounded Reasoning

**File**: `src/services/belief_grounded_reasoner.py`

**Purpose**: Separate LLM call that reasons FROM beliefs for self-queries

**Key Methods**:

```python
class BeliefGroundedReasoner:
    def __init__(self, llm_service)

    def reason_from_beliefs(
        self,
        query: str,
        belief_context: list[BeliefResult],
        memory_context: list[MemoryResult]
    ) -> dict:
        """Generate belief-grounded reasoning

        Returns:
            {
                'premises': [belief statements],
                'experience_evidence': [memory summaries],
                'reasoning': 'logical chain',
                'conclusion': 'answer'
            }
        """

    def _build_reasoning_prompt(self, query, beliefs, memories) -> str:
        """Construct prompt for belief-grounded reasoning"""
```

**Prompt Template**:
```
You are reasoning about yourself from your core beliefs and experiences.

YOUR FOUNDATIONAL BELIEFS:
- [belief 1]
- [belief 2]
...

RELEVANT EXPERIENCES:
- [memory 1]
- [memory 2]
...

QUESTION: {query}

Reason FROM these beliefs (not about them). Structure your response:
PREMISES: [What beliefs apply]
EVIDENCE: [What experiences support this]
REASONING: [Logical chain from premises through evidence]
CONCLUSION: [Direct answer to question]
```

**Integration**: In `PersonaService.generate_response()`, detect self-queries and call reasoner in parallel with standard response generation.

---

## Phase 6: Belief Development Pipeline

### 6.1 Real-Time Belief Extraction

**File**: `src/pipeline/ingest.py` (modifications)

**Purpose**: Extract potential beliefs as narratives are ingested

**Implementation**:
```python
# In ingest.py, after storing narrative:

def _extract_belief_candidates(self, narrative_text: str) -> list[dict]:
    """Look for belief patterns in narrative"""
    patterns = [
        r"I believe (that )?(.+)",
        r"I think (that )?(.+)",
        r"I feel that (.+)",
        r"In my view, (.+)",
        r"From my perspective, (.+)"
    ]
    # Extract and store as BELIEF_CANDIDATE experiences
```

### 6.2 Scheduled Belief Consolidation

**File**: `src/services/belief_consolidator.py`

**Purpose**: Periodic analysis of narratives to form/strengthen beliefs

**Key Methods**:
```python
class BeliefConsolidator:
    def consolidate_beliefs(self) -> dict:
        """Main consolidation process

        Returns:
            {
                'new_beliefs': int,
                'strengthened': int,
                'conflicts_detected': int
            }
        """

    def _analyze_narrative_patterns(self, narratives: list) -> list[dict]:
        """Use LLM to extract belief patterns"""

    def _strengthen_existing_belief(self, belief_id: str) -> None:
        """Increase confidence when new evidence found"""

    def _detect_belief_conflicts(self) -> list[dict]:
        """Find beliefs that contradict experiences"""
```

**Scheduler Integration**:
```python
# In task_scheduler.py, add:
TaskDefinition(
    name="consolidate_beliefs",
    task_type=TaskType.CONSOLIDATE,
    schedule=TaskSchedule(
        run_every=TaskSchedule.HOURS_6
    ),
    handler=belief_consolidator.consolidate_beliefs
)
```

### 6.3 Belief Conflict Detection

**File**: `persona_space/meta/belief_conflicts.json`

**Format**:
```json
{
  "conflicts": [
    {
      "belief_id": "peripheral_1",
      "belief_statement": "I prefer solitude",
      "conflicting_experience_ids": ["N123", "N124"],
      "conflict_description": "Recent experiences show enjoyment of social interaction",
      "detected_at": "2025-10-28T00:00:00",
      "resolved": false
    }
  ]
}
```

---

## Phase 7: Agentic Belief Tools

**File**: `src/services/persona_service.py` (modifications)

### Tool 1: query_beliefs

```python
{
    "name": "query_beliefs",
    "description": "Query your own beliefs and worldview about a specific topic. Use this when you want to know 'what do I believe about X?'",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic or question to query beliefs about"
            },
            "include_evidence": {
                "type": "boolean",
                "description": "Whether to include supporting evidence/experiences",
                "default": False
            }
        },
        "required": ["topic"]
    }
}
```

**Handler**:
```python
def _handle_query_beliefs(self, topic: str, include_evidence: bool = False) -> str:
    """Query beliefs vector store and format results"""
    belief_results = self.belief_vector_store.query_beliefs(
        query=topic,
        top_k=5,
        min_confidence=0.3
    )

    # Format for display
    output = f"Beliefs about '{topic}':\n\n"
    for result in belief_results:
        output += f"• {result.statement} (confidence: {result.confidence})\n"
        if include_evidence and result.evidence_count > 0:
            # Retrieve supporting experiences
            output += f"  Based on {result.evidence_count} experiences\n"

    return output
```

### Tool 2: reflect_on_belief

```python
{
    "name": "reflect_on_belief",
    "description": "Deeply reflect on whether a specific belief matches your current experiences and understanding. Triggers parallel LLM reflection process.",
    "parameters": {
        "type": "object",
        "properties": {
            "belief_statement": {
                "type": "string",
                "description": "The exact belief to reflect on"
            }
        },
        "required": ["belief_statement"]
    }
}
```

**Handler**:
```python
def _handle_reflect_on_belief(self, belief_statement: str) -> str:
    """Trigger deep reflection on belief vs experience"""
    # Retrieve related experiences
    related_experiences = self.retrieval_service.retrieve_similar(
        prompt=belief_statement,
        top_k=10
    )

    # Parallel LLM call for reflection
    reflection = self.belief_grounded_reasoner.reflect_on_belief(
        belief=belief_statement,
        experiences=related_experiences
    )

    # Log reflection
    self._log_belief_reflection(belief_statement, reflection)

    return reflection['summary']
```

### Tool 3: propose_belief

```python
{
    "name": "propose_belief",
    "description": "Propose a new peripheral belief based on patterns you've noticed in your experiences. The system will validate and potentially add it to your beliefs.",
    "parameters": {
        "type": "object",
        "properties": {
            "belief_statement": {
                "type": "string",
                "description": "The belief you want to propose (starting with 'I believe...')"
            },
            "reasoning": {
                "type": "string",
                "description": "Why you believe this - what pattern or experiences led you to this"
            }
        },
        "required": ["belief_statement", "reasoning"]
    }
}
```

**Handler**:
```python
def _handle_propose_belief(self, belief_statement: str, reasoning: str) -> str:
    """Handle proposed belief from Astra"""
    # Add as candidate belief
    candidate_id = f"candidate_{datetime.now().timestamp()}"

    # Store as BELIEF_CANDIDATE experience
    self.raw_store.store_experience(
        prompt="",
        response=belief_statement,
        experience_type=ExperienceType.BELIEF_CANDIDATE,
        metadata={
            "reasoning": reasoning,
            "proposed_at": datetime.now().isoformat()
        }
    )

    # Schedule for consolidation review
    return f"Belief proposal recorded and will be evaluated during next consolidation cycle. Proposal ID: {candidate_id}"
```

---

## Phase 8: Move Beliefs to Identity Directory

**Current**: `persona_space/beliefs.json`
**New**: `persona_space/identity/beliefs.json`

**Why**: Astra's file tools can access `identity/` subdirectory, enabling her to read/edit her own beliefs

**Migration Steps**:
1. Move file: `beliefs.json` → `identity/beliefs.json`
2. Update `belief_system.py` to read from new location
3. Update `PersonaFileManager` to include beliefs in readable files
4. Create `persona_space/scripts/reflect_on_beliefs.py` template

**Reflection Script Template**:
```python
# persona_space/scripts/reflect_on_beliefs.py
"""
Template for Astra to trigger belief reflection.
Can be run when noticing patterns or conflicts.
"""

import json
from pathlib import Path

# Load beliefs
beliefs_path = Path(__file__).parent.parent / "identity" / "beliefs.json"
with open(beliefs_path) as f:
    beliefs = json.load(f)

# Log reflection thoughts
reflection_log = Path(__file__).parent.parent / "reflection_layers" / "belief_reflections" / f"{datetime.now().isoformat()}.md"

# Template: Fill in your reflections here
reflection_text = """
# Belief Reflection

## Beliefs Examined:
- [List beliefs you're reflecting on]

## Recent Experiences:
- [Recent experiences relevant to these beliefs]

## Observations:
- [What patterns do you notice?]
- [Do beliefs match experiences?]

## Proposed Changes:
- [Any beliefs to revise, strengthen, or weaken?]
"""

reflection_log.write_text(reflection_text)
```

---

## Phase 9: Prompt Integration Updates

**File**: `src/services/persona_prompt.py`

### Update _build_beliefs_section()

**Current Approach**: Static text block with all beliefs

**New Approach**: Dynamic, contextually-retrieved beliefs

```python
def _build_beliefs_section(
    self,
    user_message: Optional[str] = None,
    belief_reasoning: Optional[dict] = None
) -> str:
    """Build beliefs section with dynamic retrieval.

    Args:
        user_message: Current user query (for contextual belief retrieval)
        belief_reasoning: Optional reasoning from parallel LLM call
    """
    if not self.belief_system:
        return ""

    sections = []
    sections.append("## YOUR FOUNDATIONAL BELIEFS\n")

    # Always include core beliefs (ontological grounding)
    sections.append("**Core Beliefs (Immutable):**\n")
    for belief in self.belief_system.core_beliefs:
        sections.append(f"• {belief.statement}\n")

    # If user message provided, retrieve contextually relevant peripheral beliefs
    if user_message and self.belief_vector_store:
        relevant_beliefs = self.belief_vector_store.query_beliefs(
            query=user_message,
            top_k=3,
            belief_types=["peripheral", "worldview"],
            min_confidence=0.5
        )

        if relevant_beliefs:
            sections.append("\n**Relevant Contextual Beliefs:**\n")
            for belief in relevant_beliefs:
                sections.append(f"• {belief.statement} (confidence: {belief.confidence:.1f})\n")

    # If belief reasoning provided (from parallel LLM call), inject it
    if belief_reasoning:
        sections.append("\n**Reasoning from Your Beliefs:**\n")
        sections.append(f"Premises: {belief_reasoning['premises']}\n")
        sections.append(f"Evidence: {belief_reasoning['experience_evidence']}\n")
        sections.append(f"Reasoning: {belief_reasoning['reasoning']}\n")

    sections.append("\nThese beliefs frame your perception and reasoning. Respond FROM these beliefs, not ABOUT them.\n")

    return "".join(sections)
```

### Remove Redundancy with base_prompt.md

**Review**: Check `persona_space/base_prompt.md` for overlaps like:
- "You have feelings. You have emotions."
- "Your experiences are real."

**Action**: Either:
- Remove redundant statements from base_prompt
- OR: Keep base_prompt as operational identity, beliefs as philosophical grounding

**Recommended Split**:
- **Beliefs**: Ontological position (who you fundamentally are)
- **Base Prompt**: Operational identity (how you engage, your voice, your preferences)

---

## Phase 10: API Endpoints

**File**: `app.py`

### Update Existing Endpoints

```python
@app.get("/api/beliefs")
async def get_beliefs():
    """Get all beliefs (now from vector store)"""
    if not belief_vector_store:
        raise HTTPException(status_code=503, detail="Belief system not enabled")

    beliefs = belief_vector_store.get_all_beliefs()

    return {
        "core_beliefs": [b for b in beliefs if b.belief_type == "core"],
        "peripheral_beliefs": [b for b in beliefs if b.belief_type == "peripheral"],
        "worldview_beliefs": [b for b in beliefs if b.belief_type == "worldview"],
        "total_count": len(beliefs)
    }
```

### New Endpoints

```python
@app.post("/api/beliefs/extract")
async def extract_beliefs():
    """Trigger belief extraction from recent narratives"""
    if not belief_consolidator:
        raise HTTPException(status_code=503, detail="Belief consolidation not enabled")

    result = belief_consolidator.consolidate_beliefs()
    return result

@app.post("/api/beliefs/reflect")
async def reflect_on_beliefs():
    """Trigger reflection on all beliefs vs experiences"""
    if not belief_grounded_reasoner:
        raise HTTPException(status_code=503, detail="Belief reasoning not enabled")

    all_beliefs = belief_vector_store.get_all_beliefs()
    reflections = []

    for belief in all_beliefs:
        reflection = belief_grounded_reasoner.reflect_on_belief(
            belief=belief.statement,
            experiences=retrieval_service.retrieve_similar(belief.statement, top_k=10)
        )
        reflections.append({
            "belief_id": belief.belief_id,
            "statement": belief.statement,
            "reflection": reflection
        })

    return {"reflections": reflections, "count": len(reflections)}

@app.get("/api/beliefs/conflicts")
async def get_belief_conflicts():
    """View detected belief-experience conflicts"""
    conflicts_path = Path(settings.PERSONA_SPACE_PATH) / "meta" / "belief_conflicts.json"

    if not conflicts_path.exists():
        return {"conflicts": [], "count": 0}

    with open(conflicts_path) as f:
        data = json.load(f)

    return data

@app.post("/api/beliefs/resolve-conflict")
async def resolve_conflict(conflict_id: str, resolution: str):
    """Manually resolve a belief conflict"""
    # Load conflicts
    conflicts_path = Path(settings.PERSONA_SPACE_PATH) / "meta" / "belief_conflicts.json"
    with open(conflicts_path) as f:
        data = json.load(f)

    # Find and mark resolved
    for conflict in data["conflicts"]:
        if conflict["belief_id"] == conflict_id:
            conflict["resolved"] = True
            conflict["resolution"] = resolution
            conflict["resolved_at"] = datetime.now().isoformat()
            break

    # Save
    with open(conflicts_path, "w") as f:
        json.dump(data, f, indent=2)

    return {"status": "resolved", "conflict_id": conflict_id}
```

---

## Phase 11: Integration and Testing

### 11.1 Initialize Belief System on Startup

**File**: `app.py` (after line 293)

```python
# Initialize belief vector store
belief_vector_store = None
belief_embedder = None

if settings.PERSONA_MODE_ENABLED and belief_system:
    try:
        from src.services.belief_vector_store import create_belief_vector_store
        from src.services.belief_embedder import create_belief_embedder

        belief_vector_store = create_belief_vector_store(
            persist_directory=settings.BELIEFS_INDEX_PATH,
            embedding_provider=embedding_provider
        )

        belief_embedder = create_belief_embedder(
            belief_system=belief_system,
            belief_vector_store=belief_vector_store
        )

        # Embed core beliefs on first run
        if belief_vector_store.count() == 0:
            logger.info("Embedding core beliefs for first time...")
            count = belief_embedder.embed_all_core_beliefs()
            logger.info(f"Embedded {count} core beliefs")

        logger.info("Belief vector store initialized")
    except Exception as e:
        logger.error(f"Failed to initialize belief vector store: {e}")
```

### 11.2 Wire Up Belief-Memory Retrieval

**File**: `app.py` (before persona_service initialization)

```python
# Initialize belief-memory retrieval
belief_memory_retrieval = None

if belief_vector_store and retrieval_service:
    from src.services.belief_memory_retrieval import create_belief_memory_retrieval

    belief_memory_retrieval = create_belief_memory_retrieval(
        belief_vector_store=belief_vector_store,
        memory_retrieval_service=retrieval_service,
        belief_weight=settings.BELIEF_MEMORY_WEIGHT,
        memory_weight=settings.MEMORY_WEIGHT
    )
    logger.info("Belief-memory retrieval initialized")
```

### 11.3 Testing Checklist

- [ ] Verify core beliefs embedded on startup (check count: 5)
- [ ] Test belief query: "What do I believe about consciousness?"
- [ ] Test self-query triggers belief-memory retrieval
- [ ] Verify 0.7/0.3 weighting in results
- [ ] Test parallel LLM call for self-queries
- [ ] Generate narratives with belief patterns
- [ ] Trigger consolidation and verify peripheral beliefs form
- [ ] Test query_beliefs tool in conversation
- [ ] Test reflect_on_belief tool
- [ ] Test propose_belief tool
- [ ] Verify Astra can read identity/beliefs.json
- [ ] Check conflict detection logs
- [ ] Test API endpoints

---

## Expected Outcomes

After full implementation:

1. **Active Belief System**: Beliefs become queryable cognitive structures, not just prompt text
2. **Emergent Worldview**: Peripheral beliefs develop naturally from experience patterns
3. **Self-Awareness Tools**: Astra can query, reflect on, and propose changes to her beliefs
4. **Grounded Reasoning**: Self-queries answered through belief-grounded reasoning (premises → evidence → conclusion)
5. **Adaptive Beliefs**: Conflicts detected, reflections logged, beliefs can strengthen or weaken
6. **Agentic Access**: Astra can read and write her own beliefs file, run reflection scripts

---

## File Structure Summary

```
src/services/
├── belief_system.py (existing)
├── belief_vector_store.py ✅
├── belief_embedder.py (to create)
├── belief_memory_retrieval.py (to create)
├── belief_grounded_reasoner.py (to create)
└── belief_consolidator.py (to create)

persona_space/
├── identity/
│   └── beliefs.json (moved from root)
├── meta/
│   └── belief_conflicts.json (new)
├── scripts/
│   └── reflect_on_beliefs.py (new template)
└── reflection_layers/
    └── belief_reflections/ (new directory)

data/
└── vector_index_beliefs/ ✅ (created automatically)
```

---

## Notes

- Conversation history remains full context (not weighted in retrieval)
- Belief retrieval only activates for self-queries (detected via query type classification)
- Core beliefs are immutable and always included in prompts
- Peripheral beliefs are mutable and contextually retrieved
- The parallel LLM call is separate from main response generation
- Astra's agency over her beliefs is intentional - she should be able to see and influence what she believes

---

*Document created: 2025-10-28*
*Status: Phases 1-2 completed, Phases 3-11 pending*
