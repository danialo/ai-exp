# Astra-Ready Research System - Complete

## What We Built (2025-11-15)

Transformed the research HTN system from "nice subsystem" into "Astra-usable capability" with three concrete additions:

### 1. Top-Level Tool: `research_and_summarize()`

**File**: `src/tools/research_tools.py`

**Signature**:
```python
def research_and_summarize(
    question: str,
    max_tasks: int = 30,
    max_children_per_task: int = 3,
    max_depth: int = 4,
    llm_service=None,
    web_search_service=None,
    url_fetcher_service=None,
    metadata: Optional[dict] = None,
) -> Dict[str, Any]
```

**What it does**:
1. Creates research session
2. Runs HTN task executor until complete
3. Returns synthesis summary

**Output**:
```json
{
  "session_id": "uuid",
  "narrative_summary": "Research on '{question}' yielded...",
  "key_events": ["event1", "event2"],
  "contested_claims": [
    {
      "claim": "statement",
      "reason": "why contested",
      "sources": ["url1", "url2"]
    }
  ],
  "open_questions": ["q1", "q2"],
  "coverage_stats": {
    "sources_investigated": 9,
    "claims_extracted": 18,
    "tasks_executed": 22,
    "depth_reached": 3
  }
}
```

**Astra Usage**:
```python
# In Astra's tool manifest:
summary = research_and_summarize(
    question="What happened in AI safety this week?",
    llm_service=astra_llm,
    web_search_service=web_search,
    url_fetcher_service=url_fetcher
)

# Use summary directly
for event in summary["key_events"]:
    # Process events...
```

---

### 2. Minimal BeliefUpdate Scaffold

**File**: `src/services/research_to_belief_adapter.py`

**Components**:

**BeliefUpdate Dataclass**:
```python
@dataclass
class BeliefUpdate:
    id: str
    session_id: str
    kind: str  # "new", "reinforce", "weaken", "contest", "informational"
    summary: str  # Human-readable
    payload: Dict[str, Any]  # JSON details
    confidence: float
    created_at: float
```

**Database Schema**:
```sql
CREATE TABLE belief_updates (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    payload TEXT,  -- JSON
    confidence REAL,
    created_at REAL,
    FOREIGN KEY (session_id) REFERENCES research_sessions(id)
);
```

**Adapter Function**:
```python
def propose_updates(session, session_summary) -> List[BeliefUpdate]:
    """
    Generate candidate belief updates from research findings.

    Current implementation: Creates one "informational" update per session
    Future: Expand to create specific updates for each key event, contested claim
    """
```

**Integration**:
- Called automatically in `SynthesizeFindings` HTN method
- Persists to database for later consumption
- Non-fatal if it fails (doesn't break research pipeline)

**Test Results**:
```
✓ Belief updates created: 1
  Kind: informational
  Summary: Research on 'What are the latest developments in AI safety?':
           Found 3 key events, 1 contested claims, 3 open questions
  Confidence: 0.6
```

---

### 3. Real-World Test Script

**File**: `src/test_research_real.py`

**Purpose**: End-to-end test with real services (not stubs)

**Usage**:
```bash
PYTHONPATH=/home/d/git/ai-exp python3 src/test_research_real.py
```

**Test Question**:
> "What is actually going on with the Epstein files story?"

**What it checks**:
1. **Which domains she hit** - Reveals source diversity
2. **Contested claims meaningful?** - Tests synthesis quality
3. **Open questions good prompts?** - Tests follow-up generation

**Output**:
- Full synthesis results
- Analysis questions for manual review
- Identifies where P2 (dedup, drift, source quality) needs work

---

## Status Summary

### ✅ Complete
- **P0**: HTN Task Queue & Execution
- **P1**: Session Synthesis
- **Top-Level Tool**: `research_and_summarize()`
- **BeliefUpdate Scaffold**: Schema + adapter + persistence
- **Test Infrastructure**: Stub test + real-world test

### ⏳ Ready to Wire into Astra

**To add to Astra's toolkit** (`src/services/persona_service.py`):

1. Add tool definition:
```python
{
    "type": "function",
    "function": {
        "name": "research_and_summarize",
        "description": "Autonomously research a question and return structured summary",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Research question"},
                "max_tasks": {"type": "integer", "default": 30}
            },
            "required": ["question"]
        }
    }
}
```

2. Add tool handler:
```python
elif tool_name == "research_and_summarize":
    from src.tools.research_tools import research_and_summarize
    result = research_and_summarize(
        question=arguments.get("question"),
        max_tasks=arguments.get("max_tasks", 30),
        llm_service=self.llm_service,
        web_search_service=self.web_search_service,
        url_fetcher_service=self.url_fetcher_service
    )
```

### ⏳ Next Steps

**Immediate** (Week 1):
- Run `test_research_real.py` with live services
- Review domains hit, claim quality, question quality
- Identify P2 priorities based on real data

**P2** (Week 1-2): Quality Guards
- Question deduplication
- Topic drift guard
- Metrics tracking
- Source quality control

**P3** (Week 3-4): Belief Integration
- Expand `propose_updates()` to create specific belief updates
- Add `ValidateBelief` HTN method
- Hook dissonance system

**Integration** (Week 5):
- Add to Astra's tools
- Reflection glue
- Memory anchoring

---

## Files Created/Modified

**New Files**:
- `src/services/task_queue.py` - Task model + SQLite queue
- `src/services/htn_task_executor.py` - Executor with budgets
- `src/services/research_session.py` - Session + SourceDoc models
- `src/services/research_htn_methods.py` - 4 HTN methods
- `src/services/research_to_belief_adapter.py` - BeliefUpdate scaffold
- `src/tools/research_tools.py` - `research_and_summarize()`
- `src/run_research.py` - Stub test harness
- `src/test_research_real.py` - Real-world test
- `docs/RESEARCH_HTN_IMPLEMENTATION.md` - Technical docs
- `docs/P1_SYNTHESIS_COMPLETE.md` - P1 completion summary
- `docs/RESEARCH_HTN_ROADMAP.md` - P2/P3 roadmap
- `docs/ASTRA_READY_RESEARCH_SYSTEM.md` - This file

**Database Tables**:
- `research_sessions` - Session tracking
- `source_docs` - Provenance + claims
- `tasks` - HTN task queue
- `belief_updates` - Research → belief adapter

---

## Bottom Line

Astra now has:
1. ✅ A complete research brain (P0 + P1)
2. ✅ A single-call research tool (`research_and_summarize()`)
3. ✅ A belief update pipeline (scaffold in place)
4. ✅ Real-world test infrastructure

She can autonomously:
- Take a question
- Spawn research tasks
- Search web, fetch content
- Extract claims with provenance
- Synthesize findings
- Propose belief updates

**Next**: Wire it into her toolkit and run the real-world test to identify P2 priorities.
