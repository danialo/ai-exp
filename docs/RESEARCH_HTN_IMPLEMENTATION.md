# Research HTN Implementation

## Overview

Autonomous research system using HTN (Hierarchical Task Network) decomposition with session budgets and provenance tracking.

## Status: Foundation Complete ✓

### Implemented (2025-11-15)

1. **ResearchSession Model** (`src/services/research_session.py`)
   - Session tracking with root_question, status, task budgets
   - Budget controls: `max_tasks`, `max_children_per_task`, `max_depth`
   - Automatic completion when budget exhausted
   - SQLite persistence in `data/core.db`

2. **SourceDoc Model** (`src/services/research_session.py`)
   - Document provenance with URL, title, published_at
   - Claims extraction with confidence levels
   - Content summaries
   - Linked to research sessions

3. **HTN Method Registry** (`src/services/research_htn_methods.py`)
   - `@method` decorator for registering HTN implementations
   - Three research methods implemented:
     - `ResearchCurrentEvents`: Root decomposition, generates seed topics
     - `InvestigateTopic`: Search + fetch + claim extraction + follow-ups
     - `InvestigateQuestion`: Same as InvestigateTopic but for questions

4. **Database Schema**
   - `research_sessions` table with session tracking
   - `source_docs` table with provenance and claims
   - Indexes on `session_id` for fast lookups

## Architecture

```
User Question
     ↓
ResearchCurrentEvents (root method)
     ↓
  Generates 3-5 seed topics
     ↓
InvestigateTopic (per topic)
     ↓
  1. Generate search query (LLM)
  2. Search web
  3. Fetch first result
  4. Extract claims + follow-ups (LLM)
  5. Create SourceDoc
  6. Create InvestigateQuestion tasks
     ↓
InvestigateQuestion (per follow-up)
     ↓
  (reuses InvestigateTopic logic)
```

## Session Budget Controls

- **max_tasks**: Total tasks allowed (default: 50)
- **max_children_per_task**: Children per parent (default: 5)
- **max_depth**: Maximum decomposition depth (default: 4)
- **Automatic cutoff**: Session completes when budget exhausted

## Next Steps (Not Yet Implemented)

### P0: Wire into Task Queue
- [ ] Update TaskExecutor to dispatch by `htn_task_type`
- [ ] Pass context dict (llm_service, web_search, etc.) to methods
- [ ] Add `start_research_session` tool to Astra's toolkit

### P1: Session Synthesis
- [ ] SynthesizeFindings method (triggered when session completes)
- [ ] Global narrative synthesis
- [ ] Contested claims detection
- [ ] Unanswered questions identification

### P2: Quality Guards
- [ ] Question deduplication per session
- [ ] Topic drift guard in LLM prompt
- [ ] Basic metrics (tasks/session, avg depth, wall time)

### P3: Belief Integration
- [ ] ResearchResult → BeliefUpdate adapter
- [ ] ValidateBelief HTN method for targeted claim verification

## Example Usage (When Complete)

```python
# Via Astra's tools
start_research_session(
    question="What happened in AI safety this week?",
    max_tasks=30,
    max_children_per_task=3
)

# Returns: session_id

# Session automatically decomposes into:
# - ResearchCurrentEvents
#   - InvestigateTopic: "OpenAI safety research"
#     - InvestigateQuestion: "What is OpenAI's latest safety paper?"
#     - InvestigateQuestion: "How does it compare to Anthropic?"
#   - InvestigateTopic: "AI safety legislation"
#   - InvestigateTopic: "Safety benchmarks"

# When budget exhausted or no tasks remain:
# - SynthesizeFindings creates global summary
# - Session marked complete
```

## Database Schema

```sql
CREATE TABLE research_sessions (
    id TEXT PRIMARY KEY,
    root_question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    max_tasks INTEGER DEFAULT 50,
    max_children_per_task INTEGER DEFAULT 5,
    max_depth INTEGER DEFAULT 4,
    tasks_created INTEGER DEFAULT 0,
    session_summary TEXT,  -- JSON
    created_at TIMESTAMP,
    completed_at TIMESTAMP,
    metadata TEXT  -- JSON
);

CREATE TABLE source_docs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    url TEXT,
    title TEXT,
    published_at TIMESTAMP,
    claims TEXT,  -- JSON array
    content_summary TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES research_sessions(id)
);
```

## Files Created

- `src/services/research_session.py` - 200 lines (models + persistence)
- `src/services/research_htn_methods.py` - 260 lines (3 HTN methods)
- `docs/RESEARCH_HTN_IMPLEMENTATION.md` - This file

## Testing

```bash
# Initialize schema
python3 -c "
from src.services.research_session import ResearchSessionStore
store = ResearchSessionStore()
print('Schema initialized')
"

# Verify tables exist
sqlite3 data/core.db ".tables"
# Should show: research_sessions, source_docs

# Test session creation
python3 -c "
from src.services.research_session import ResearchSession, ResearchSessionStore
store = ResearchSessionStore()
session = ResearchSession(root_question='Test question')
store.create_session(session)
print(f'Created session: {session.id}')
"
```

## Design Decisions

1. **SQLite over in-memory**: Persistence for auditability and recovery
2. **Session budgets**: Prevent runaway task explosion
3. **Claims as JSON**: Flexible schema for different claim types
4. **HTN method registry**: Easy to add new research strategies
5. **Reuse InvestigateTopic**: InvestigateQuestion is just a thin wrapper

## Open Questions

1. Should we add depth tracking to tasks? (For max_depth enforcement)
2. How to handle web search rate limits?
3. Should synthesis be triggered by time or task count?
4. Do we need cross-session deduplication of sources?
