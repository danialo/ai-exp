# Research HTN System - Implementation Roadmap

## Status Overview

- ✅ **P0: HTN Task Queue & Execution** (Complete 2025-11-15)
- ✅ **P1: Session Synthesis** (Complete 2025-11-15)
- ⏳ **P2: Quality Guards** (Not Started)
- ⏳ **P3: Belief Integration** (Not Started)
- ⏳ **Astra Integration** (Not Started)

---

## P2: Quality Guards

### 1. Question Deduplication (Real, Not Just dedup_key)

- [ ] Add `research_session_questions` table:

```sql
CREATE TABLE research_session_questions (
  session_id TEXT NOT NULL,
  norm_hash TEXT NOT NULL,
  raw_text TEXT NOT NULL,
  PRIMARY KEY (session_id, norm_hash),
  FOREIGN KEY (session_id) REFERENCES research_sessions(id)
);
```

- [ ] Implement `normalize_question(text)`:
  - lowercase, trim whitespace
  - strip punctuation
  - collapse multiple spaces

- [ ] In `_enforce_budgets_and_enqueue`:
  - Compute `norm_hash = sha256(normalize_question(q))[:32]`
  - Check `research_session_questions` before accepting a proposal
  - Insert on accept

**Benefit**: Hard dedup across session, not just in-memory

---

### 2. Topic Drift Guard

**Goal**: Prevent followups like "what are Taylor Swift's political views" when you are on Epstein emails.

- [ ] Add `llm_service.score_relevance(child_question, parent_question, root_question)` → 0 to 1

- [ ] In `InvestigateTopic` and `InvestigateQuestion`:
  - For each proposed followup, call relevance scorer
  - Drop anything below threshold (e.g., 0.4)

- [ ] Optionally log dropped questions to `drift_events` table for debugging

**Implementation note**: LLM prompt is easier to start than embeddings cosine similarity.

---

### 3. Metrics and Telemetry

At minimum:

- [ ] New table `research_session_metrics`:
  - `session_id`
  - `total_tasks`
  - `max_depth_reached`
  - `source_doc_count`
  - `avg_claims_per_doc`
  - `contested_claims_count`
  - `open_questions_count`

- [ ] Populate after SynthesizeFindings runs:
  - Compute from `tasks` and `source_docs` and `session_summary`

- [ ] Add simple logging summary per session in executor:
  - `logger.info(f"Session {id}: tasks={n_tasks}, docs={n_docs}, depth={max_depth}")`

**Benefit**: See quickly when she's going off the rails

---

### 4. Basic Source Quality Control

Right now every link is equal. That will bite you.

- [ ] Extend `source_docs` with:
  - `source_domain TEXT`
  - `quality_score REAL`

- [ ] Parse `source_domain` from URL on insert

- [ ] In `llm_service.extract_summary_claims_questions`:
  - Ask the model to rate source reliability on a coarse scale (e.g., 1 to 3)

- [ ] In synthesis:
  - When aggregating contested claims, surface whether they come mostly from low quality sources

**Later**: Add static allowlist or soft denylist by domain

---

## P3: Belief Integration

This is where Astra stops being just a research engine and starts changing her world model.

### 1. Define BeliefUpdate Schema

- [ ] New dataclass and table `belief_updates`:

```sql
CREATE TABLE belief_updates (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  created_at TIMESTAMP,
  target_belief_id TEXT,   -- nullable for new beliefs
  summary TEXT,            -- human readable
  diff JSON,               -- machine readable changes
  confidence REAL,
  FOREIGN KEY (session_id) REFERENCES research_sessions(id)
);
```

- [ ] `diff` could include:
  - `proposed_truth_value`
  - `new_confidence`
  - `supporting_claim_ids`
  - `contradicting_claim_ids`

---

### 2. Adapter from session_summary to BeliefUpdate

- [ ] Implement `research_to_belief_adapter.from_session_summary(session, summary)`:
  - For each key event or contested claim, generate candidate belief updates
  - Attach provenance (SourceDoc ids, claim indices)
  - Set initial confidence based on:
    - number of supporting claims
    - source quality
    - presence of strong contradiction

- [ ] Call this adapter at the end of SynthesizeFindings and write rows into `belief_updates`

---

### 3. HTN Method: ValidateBelief

- [ ] Add `ValidateBelief` HTN method:

Flow:
```
ValidateBelief(belief_id)
  -> Start a new ResearchSession seeded with the belief statement as root_question
  -> Run standard InvestigateTopic / InvestigateQuestion loop
  -> SynthesizeFindings
  -> Run adapter to produce a BeliefUpdate for that belief_id
```

- [ ] Add a tool `validate_belief(belief_id)` that:
  - Looks up belief text
  - Creates a new research session
  - Runs executor
  - Returns the resulting BeliefUpdate to Astra

---

### 4. Hook from Dissonance System

Once this is wired:

- [ ] When belief dissonance is detected, enqueue a `ValidateBelief` task rather than just logging it

- [ ] Give dissonance engine a "cooldown" per belief, so it doesn't spam research on the same conflict repeatedly

---

## Astra Integration

Now that research and synthesis are solid, give her clean surfaces.

### 1. Tools in persona_service

Add three tools Astra can see:

- [ ] `start_research_and_wait(question, max_tasks=..., ...)`
  - Creates session
  - Runs executor synchronously
  - Returns `session_summary` JSON

- [ ] `start_research_async(question, ...)`
  - Creates session and root task
  - Returns `session_id` only
  - Executor runs separately

- [ ] `get_research_session_summary(session_id)`
  - Reads `session_summary` and returns it

---

### 2. Reflection Glue

- [ ] Add an internal Astra function `reflect_on_research_summary(summary)` that:
  - Writes a short first person internal narrative
  - Extracts 3-5 "things I now think are more likely"
  - Optionally triggers BeliefUpdate consumption

**Note**: This can be a local prompt that turns the structured summary into her "thoughts".

---

### 3. Memory Anchoring

- [ ] When a research session completes, write a pointer into her episodic memory:
  - Topic
  - Session id
  - Short one line summary

- [ ] Let future prompts refer to "my previous research on X" by looking up those anchors and pulling the associated summary

---

## Implementation Priority

**Week 1-2**: P2 Quality Guards
- Question dedup (prevents redundant work)
- Topic drift guard (prevents going off-rails)
- Metrics (visibility into what's happening)

**Week 3-4**: P3 Belief Integration
- BeliefUpdate schema
- Adapter from synthesis
- ValidateBelief HTN method

**Week 5**: Astra Integration
- Add tools to persona_service
- Reflection glue
- Memory anchoring

**Timeline**: ~5 weeks to full production integration with Astra's belief system
