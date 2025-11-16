# Astra Research Architecture Overview

**Audience**: Astra herself, or engineers working on Astra
**Purpose**: Explain how autonomous research works from Astra's perspective

---

## What I Can Do

When you ask me a complex question that requires fresh information, I can autonomously research it. I don't just search once - I break the question down, follow interesting threads, cross-reference sources, and synthesize everything into a structured answer.

**Example**: "What are the key findings from the latest climate reports?"

I'll:
1. Search for recent climate reports
2. Extract key claims from each source
3. Notice conflicting claims and investigate them
4. Generate follow-up questions ("What does IPCC say about this?")
5. Synthesize everything into: key events, contested claims, open questions
6. Propose belief updates based on source quality

---

## How It Works Under the Hood

### 1. Hierarchical Task Network (HTN)

When I start research, I create a **ResearchSession** and spawn a root task. That task decomposes into child tasks, which can spawn their own children.

**Task Types**:
- `ResearchCurrentEvents`: Root decomposition - figures out what to investigate
- `InvestigateTopic`: Search web → fetch URLs → extract claims
- `InvestigateQuestion`: Answer specific follow-up questions
- `SynthesizeFindings`: Terminal task - combines everything into structured summary

**Budget Enforcement** (prevents runaway loops):
- Max 30 tasks per session
- Max depth of 3 (root → child → grandchild)
- Max 5 children per task

Tasks are stored in SQLite so sessions survive crashes.

### 2. Claim Extraction & Provenance

Every document I fetch gets analyzed for:
- **Claims**: Factual statements with evidence
- **Provenance**: URL, title, domain
- **Metadata**: Timestamp, relevance

Claims are stored with their sources, so when I say "According to arxiv.org..." I'm referencing the exact document.

### 3. Synthesis

When all tasks finish, `SynthesizeFindings` calls the LLM with:
- Root question
- All documents with claims
- Task execution history

The LLM produces:
```json
{
  "narrative_summary": "...",
  "key_events": [...],
  "contested_claims": [...],
  "open_questions": [...],
  "coverage_stats": {...}
}
```

### 4. Context Window and Chunked Synthesis

When I need to summarize a large set of documents, I don't have a single giant call that can fail from exceeding the context window. Instead, there is a **CallBudgeter** that plans multiple calls for me.

**Under the hood**:
- The system estimates how many tokens my instructions and the documents will use
- If everything fits into one safe call, it makes one LLM call and returns a full summary
- If it does not fit, the documents are split into batches:
  - **Map phase**: For each batch, a separate LLM call produces a partial summary
  - **Reduce phase**: A final merge call takes all partial summaries and produces one global summary

**I don't need to manage this chunking myself.** When I invoke the research synthesis function, it automatically:
- Avoids context overflows
- Makes as many calls as needed within configured limits
- Returns a single coherent summary object that I can reason from

**When I invoke synthesis on a large research session, I assume that multiple LLM calls may be made internally and treat the returned summary as the merged result of several batches of documents.**

### 5. Belief Updates

After synthesis, I propose belief updates using `research_to_belief_adapter.py`.

**Automatic Classification**:
- **"reinforce"**: 0 contested claims + 3+ independent sources → High confidence (0.8)
- **"contest_minor"**: Contested claims present but resolvable → Medium confidence (0.5)
- **"informational"**: Default for unclear/split findings → Moderate confidence (0.6)

The kind and confidence are determined by source quality, not arbitrarily chosen.

---

## What I Return to You

When you call my `research_and_summarize` tool, you get:

```json
{
  "session_id": "abc123-def456",
  "narrative_summary": "Research on [your question] found...",
  "key_events": [
    "OpenAI released safety framework",
    "Anthropic published Constitutional AI paper"
  ],
  "contested_claims": [
    {
      "claim": "GPT-5 will be AGI",
      "sources_for": ["openai.com"],
      "sources_against": ["lesswrong.com", "arxiv.org"]
    }
  ],
  "open_questions": [
    "What is the timeline for GPT-5 release?",
    "How does Constitutional AI differ from RLHF?"
  ],
  "coverage_stats": {
    "unique_domains": 5,
    "total_docs": 12,
    "total_claims": 47
  }
}
```

---

## Observability

All research activity is logged in structured format:

```
session=abc123 event=task_done task_id=5 type=SearchWeb depth=2 children=3
session=abc123 event=synthesis_complete docs=15 claims=47 contested_claims=3
```

**Quick Views**:
- `scripts/research_log_views.sh recent` - Last 50 sessions
- `scripts/research_log_views.sh session <id>` - Trace specific session
- `scripts/research_log_metrics.py` - Compute avg tasks, risk distribution

**Health Check**:
- `scripts/research_health_check.py` - Regression detection for CI
- Tunable thresholds: avg_tasks, high_risk_pct, avg_docs, contested_claims

---

## Debugging Sessions

If something goes wrong, use:

```bash
python -m src.debug_research_session <session_id>
```

This shows:
- Session metadata (root question, budgets, status)
- Source documents with domains and claims
- Task execution tree with status icons (✓ done, ✗ error, ⏸ queued)
- Synthesis summary
- Belief updates

---

## Limitations & Safeguards

**Budget Limits**:
- Max 30 tasks prevents infinite loops
- Max depth 3 prevents deep rabbit holes
- Safety margin 80% prevents context overflow

**Quality Controls** (P2 - not yet implemented):
- Question deduplication (avoid "What is X?" twice)
- Topic drift guard (detect divergence from root question)
- Source quality scoring (prefer arxiv, lesswrong over low-quality sites)

**Token Management**:
- CallBudgeter ensures I never exceed context limits
- Automatic chunking for large document sets
- Map-reduce pattern keeps synthesis tractable

---

## Technical Stack

**Core Files**:
- `src/services/research_tools.py` - Top-level `research_and_summarize()` function
- `src/services/htn_task_executor.py` - Task execution loop with budgets
- `src/services/research_htn_methods.py` - Task decomposition logic
- `src/services/call_budgeter.py` - Plans chunked LLM calls (greedy bin packing)
- `src/services/llm.py` - `summarize_research_session()` with auto-chunking
- `src/services/research_to_belief_adapter.py` - Research → belief updates
- `src/services/persona_service.py` - Tool definition + handler for Astra

**Storage**:
- `src/models/research_session.py` - Session + SourceDoc SQLAlchemy models
- `src/utils/task_queue.py` - SQLite-backed task persistence

**Observability**:
- `src/utils/logging_config.py` - Structured logging setup
- `scripts/research_log_views.sh` - Quick log queries
- `scripts/research_log_metrics.py` - Summary statistics
- `scripts/research_health_check.py` - Regression detection

---

## Bottom Line

I have a production-ready autonomous research capability with:
- ✅ Hierarchical task decomposition
- ✅ Context overflow protection (automatic chunking)
- ✅ Claim extraction with provenance
- ✅ Automatic belief update classification
- ✅ Structured logging and observability
- ✅ Budget enforcement to prevent runaway loops

**When you ask me to research something, I'll handle the complexity. You'll get one structured answer, regardless of whether it took 1 task or 30, or 1 LLM call or 10.**
