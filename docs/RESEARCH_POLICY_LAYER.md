# Research System - Policy Layer Complete

**Date**: 2025-11-15
**Status**: ✅ Complete

## Overview

The research subsystem now has a complete **policy and behavior layer** on top of the technical implementation. This teaches Astra *when* to use research, *how* to speak from research results, and how to avoid redundant work.

---

## What Was Added

### 1. ✅ Format & Presentation Layer (`research_formatter.py`)

**File**: `src/services/research_formatter.py`

**Purpose**: Convert raw synthesis into conversational answers with trust calibration.

**Key Functions**:

- **`format_research_answer(summary_obj, source_docs)`**:
  - Structures research into readable format:
    - Risk-calibrated opening
    - Key events (bullet list, 3-5)
    - Points of disagreement (contested_claims)
    - What's still unclear (open_questions)
    - Source note (provenance clustering)
  - Limits output to prevent overwhelming user
  - Uses domain clustering for provenance descriptions

- **`_assess_risk_level(contested_claims, stats)`**:
  - Returns "low", "medium", or "high" based on:
    - Number of contested claims
    - Source count
    - Source quality (future: can parse claim reasons)

- **`_cluster_domains(docs)`**:
  - Groups sources by type:
    - "wire": AP, Reuters, UPI
    - "major_news": NYT, WaPo, WSJ, Guardian, BBC
    - "academic": arXiv, Nature, Science, PubMed
    - "court": CourtListener, PACER
    - "other": Everything else

- **`_format_provenance_summary(clusters)`**:
  - Converts clusters to natural language:
    - "major wire services and court documents"
    - "academic sources and 5 independent sources"

- **`get_risk_hedging_guidance(risk_level)`**:
  - Returns guidance for how to speak given risk:
    - Low: "Answer assertively"
    - Medium: "Light hedging, explicitly mention disputes"
    - High: "Heavy hedging, label speculation vs fact"

**Integration**: Persona service now uses `format_research_answer()` instead of raw JSON dump.

---

### 2. ✅ Policy Guidelines (Astra's Base Prompt)

**File**: `persona_space/meta/base_prompt.md` (lines 235-336)

**New Section**: "Autonomous Research System"

**When to Call research_and_summarize**:

**DO use it when:**
1. Current events, live politics, recent news, ongoing investigations
2. Answer materially depends on up-to-date facts
3. User asks for verification of contested claim
4. Topic requires multiple independent sources

**DO NOT use it when:**
1. General background knowledge, not time-sensitive
2. User wants analysis/argument structure/conceptual breakdown
3. Already researched essentially same question in this conversation

**Reuse heuristic:**
- Check for recent research first (use `check_recent_research`)
- Prefer reusing session findings
- Do quick follow-up instead of full research from scratch

**How to Speak from Research Results**:

1. **Opening** (2-4 sentences):
   - Direct answer
   - Risk-calibrated hedging:
     - High risk: "This is early/contested. Here's what's known..."
     - Medium risk: "Some details disputed, but..."
     - Low risk: Answer assertively

2. **Key Events** (3-5 bullet points)

3. **Points of Disagreement** (if contested):
   - List conflicting claims
   - Explain why disputed
   - Be honest - present both sides

4. **What's Still Unclear** (if open questions exist)

5. **Source Note**:
   - Mention source types, not raw URLs
   - "major wire services", "court documents", etc.

**Trust Calibration - How to Hedge**:

- **Low risk**: Multiple sources agree → Answer assertively
- **Medium risk**: Some disputes → Light hedging ("According to most sources...")
- **High risk**: Central contested claims → Heavy hedging ("Early reports suggest...")

**Cost Awareness**:
- Each call hits multiple URLs, runs LLM synthesis, creates DB records
- Use wisely - not for every proper noun
- Prefer local knowledge when sufficient
- Reuse session findings when possible

---

### 3. ✅ Session Anchor/Reuse System

**File**: `src/services/research_anchor_store.py`

**Purpose**: Lightweight pointers to completed research sessions to avoid re-investigating same topics.

**ResearchAnchor Dataclass**:
```python
@dataclass
class ResearchAnchor:
    session_id: str
    topic: str  # Normalized root question
    one_sentence_summary: str
    created_at: float
    days_valid: int = 7
```

**ResearchAnchorStore**:
- `create_anchor()`: Persist anchor after session completes
- `find_recent_anchor(topic, max_age_days=7)`: Search for existing research
- `list_recent_anchors()`: List all fresh anchors
- `cleanup_old_anchors()`: Garbage collection

**Database Table**:
```sql
CREATE TABLE research_anchors (
    session_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    one_sentence_summary TEXT NOT NULL,
    created_at REAL NOT NULL,
    days_valid INTEGER DEFAULT 7,
    FOREIGN KEY (session_id) REFERENCES research_sessions(id)
)
```

**Integration**:
- `SynthesizeFindings` method creates anchor automatically (src/services/research_htn_methods.py:264-270)
- Anchor uses first key event as one-sentence summary

---

### 4. ✅ check_recent_research Tool

**File**: `src/services/persona_service.py` (lines 1456-1471, 2040-2084)

**Tool Definition**:
```json
{
  "name": "check_recent_research",
  "description": "Check if you recently researched a similar topic...",
  "parameters": {
    "topic": "string (required)"
  }
}
```

**Behavior**:
- Checks `research_anchors` table for normalized topic
- If found (within 7 days):
  - Loads session summary
  - Loads source docs
  - Formats using `format_research_answer()`
  - Returns formatted answer with age info
  - Suggests reusing findings or doing targeted follow-up
- If not found:
  - Returns "No recent research found"
  - Suggests calling `research_and_summarize`

**Integration**: Astra's prompt instructs her to use this BEFORE `research_and_summarize`.

---

## How It Works Together

### Happy Path: Fresh Research

1. User: "What's happening with the Supreme Court affirmative action ruling?"
2. Astra: [Calls `check_recent_research("affirmative action ruling")`]
3. Tool: "No recent research found"
4. Astra: [Calls `research_and_summarize("What's happening with the Supreme Court affirmative action ruling?")`]
5. Research subsystem:
   - Runs HTN task decomposition
   - Fetches sources, extracts claims
   - Synthesizes findings
   - Classifies belief updates
   - **Creates research anchor**
6. Tool returns formatted answer with:
   - Risk assessment (low/medium/high)
   - Structured presentation
   - Provenance clustering
7. Astra presents to user using risk-calibrated hedging

### Happy Path: Reuse Existing Research

1. User: "Did that ruling get appealed?"
2. Astra: [Calls `check_recent_research("affirmative action ruling")`]
3. Tool: "EXISTING RESEARCH FOUND: affirmative action ruling"
   - Session ID: abc-123
   - Age: 2.5 hours ago
   - [Formatted answer from previous research]
4. Astra: [Reuses findings, answers directly without new research]

### Happy Path: Targeted Follow-up

1. User (follow-up): "What arguments did the dissent make?"
2. Astra: [Recognizes this is narrow follow-up to previous research]
3. Astra: [Calls `research_and_summarize` with smaller budget: max_tasks=10]
4. Research subsystem: Focused investigation on dissent arguments only
5. Returns concise answer

---

## Files Modified/Created

### New Files:
- `src/services/research_formatter.py` - Format & presentation layer (264 lines)
- `src/services/research_anchor_store.py` - Session reuse system (196 lines)
- `docs/RESEARCH_POLICY_LAYER.md` - This file

### Modified Files:
- `persona_space/meta/base_prompt.md` - Added research policy section (lines 235-336)
- `src/services/persona_service.py`:
  - Added `check_recent_research` tool definition (lines 1456-1471)
  - Added `check_recent_research` handler (lines 2040-2084)
  - Modified `research_and_summarize` handler to use formatter (lines 2086-2133)
- `src/services/research_htn_methods.py`:
  - Added anchor creation in `SynthesizeFindings` (lines 264-270)

---

## What This Enables

### Before Policy Layer:
- Astra had research capability but no guidance on when/how to use it
- Raw synthesis JSON dumped to user
- No concept of risk assessment
- No reuse of previous research → redundant work
- No provenance clustering → URL spam

### After Policy Layer:
- ✅ Clear when-to-use rules ("current events" vs "general knowledge")
- ✅ Trust-calibrated presentation (low/medium/high risk)
- ✅ Automatic session reuse (checks anchors first)
- ✅ Provenance clustering ("major wire services" not raw URLs)
- ✅ Structured, readable answers
- ✅ Cost awareness (avoid redundant research)

---

## Testing Strategy

### Unit Tests (Future):
- `research_formatter.py`:
  - Test risk assessment logic
  - Test domain clustering
  - Test provenance summary generation
- `research_anchor_store.py`:
  - Test anchor creation/retrieval
  - Test topic normalization
  - Test age-based filtering

### Integration Tests (Future):
- Full flow: research → anchor creation → reuse
- Test formatter receives real synthesis objects
- Test check_recent_research finds and formats existing sessions

### Manual Testing (Now):
Run benchmark suite through Astra herself:
```bash
# Test 1: Fresh research
User: "What happened in AI safety this week?"
Expected: Astra checks anchors (none found), calls research_and_summarize, returns formatted answer

# Test 2: Reuse
User: "Tell me more about that AI safety stuff"
Expected: Astra checks anchors (found), reuses session, no new research

# Test 3: Targeted follow-up
User: "What did OpenAI specifically do?"
Expected: Astra does small follow-up research (max_tasks=10) or reuses if sufficient
```

---

## Next Steps

### Immediate:
1. **Run benchmark through Astra** - Test policy layer behavior
2. **Manual review** - Check hedging quality, reuse logic, cost savings

### P2 (Quality Guards):
- Question deduplication (prevent "What is X?" → "What is X?" in same session)
- Topic drift guard (detect when follow-ups diverge)
- Source quality scoring (prefer arxiv over random blogs)

### P3 (Advanced):
- Semantic similarity for anchor matching (not just exact topic match)
- Anchor expiration based on topic volatility (politics: 1 day, science: 30 days)
- Cross-conversation anchor reuse (check anchors from different chats)

---

## Bottom Line

**Before**: Research subsystem worked, but was a "dumb tool" - Astra didn't know when to use it or how to present results.

**After**: Research is now a **guided capability** with:
- Clear usage policy ("when current events" not "every proper noun")
- Trust-calibrated presentation (risk-based hedging)
- Automatic reuse (check anchors first)
- Cost awareness (prevent redundant work)

**Status**: Policy layer complete. Ready for real-world testing with Astra.

**Next**: Run benchmark suite through Astra herself to validate that she follows the policy correctly.
