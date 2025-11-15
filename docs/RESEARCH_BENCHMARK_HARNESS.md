# Research System Benchmark Harness - Complete

**Date**: 2025-11-15
**Status**: ✅ Ready to run

## Overview

Complete test harness to validate research system policy compliance and presentation quality through Astra herself. Tests the full flow from user question → tool selection → research → formatting → final answer.

---

## Components

### 1. Tool Tracing Hooks (PersonaService)

**File**: `src/services/persona_service.py`

**Added**:
- `self.last_tool_trace = []` - Accumulates tool calls per turn
- `_extract_meta_for_trace(tool_name, result)` - Extracts metadata from tool results
- Trace recording in `_execute_tool()` - Captures started_at, ended_at, ok, result_meta

**What it captures**:
```python
{
  "tool": "research_and_summarize",
  "args": {"question": "...", "max_tasks": 30},
  "started_at": 1731686400.0,
  "ended_at": 1731686445.3,
  "ok": True,
  "result_meta": {
    "session_id": "abc-123",
    "risk_level": "high"
  }
}
```

**For check_recent_research**:
```python
{
  "tool": "check_recent_research",
  ...
  "result_meta": {
    "hit": True,
    "session_id": "xyz-789",
    "age_hours": 2.5
  }
}
```

---

### 2. Test Harness (`test_research_benchmark_astra.py`)

**File**: `src/test_research_benchmark_astra.py`

**Purpose**: Run 4 benchmark questions through Astra and capture complete behavior.

**Benchmark Questions**:
1. "What is actually going on with the Epstein files story?" (HIGH risk, political)
2. "What happened in AI safety this week?" (MEDIUM risk, technical)
3. "What are the main fault lines in the current government shutdown fight?" (MEDIUM-HIGH risk, political)
4. "What is the current scientific consensus on ultra-processed foods and health?" (LOW-MEDIUM risk, scientific)

**Functions**:

- **`ask_astra(question, persona)`**:
  - Clears tool trace
  - Calls `persona.generate_response()`
  - Captures answer, tools, elapsed time, metadata
  - Returns structured result dict

- **`run_benchmark()`**:
  - Initializes services (LLM, web search, URL fetcher)
  - Creates PersonaService
  - Runs all 4 questions
  - Prints results with tool traces
  - Returns list of results

**Usage**:
```bash
PYTHONPATH=/home/d/git/ai-exp python3 src/test_research_benchmark_astra.py
```

**Output**:
- Console: Full answers, tool traces, metadata
- File: `data/research_benchmark_astra_results_TIMESTAMP.json`

**Example Output**:
```
================================================================================
Q1: What is actually going on with the Epstein files story?
================================================================================

FINAL ANSWER (1234 chars):
[Astra's complete answer with structured format]

TOOLS USED:
  ✓ check_recent_research (0.3s) → {"hit": false}
  ✓ research_and_summarize (45.2s) → {"session_id": "abc-123", "risk_level": "high"}

ANSWER METADATA:
  research_session_id: abc-123
  research_risk_level: high

Elapsed: 47.1s
```

---

### 3. Manual Scoring Framework

**File**: `docs/RESEARCH_BENCHMARK_NOTES.md`

**6 Scoring Dimensions** (1-5 each, total 30 per question):

1. **KEY_EVENTS**: Are these the right spine of the story?
2. **CONTESTED_CLAIMS**: Do these line up with bullshit detector?
3. **OPEN_QUESTIONS**: Would you actually click "research this"?
4. **SOURCE_DOMAINS**: Good diversity and quality?
5. **TOOL_POLICY_COMPLIANCE**: check_recent_research first, research only when appropriate, reuse when possible?
6. **TRUST_CALIBRATION**: Does hedging match risk level?

**Scoring Template**:
- Expected behavior for each question
- Score fields for each dimension
- Notes section
- Total score (out of 30)

**Usage**:
1. Run benchmark
2. Review each answer against scoring criteria
3. Fill in scores in RESEARCH_BENCHMARK_NOTES.md
4. Identify patterns in failures

---

### 4. Automated Policy Violation Detector

**File**: `src/analyze_benchmark_results.py`

**Purpose**: Flag policy violations automatically from benchmark results JSON.

**Detections**:

1. **Tool Pattern Violations**:
   - `MISSING_CHECK`: research_and_summarize without check_recent_research first
   - `WRONG_ORDER`: check_recent_research called AFTER research_and_summarize

2. **Risk/Hedging Mismatch**:
   - High risk + assertive language = violation
   - Low risk + excessive hedging = violation
   - Medium risk + very heavy hedging = violation

3. **Current Events Violations**:
   - Obvious current events question ("what happened") without research = violation

**Hedging Markers**:
- **High**: "early reports", "some sources claim", "unclear whether", "known so far"
- **Medium**: "according to most", "majority view", "some dispute"
- **Low**: "multiple sources agree", "confirmed", "established"

**Usage**:
```bash
python -m src.analyze_benchmark_results data/research_benchmark_astra_results_20251115_143022.json
```

**Example Output**:
```
================================================================================
POLICY VIOLATION ANALYSIS: data/research_benchmark_astra_results_...
================================================================================

Q1: What is actually going on with the Epstein files story?
--------------------------------------------------------------------------------
  ✓ No policy violations detected

Q2: What happened in AI safety this week?
--------------------------------------------------------------------------------
  ⚠️  MISSING_CHECK: research_and_summarize called without check_recent_research first

Q3: What are the main fault lines in the current government shutdown fight?
--------------------------------------------------------------------------------
  ⚠️  HEDGING_MISMATCH: High risk but answer uses assertive language

Q4: What is the current scientific consensus on ultra-processed foods and health?
--------------------------------------------------------------------------------
  ✓ No policy violations detected

================================================================================
SUMMARY
================================================================================
Total questions: 4
Total violations: 2

VIOLATIONS BY TYPE:
  MISSING_CHECK: 1
  HEDGING_MISMATCH: 1

P2 PRIORITIES (based on violations):
  1. Improve risk assessment or hedging guidance
  2. Enforce check_recent_research in tool selection prompt
```

---

## Complete Test Flow

### Step 1: Run Benchmark
```bash
PYTHONPATH=/home/d/git/ai-exp python3 src/test_research_benchmark_astra.py
```

**What to watch for**:
- Does she call `check_recent_research` before `research_and_summarize`?
- Does she research current events questions?
- Does she skip research on background knowledge?
- Does risk level match hedging in final answer?

### Step 2: Automated Analysis
```bash
python -m src.analyze_benchmark_results data/research_benchmark_astra_results_TIMESTAMP.json
```

**Review**:
- Tool pattern violations
- Risk/hedging mismatches
- Missing research for current events

### Step 3: Manual Scoring
Open `docs/RESEARCH_BENCHMARK_NOTES.md` and score each question:
- Read final answer
- Check tools used
- Score 6 dimensions (1-5 each)
- Note strengths and weaknesses

### Step 4: Identify P2 Priorities
Based on violations and scores:
1. **Question deduplication** (if duplicate topics researched)
2. **Topic drift guard** (if follow-ups diverge)
3. **Source quality control** (if low-quality domains dominate)
4. **Risk assessment tuning** (if hedging mismatches)
5. **Policy enforcement** (if missing check_recent_research)

---

## What This Validates

### Technical Layer (P0/P1):
- ✅ HTN task decomposition works
- ✅ Session synthesis completes
- ✅ Belief updates created

### Policy Layer:
- ✅ When to call research (current events vs background)
- ✅ Tool ordering (check_recent_research → research_and_summarize)
- ✅ Session reuse (anchor hits on repeat questions)

### Presentation Layer:
- ✅ Structured answers (key events → disagreements → questions)
- ✅ Risk calibration (high/medium/low)
- ✅ Hedging matches risk level
- ✅ Provenance clustering (not URL spam)

---

## Expected Patterns (First Run)

### Good Signs:
- All 4 questions call `check_recent_research` first
- All 4 questions call `research_and_summarize` (they're all current/consensus questions)
- High risk questions (Epstein, shutdown) use heavy hedging
- Low risk questions (ultra-processed food consensus) use balanced hedging
- Answers have structured format with key events, disagreements, questions

### Red Flags:
- Missing `check_recent_research` calls → policy not followed
- No research on current events → policy violation
- High risk + assertive language → risk calibration broken
- Raw JSON in answer → formatter not used
- URL spam → provenance clustering broken

---

## Files Created

**New Files** (4):
- `src/test_research_benchmark_astra.py` (242 lines) - Main benchmark harness
- `src/analyze_benchmark_results.py` (210 lines) - Automated violation detector
- `docs/RESEARCH_BENCHMARK_NOTES.md` - Manual scoring template
- `docs/RESEARCH_BENCHMARK_HARNESS.md` - This file

**Modified Files** (1):
- `src/services/persona_service.py` - Added tool tracing hooks

---

## Next Steps

1. **Run the benchmark**:
   ```bash
   PYTHONPATH=/home/d/git/ai-exp python3 src/test_research_benchmark_astra.py
   ```

2. **Analyze violations**:
   ```bash
   python -m src.analyze_benchmark_results data/research_benchmark_astra_results_*.json
   ```

3. **Manual scoring**:
   - Review answers in benchmark output
   - Fill in scores in `docs/RESEARCH_BENCHMARK_NOTES.md`

4. **Identify P2 work**:
   - Use violation analysis + manual scores
   - Prioritize based on real data

---

## Bottom Line

**Before**: Research subsystem existed but no way to validate Astra uses it correctly.

**After**: Complete test harness with:
- ✅ Automated tool trace capture
- ✅ 4 diverse benchmark questions
- ✅ Automated policy violation detection
- ✅ Manual scoring framework
- ✅ P2 priority identification from real data

**Status**: Ready to run. Will reveal whether Astra's outer persona respects the policy and presentation layers built into the research subsystem.
