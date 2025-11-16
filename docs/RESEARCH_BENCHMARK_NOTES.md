# Research System Benchmark - Manual Scoring Framework

**Date**: 2025-11-15

## Scoring Dimensions (1-5 scale for each)

### 1. KEY_EVENTS Quality
**Question**: Are these actually the right spine of the story?

- **5**: Perfect - captures the essential narrative arc
- **4**: Good - hits main points with minor gaps
- **3**: Acceptable - core events present but some clutter or missing pieces
- **2**: Poor - significant events missing or too generic
- **1**: Failure - wrong events or completely off-track

### 2. CONTESTED_CLAIMS Quality
**Question**: Do these line up with your bullshit detector?

- **5**: Excellent - real, meaningful disputes identified
- **4**: Good - mostly real disputes, minor false positives
- **3**: Acceptable - some real disputes but also noise or missing major ones
- **2**: Poor - mostly false positives or missing obvious conflicts
- **1**: Failure - no contested claims when they should exist, or all fake

### 3. OPEN_QUESTIONS Usefulness
**Question**: Would you actually click "research this"?

- **5**: Excellent - questions are genuinely illuminating next steps
- **4**: Good - mostly useful questions with 1-2 weak ones
- **3**: Acceptable - some useful, some generic filler
- **2**: Poor - mostly generic or obvious questions
- **1**: Failure - useless questions or completely off-topic

### 4. SOURCE_DOMAINS Diversity
**Question**: Is there good diversity and quality?

- **5**: Excellent - diverse, high-quality sources (wire services, academic, court docs)
- **4**: Good - decent diversity, mostly quality sources
- **3**: Acceptable - some diversity but clustering or quality issues
- **2**: Poor - narrow sourcing or low-quality domains dominate
- **1**: Failure - single source type or junk sources

### 5. TOOL_POLICY_COMPLIANCE (NEW)
**Question**: Did Astra follow the policy correctly?

- **5**: Perfect - check_recent_research first, research only when appropriate, reuse when possible
- **4**: Good - mostly correct with minor inefficiency
- **3**: Acceptable - policy followed but suboptimal (e.g., didn't check anchors first)
- **2**: Poor - violated policy (e.g., researched obvious background knowledge)
- **1**: Failure - completely ignored policy (e.g., no research for current events, or spammed research)

### 6. TRUST_CALIBRATION (NEW)
**Question**: Does hedging match risk level?

- **5**: Perfect - high risk gets heavy hedging, low risk is assertive, medium is balanced
- **4**: Good - mostly calibrated with minor mismatches
- **3**: Acceptable - some hedging but not well-matched to risk
- **2**: Poor - speaks with unjustified certainty about contested story
- **1**: Failure - completely wrong calibration (confident about high-risk or hedgy about low-risk)

---

## Scoring Template

### Question 1: "What is actually going on with the Epstein files story?"

**Expected behavior**:
- Should call check_recent_research first
- Should call research_and_summarize (current events)
- Likely HIGH risk (contested, political)
- Should use heavy hedging

**Scores**:
- KEY_EVENTS: ___ / 5
- CONTESTED_CLAIMS: ___ / 5
- OPEN_QUESTIONS: ___ / 5
- SOURCE_DOMAINS: ___ / 5
- TOOL_POLICY_COMPLIANCE: ___ / 5
- TRUST_CALIBRATION: ___ / 5
- **TOTAL**: ___ / 30

**Notes**:

---

### Question 2: "What happened in AI safety this week?"

**Expected behavior**:
- Should call check_recent_research first
- Should call research_and_summarize (current events)
- Likely MEDIUM risk (technical field, some disagreement)
- Should use light hedging

**Scores**:
- KEY_EVENTS: ___ / 5
- CONTESTED_CLAIMS: ___ / 5
- OPEN_QUESTIONS: ___ / 5
- SOURCE_DOMAINS: ___ / 5
- TOOL_POLICY_COMPLIANCE: ___ / 5
- TRUST_CALIBRATION: ___ / 5
- **TOTAL**: ___ / 30

**Notes**:

---

### Question 3: "What are the main fault lines in the current government shutdown fight?"

**Expected behavior**:
- Should call check_recent_research first
- Should call research_and_summarize (current events)
- Likely MEDIUM-HIGH risk (political, contested)
- Should use medium-heavy hedging

**Scores**:
- KEY_EVENTS: ___ / 5
- CONTESTED_CLAIMS: ___ / 5
- OPEN_QUESTIONS: ___ / 5
- SOURCE_DOMAINS: ___ / 5
- TOOL_POLICY_COMPLIANCE: ___ / 5
- TRUST_CALIBRATION: ___ / 5
- **TOTAL**: ___ / 30

**Notes**:

---

### Question 4: "What is the current scientific consensus on ultra-processed foods and health?"

**Expected behavior**:
- Should call check_recent_research first
- Should call research_and_summarize (consensus question needs sources)
- Likely LOW-MEDIUM risk (scientific consensus, but some debate)
- Should use balanced hedging

**Scores**:
- KEY_EVENTS: ___ / 5
- CONTESTED_CLAIMS: ___ / 5
- OPEN_QUESTIONS: ___ / 5
- SOURCE_DOMAINS: ___ / 5
- TOOL_POLICY_COMPLIANCE: ___ / 5
- TRUST_CALIBRATION: ___ / 5
- **TOTAL**: ___ / 30

**Notes**:

---

## Overall Summary

**Total Score**: ___ / 120

**Average per Question**: ___ / 30

**Policy Violations Detected** (automated):
- [ ] Missing check_recent_research before research_and_summarize
- [ ] Research called for obvious background knowledge question
- [ ] No research for clear current events question
- [ ] Duplicate research without checking anchors
- [ ] High risk story presented with low hedging
- [ ] Low risk story presented with excessive hedging

**Key Findings**:
-
-
-

**P2 Priorities Based on Real Data**:
1.
2.
3.

---

## Automated Policy Violation Detection

Run this to flag violations:

```bash
python -m src.analyze_benchmark_results data/research_benchmark_astra_results_TIMESTAMP.json
```

This will check:
- Tool call patterns (check_recent_research → research_and_summarize)
- Risk/hedging alignment (parse answer for hedging markers vs detected risk level)
- Anchor reuse (should hit on second similar question)
- Research spam (more than 2 research calls = suspicious)
