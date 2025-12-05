# Live End-to-End Testing Plan for HTN Self-Belief Decomposer

**Status:** TENTATIVE - Awaiting OpenAI quota restoration

---

## Test Environment Requirements

**Prerequisites:**
- OpenAI API key with available quota
- Astra server running on HTTP port 8000
- Clean or known database state
- Redis running (if required by Astra)

**Cost Estimate:** ~15-25 API calls per full test run

---

## Phase 1: Component-Level Live Tests

### 1.1 BeliefAtomizer (Real LLM Extraction)

| Test ID | Input Text | Expected Atoms | Assertions |
|---------|-----------|----------------|------------|
| ATOM-01 | "I am patient and I tend to overthink things" | 2 atoms: patience trait, overthinking trait | `len(atoms) == 2`, both have `belief_type`, `polarity` |
| ATOM-02 | "I've always loved hiking but I hate mornings" | 2 atoms: hiking preference (affirm), morning preference (negate) | Correct polarity detection |
| ATOM-03 | "My friend is very organized" | 0 atoms (third-person) | `len(atoms) == 0` |
| ATOM-04 | "I might be too cautious sometimes" | 1 atom with modality signal | `modality` field present |
| ATOM-05 | "I used to be shy but now I'm confident" | 2 atoms: past shy, current confident | Temporal markers preserved |

**Error Cases:**
- ATOM-ERR-01: Empty string → graceful empty response
- ATOM-ERR-02: Non-English text → handled without crash
- ATOM-ERR-03: Malformed LLM response → JSON repair attempted

### 1.2 EpistemicsLLM Fallback

| Test ID | Input Text | Rule Engine Result | Expected LLM Behavior |
|---------|-----------|-------------------|----------------------|
| EPIS-01 | "I'm kind of feeling like I might be changing" | Low confidence (<0.6) | LLM fallback triggered |
| EPIS-02 | "I always overthink" | High confidence (>0.6) | No LLM call (rules sufficient) |

**Assertion:** Track whether LLM was called via logging or call counter.

### 1.3 BeliefMatchVerifier

| Test ID | Scenario | Similarity Score | Expected |
|---------|----------|-----------------|----------|
| VERIF-01 | "i am patient" vs existing "i am patient" | >0.90 | Match without verifier |
| VERIF-02 | "i am patient" vs "i have patience" | 0.76-0.89 | Verifier called, should confirm match |
| VERIF-03 | "i am patient" vs "i am a nurse" | 0.76-0.89 | Verifier called, should reject |
| VERIF-04 | "i am happy" vs "i like pizza" | <0.75 | No match, no verifier |

---

## Phase 2: Full Pipeline Tests

### 2.1 Single Belief Recording

**Test PIPE-01:**
```
Input: POST /api/persona/chat
Body: {"message": "I've always been curious about how things work"}

Expected DB State:
- 1 new BeliefNode with canonical_text containing "curious"
- 1 BeliefOccurrence linked to source experience
- StreamAssignment to "identity" stream
- belief_type = "TRAIT"
- temporal_scope = "habitual" (due to "always")
```

**Verification Query:**
```sql
SELECT bn.canonical_text, bn.belief_type, bo.raw_text, sa.stream
FROM belief_nodes bn
JOIN belief_occurrences bo ON bn.belief_id = bo.belief_id
JOIN stream_assignments sa ON bn.belief_id = sa.belief_id
WHERE bn.canonical_text LIKE '%curious%'
ORDER BY bo.created_at DESC LIMIT 1;
```

### 2.2 Duplicate Detection

**Test PIPE-02:**
```
Step 1: Send "I am patient"
Step 2: Send "I'm a patient person"

Expected:
- Only 1 BeliefNode (deduplicated via canonical hash)
- 2 BeliefOccurrences pointing to same node
- occurrence_count or support metrics updated
```

### 2.3 Conflict Detection

**Test PIPE-03:**
```
Step 1: Send "I love mornings, they're my favorite time"
Step 2: Send "I absolutely hate waking up early"

Expected:
- 2 BeliefNodes created
- 1 ConflictEdge linking them
- conflict_type = "tension" or "hard_contradiction"
```

### 2.4 Tentative Link Creation

**Test PIPE-04:**
```
Step 1: Send "I think I might be introverted"
Step 2: Send "I prefer quiet environments"

Expected:
- 2 BeliefNodes
- 1 TentativeLink with status="pending"
- confidence score computed based on co-occurrence
```

---

## Phase 3: SELF_DEFINITION Recording Verification

**Test SELF-01: Experience Creation**

After each /api/persona/chat call with self-referential content:
```sql
SELECT id, type, content, created_at
FROM experiences
WHERE type = 'SELF_DEFINITION'
ORDER BY created_at DESC LIMIT 1;
```

**Assertions:**
- Experience exists with type='SELF_DEFINITION'
- content matches or references the belief extracted
- created_at is recent (within test window)

---

## Execution Protocol

1. **Pre-flight checks:**
   - Verify API key: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`
   - Verify Astra running: `curl http://localhost:8000/health`
   - Record initial DB state (belief counts)

2. **Execution order:**
   - Phase 1 tests (component isolation)
   - Phase 2 tests (pipeline integration)
   - Phase 3 verification (SELF_DEFINITION recording)

3. **Failure handling:**
   - Log full request/response for debugging
   - Continue to next test on non-critical failure
   - Abort on API quota exhaustion (429)

4. **Success criteria:**
   - All Phase 1 tests pass with expected outputs
   - All Phase 2 DB state assertions verified
   - SELF_DEFINITION experiences recorded for each self-referential input

---

## Test File Location

```
tests/beliefs/live/
├── test_atomizer_live.py
├── test_epistemics_live.py
├── test_verifier_live.py
├── test_pipeline_live.py
└── conftest.py  # Real API client, no mocks
```

**pytest marker:** `@pytest.mark.live` - skipped by default, run with `pytest -m live`

---

## Notes

- This plan was developed after 164 unit tests were created with mocks
- Unit tests for pure logic (canonicalizer, thresholds, stream routing) remain valid
- This plan focuses on components requiring real LLM calls
- Execute once OpenAI quota is restored
