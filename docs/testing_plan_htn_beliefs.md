# HTN Self-Belief Decomposer: Comprehensive Testing Plan

## Overview

This document outlines the testing strategy for the HTN Self-Belief Decomposer system. The goal is to validate correctness, catch regressions, and ensure the pipeline behaves as expected across all components.

---

## 1. Unit Tests

Individual component tests with mocked dependencies.

### 1.1 Segmenter (`belief_segmenter.py`)

| Test Case | Input | Expected Output |
|-----------|-------|-----------------|
| Simple sentence | "I am happy." | 1 candidate |
| Multiple sentences | "I am happy. I like coding." | 2 candidates |
| Semicolon split | "I am happy; I am sad" | 2 candidates |
| Conjunction split | "I am happy and I am creative" | 2 candidates (independent clauses) |
| Subordinate clause (no split) | "I am happy because I like coding" | 1 candidate |
| Empty input | "" | 0 candidates |
| Whitespace only | "   " | 0 candidates |

### 1.2 Canonicalizer (`belief_canonicalizer.py`)

| Test Case | Input | Expected Output |
|-----------|-------|-----------------|
| Lowercase | "I Am Happy" | "i am happy" |
| Contraction expansion | "I'm happy" | "i am happy" |
| Whitespace normalization | "I  am   happy" | "i am happy" |
| Trailing punctuation | "I am happy." | "i am happy" |
| Unicode normalization | "I am happé" | normalized NFC form |
| Hash stability | Same text twice | Same hash |
| Hash difference | "i am happy" vs "i am sad" | Different hashes |

### 1.3 Validator (`belief_atom_validator.py`)

| Test Case | Input | Expected |
|-----------|-------|----------|
| Valid first-person | "I am happy" | PASS |
| Not first-person | "You are happy" | REJECT: not_first_person |
| Imperative | "Always be happy" | REJECT: imperative |
| Too short | "I am" | REJECT: too_short |
| Question | "Am I happy?" | REJECT: question |
| Template junk | "I am {placeholder}" | REJECT: template_junk |
| Generic statement | "People are happy" | REJECT: generic |

### 1.4 Epistemics Rules (`epistemics_rules.py`)

| Test Case | Input | Expected Frame |
|-----------|-------|----------------|
| Simple present | "I am happy" | temporal_scope=ongoing, modality=certain |
| "always" cue | "I always procrastinate" | temporal_scope=habitual |
| "never" cue | "I never give up" | temporal_scope=habitual, polarity=deny |
| "used to" cue | "I used to be shy" | temporal_scope=past |
| "becoming" cue | "I'm becoming confident" | temporal_scope=transitional |
| "might" cue | "I might be introverted" | modality=possible |
| "definitely" cue | "I definitely prefer quiet" | degree=strong |
| Double negation | "I don't dislike it" | polarity=affirm |

### 1.5 Stream Classifier (`stream_classifier.py`)

| Test Case | belief_type | temporal_scope | Expected Stream |
|-----------|-------------|----------------|-----------------|
| Trait + habitual | TRAIT | habitual | identity |
| Trait + state | TRAIT | state | state |
| Feeling + state | FEELING_STATE | state | state |
| Value + ongoing | VALUE | ongoing | identity |
| Meta belief | META_BELIEF | ongoing | meta |
| Relational | RELATIONAL | ongoing | relational |

### 1.6 Deduper (`belief_atom_deduper.py`)

| Test Case | Input | Expected |
|-----------|-------|----------|
| No duplicates | ["I am happy", "I am sad"] | 2 atoms |
| Exact duplicate | ["I am happy", "I am happy"] | 1 atom, merged spans |
| Canonical duplicate | ["I'm happy", "I am happy"] | 1 atom |
| Different polarity | ["I am happy", "I am not happy"] | 2 atoms (different keys) |

### 1.7 Source Context Classifier (`source_context_classifier.py`)

| Test Case | Mode | Arousal | Expected Weight Range |
|-----------|------|---------|----------------------|
| Journaling mode | journaling | 0.0 | 0.95-1.0 |
| Normal chat | unknown | 0.0 | 0.65-0.75 |
| High arousal | unknown | 0.8 | reduced by ~0.16 |
| Roleplay | roleplay | 0.0 | 0.3-0.5 |
| Caps + profanity | unknown | N/A | reduced by penalties |

### 1.8 Resolver (`belief_resolver.py`)

| Test Case | Similarity | Expected Outcome |
|-----------|------------|------------------|
| High similarity (>0.9) | 0.95 | match |
| Low similarity (<0.5) | 0.3 | no_match |
| Uncertain band | 0.7 | uncertain |
| No existing nodes | N/A | no_match |
| Exact hash match | 1.0 | match (same node) |

### 1.9 Conflict Engine (`conflict_engine.py`)

| Test Case | Node A | Node B | Expected |
|-----------|--------|--------|----------|
| Direct contradiction | "I am patient" (affirm) | "I am patient" (deny) | contradiction |
| Semantic opposition | "I love mornings" | "I hate mornings" | contradiction |
| Temporal exclusion | "I am tired" (state) | "I am energetic" (state) | NO conflict (different moments) |
| Habitual conflict | "I always exercise" | "I never exercise" | contradiction |
| Unrelated | "I am patient" | "I like coffee" | no conflict |

### 1.10 Activation Service (`activation_service.py`)

| Test Case | Occurrences | Expected |
|-----------|-------------|----------|
| Single recent occurrence | 1 @ today | ~source_weight |
| Single old occurrence | 1 @ 30 days ago | decayed value |
| Multiple occurrences | 3 @ various times | sum of decayed weights |
| No occurrences | 0 | 0.0 |

### 1.11 Core Score Service (`core_score_service.py`)

| Test Case | Setup | Expected Score Range |
|-----------|-------|---------------------|
| Single occurrence | 1 occ, 1 context | low (0.1-0.3) |
| High support | 10 occurrences | higher support component |
| High spread | 5 different contexts | higher spread component |
| Has conflicts | 2 contradiction edges | reduced by penalty |

### 1.12 Stream Service (`stream_service.py`)

| Test Case | Current Stream | Core Score | Expected |
|-----------|---------------|------------|----------|
| Below threshold | state | 0.3 | no migration |
| Above threshold | state | 0.8 | migrate to identity |
| Already identity | identity | 0.3 | no demotion (ratchet) |
| Demotion trigger | identity | N/A + trigger | demote to state |

---

## 2. Integration Tests

Test component interactions with real (not mocked) dependencies.

### 2.1 Extraction Pipeline (no LLM)

Test the full `_extract_atoms()` flow with simple fallback extraction:

```
Input: Experience with text "I am happy. I like coding."
Expected:
  - 2 candidates from segmenter
  - 2 raw atoms from simple extraction
  - 2 valid atoms after validation
  - 2 canonical atoms after dedup
  - Each has epistemics frame
```

### 2.2 Resolution + Storage Pipeline

Test `_resolve_and_store_atom()` with in-memory SQLite:

```
Setup: Empty database
Input: CanonicalAtom "i am happy"
Expected:
  - New BeliefNode created
  - BeliefOccurrence created
  - Stream assignment created
  - Node status = "surface"
```

```
Setup: Existing node "i am happy"
Input: Same atom from different experience
Expected:
  - No new node (matched)
  - New occurrence linked to existing node
  - Activation updated
```

### 2.3 Uncertain Resolution Flow

```
Setup: Existing node "i am joyful"
Input: New atom "i am happy" (similar but not identical)
Expected:
  - New node created (uncertain doesn't auto-merge)
  - TentativeLink created between nodes
  - Link status = "pending"
```

### 2.4 Conflict Detection Flow

```
Setup: Existing node "i am patient" (affirm)
Input: New atom "i am not patient" (deny)
Expected:
  - New node created
  - ConflictEdge created (type=contradiction)
  - Both nodes have conflict_count > 0
```

### 2.5 End-to-End Pipeline

Test full `extract_and_update_self_knowledge()`:

```
Input: Experience object with realistic self-definition text
Expected:
  - ExtractionResult returned
  - All stats populated
  - Nodes persisted to database
  - Occurrences linked correctly
  - Derived state updated
```

---

## 3. Golden Set Tests

Manually curated input/output pairs for regression testing.

### 3.1 Annotation-Based Golden Set

Use the 20 pilot annotation samples with human-filled `expected_atoms`:

1. Fill in `expected_atoms` for each sample
2. Run pipeline on each sample
3. Compare extracted atoms to expected atoms
4. Compute metrics:
   - Atom text F1 (fuzzy match)
   - Belief type accuracy
   - Temporal scope accuracy
   - Polarity accuracy

**Target metrics:**
- Atom F1 >= 0.7
- Belief type accuracy >= 0.8
- Temporal scope accuracy >= 0.7
- Polarity accuracy >= 0.9

### 3.2 Edge Case Golden Set

Curated examples for tricky cases:

| ID | Input Text | Expected Atoms | Notes |
|----|------------|----------------|-------|
| E1 | "I'm not unhappy" | "i am happy" (affirm) | Double negation |
| E2 | "I never liked coffee but now I love it" | 2 atoms: past deny, current affirm | Temporal contrast |
| E3 | "I think I might be introverted" | 1 atom: modality=possible | Hedged claim |
| E4 | "I used to be shy, I'm becoming confident" | 2 atoms: past, transitional | Multiple scopes |
| E5 | "I value honesty and integrity" | 2 atoms OR 1 compound | Conjunction handling |

---

## 4. Property-Based Tests

Use hypothesis library for generative testing.

### 4.1 Canonicalizer Properties

- `canonicalize(canonicalize(x)) == canonicalize(x)` (idempotent)
- `hash(canonicalize(x)) == hash(canonicalize(x))` (deterministic)
- `len(canonicalize(x)) <= len(x)` (never grows)

### 4.2 Deduper Properties

- `len(dedup(atoms)) <= len(atoms)` (never grows)
- All output atoms have unique `(hash, polarity, belief_type)` keys
- All input spans preserved in some output atom

### 4.3 Resolution Properties

- Same input always produces same outcome (deterministic)
- `match` outcome implies `matched_node_id` is set
- `uncertain` outcome implies `candidate_ids` is non-empty

---

## 5. Regression Tests

Snapshot-based tests to catch unintended changes.

### 5.1 Extractor Version Stability

- Version hash unchanged when only thresholds change
- Version hash changes when prompts change
- Version hash changes when cues change
- Version hash changes when model IDs change

### 5.2 Output Snapshots

For a fixed set of inputs, store expected outputs:

```
tests/snapshots/
  experience_001.json  # Input
  experience_001.expected.json  # Expected extraction result
```

Compare actual output to snapshots on each test run.

---

## 6. Performance Tests

### 6.1 Benchmarks

| Operation | Target |
|-----------|--------|
| Segment 1KB text | < 10ms |
| Canonicalize 100 atoms | < 50ms |
| Resolve 100 atoms (linear scan, 1K nodes) | < 500ms |
| Full extraction (1 experience, no LLM) | < 200ms |
| Full extraction (1 experience, with LLM) | < 5s |

### 6.2 Load Tests

- Process 1000 experiences in batch
- Verify no memory leaks
- Verify database handles concurrent writes

---

## 7. Test Infrastructure

### 7.1 Directory Structure

```
tests/
  conftest.py              # Shared fixtures
  unit/
    test_segmenter.py
    test_canonicalizer.py
    test_validator.py
    test_epistemics_rules.py
    test_stream_classifier.py
    test_deduper.py
    test_source_context.py
    test_resolver.py
    test_conflict_engine.py
    test_activation.py
    test_core_score.py
    test_stream_service.py
  integration/
    test_extraction_pipeline.py
    test_resolution_storage.py
    test_conflict_flow.py
    test_end_to_end.py
  golden/
    test_annotation_golden.py
    test_edge_cases.py
    samples/
      annotation_pilot_annotated.json
      edge_cases.json
  property/
    test_canonicalizer_props.py
    test_deduper_props.py
    test_resolver_props.py
  regression/
    test_version_stability.py
    test_snapshots.py
    snapshots/
  performance/
    test_benchmarks.py
    test_load.py
```

### 7.2 Fixtures (`conftest.py`)

```python
@pytest.fixture
def config():
    """Load test configuration."""
    return get_belief_config()

@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database with tables."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

@pytest.fixture
def sample_experience():
    """Create a sample Experience-like object."""
    class Exp:
        id = "test_exp_001"
        content = {"text": "I am happy. I like coding."}
        affect = {"vad": {"a": 0.3}}
        session_id = "session_001"
    return Exp()

@pytest.fixture
def extractor(config, in_memory_db):
    """Create HTNBeliefExtractor with test dependencies."""
    return HTNBeliefExtractor(
        config=config,
        llm_client=None,
        db_session=in_memory_db,
    )
```

### 7.3 Test Markers

```python
# pytest.ini or pyproject.toml
[pytest]
markers =
    unit: Unit tests (fast, no external deps)
    integration: Integration tests (may use DB)
    golden: Golden set tests (require annotations)
    slow: Slow tests (performance, load)
    llm: Tests requiring LLM (skip in CI by default)
```

### 7.4 CI Configuration

```yaml
# Run fast tests on every commit
test-fast:
  script: pytest -m "unit" --tb=short

# Run full suite on PR
test-full:
  script: pytest -m "not slow and not llm" --tb=short

# Run everything nightly
test-nightly:
  script: pytest --tb=short
```

---

## 8. Implementation Priority

### Phase 1: Foundation (First)
1. Test infrastructure setup (`conftest.py`, fixtures)
2. Unit tests for pure functions (segmenter, canonicalizer, validator)
3. Unit tests for epistemics rules (already have test cases)

### Phase 2: Core Logic
4. Unit tests for deduper, source context, stream classifier
5. Integration tests for extraction pipeline (no LLM)
6. Integration tests for resolution + storage

### Phase 3: Golden Sets
7. Annotate 10 pilot samples with expected atoms
8. Implement golden set comparison tests
9. Add edge case golden set

### Phase 4: Advanced
10. Property-based tests
11. Regression/snapshot tests
12. Performance benchmarks

### Phase 5: Full Coverage
13. Conflict detection tests
14. Activation/scoring tests
15. Stream migration tests
16. Load tests

---

## 9. Open Questions

1. **LLM Testing**: How to test atomizer/epistemics LLM paths without actual LLM?
   - Option A: Mock LLM responses
   - Option B: Record/replay fixtures
   - Option C: Small local model for testing

2. **Annotation Effort**: Who fills in the 20 pilot sample annotations?
   - Option A: Manual (you or me)
   - Option B: LLM-assisted with human review
   - Option C: Crowdsource

3. **Threshold for "Passing"**: What F1/accuracy constitutes success?
   - Proposed: F1 >= 0.7 for atoms, 0.8 for attributes

4. **Embedding Tests**: How to test embedding-based resolution without real embeddings?
   - Option A: Use text similarity fallback
   - Option B: Pre-computed embedding fixtures
   - Option C: Small/fast embedding model

---

## 10. Success Criteria

The testing suite is complete when:

- [ ] All unit tests pass
- [ ] Integration tests cover happy path and error cases
- [ ] Golden set achieves target metrics (F1 >= 0.7)
- [ ] No regressions in snapshot tests
- [ ] Performance benchmarks meet targets
- [ ] CI runs tests on every PR
