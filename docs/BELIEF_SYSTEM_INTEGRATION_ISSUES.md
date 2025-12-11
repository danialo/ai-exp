# Belief System Integration Issues

**Date:** 2025-12-06
**Status:** Critical - Two parallel systems running, new system dormant

## Executive Summary

Astra has TWO completely separate belief systems:

| System | Storage | Runtime Status | Data |
|--------|---------|----------------|------|
| OLD (BeliefVersionStore) | File-based (`data/beliefs/`) | ACTIVE | ~5 beliefs |
| NEW (HTN Decomposer) | SQLite (`raw_store.db`) | DORMANT | 354 beliefs |

The NEW system was designed, implemented, tested, and backfilled - but **never wired into runtime**. The OLD system continues to run every 3 minutes, processing experiences but storing to a different location.

---

## ISSUE #1: HTN System Not Wired Into Runtime

**Severity:** Critical
**Impact:** 354 carefully extracted beliefs are unused; old system runs with ~5 beliefs

### Current State
- `app.py` imports and initializes `belief_store` (OLD system)
- `app.py` does NOT import `htn_belief_methods` (NEW system)
- HTN tables have data from backfill but nothing reads them at runtime
- Awareness loop dissonance check only sees 5 beliefs (from OLD system)

### What Needs to Happen
1. Initialize `HTNBeliefExtractor` in `app.py`
2. Call `extract_and_update_self_knowledge()` when new experiences are created
3. Update dissonance check to query HTN `belief_nodes` table instead of BeliefVersionStore

### Files to Modify
- `app.py` - Add HTN initialization and wiring
- `src/services/awareness_loop.py` - Update `_dissonance_check_tick()` to use HTN beliefs
- `src/services/persona_service.py` - Call HTN extraction after experience creation

---

## ISSUE #2: Old BeliefVersionStore Still Active

**Severity:** High
**Impact:** Wasted compute, potential confusion, duplicate data paths

### Current State
- `BeliefGardener` runs every 3 minutes via `gardener_tick_loop`
- Processes 200 SELF_DEFINITION experiences each scan
- Creates patterns, attempts belief creation
- Stores to `data/beliefs/current.json` (file-based)
- Multiple services read from this store (see full list below)

### Services Using OLD System
1. `belief_gardener.py` - Creates/updates beliefs
2. `awareness_loop.py` - Dissonance checking (reads 10 beliefs)
3. `contrarian_sampler.py` - Challenge sampling
4. `belief_consolidator.py` - Conflict detection
5. `tag_injector.py` - Reference detection
6. `goal_generator.py` - Alignment checking
7. `outcome_evaluator.py` - Credit assignment
8. `memory_pruner.py` - Retention logic

### What Needs to Happen
1. Disable `gardener_tick_loop` in `app.py`
2. Create adapter layer OR migrate services to use HTN tables
3. Eventually deprecate `belief_store.py` entirely

### Files to Modify
- `app.py` - Disable gardener loop, update service initialization
- Each service above needs migration or adapter

---

## ISSUE #3: Dissonance Check Hardcoded to 10 Beliefs

**Severity:** Medium
**Impact:** Only 0.3% of beliefs checked per cycle

### Current State
```python
# awareness_loop.py line 746-748
active_beliefs = [
    b.statement for b in list(current_beliefs.values())[:10]
]
```

### What Needs to Happen
1. Remove hardcoded limit OR
2. Implement random sampling OR
3. Implement rotation through beliefs

### Files to Modify
- `src/services/awareness_loop.py` - Line 746-748

---

## ISSUE #4: No Runtime Belief Extraction

**Severity:** Critical
**Impact:** New experiences never create HTN beliefs during chat

### Current State
- HTN extraction only happens via manual backfill script
- Chat creates experiences but doesn't extract beliefs
- Beliefs become stale as new experiences accumulate

### What Needs to Happen
1. Hook HTN extraction into experience creation flow
2. Consider async/background extraction to not block responses
3. May need throttling to avoid LLM cost explosion

### Integration Points
- `src/services/persona_service.py` - After `raw_store.append_experience()`
- OR `src/memory/raw_store.py` - Add post-commit hook
- OR Background task similar to gardener loop

---

## ISSUE #5: Duplicate Pattern Detection Logic

**Severity:** Medium
**Impact:** Two systems doing similar work differently

### Current State
- OLD: `BeliefGardener.PatternDetector` uses regex + Jaccard + embeddings
- NEW: `HTNBeliefExtractor` uses LLM atomization + canonical dedup

### What Needs to Happen
1. Decide which approach is authoritative
2. Likely: Use HTN (more sophisticated) and disable gardener pattern detection
3. May want to keep gardener's embedding-based soft merge as quality check

---

## ISSUE #6: Conflicting Belief Schemas

**Severity:** Medium
**Impact:** Can't easily migrate data between systems

### OLD System Schema (BeliefVersion)
```python
belief_id: str
ver: int
statement: str
state: BeliefState  # TENTATIVE/ASSERTED/DEPRECATED
confidence: float
evidence_refs: List[str]
belief_type: str
immutable: bool
stability: float
is_core: bool
```

### NEW System Schema (BeliefNode)
```python
belief_id: UUID
canonical_text: str
canonical_hash: str
belief_type: str
polarity: str  # affirm/deny
activation: float
core_score: float
status: str  # surface/developing/core/orphaned
embedding: bytes
```

### Key Differences
- OLD has `confidence` + `state`, NEW has `activation` + `core_score` + `status`
- OLD has `evidence_refs` list, NEW has separate `belief_occurrences` table
- OLD has `immutable` flag, NEW has no equivalent
- NEW has `polarity`, OLD doesn't track this explicitly
- NEW has embeddings stored, OLD computes on-demand

### What Needs to Happen
1. Create migration script to convert OLD beliefs to NEW format
2. Map confidence → activation (or recompute)
3. Map state → status
4. Extract polarity from statement text
5. Compute embeddings for migrated beliefs

---

## ISSUE #7: No Unified Belief Query API

**Severity:** Medium
**Impact:** Services query beliefs differently, inconsistent results

### Current State
- OLD: `belief_store.get_current()` returns `Dict[str, BeliefVersion]`
- NEW: Direct SQLModel queries on `belief_nodes` table

### What Needs to Happen
1. Create unified query interface
2. Options:
   a. Adapter pattern: Make HTN tables queryable via BeliefStore interface
   b. New service: `BeliefQueryService` that abstracts storage
   c. Full migration: Update all callers to use HTN directly

---

## ISSUE #8: Dissonance Check Not Using HTN Conflict Data

**Severity:** Medium
**Impact:** Detected conflicts in HTN ignored at runtime

### Current State
- HTN has `conflict_edges` table with 0 conflicts (currently)
- Dissonance check runs `proactive_scan()` which does its own analysis
- HTN conflict detection happens during extraction but isn't used

### What Needs to Happen
1. Dissonance check should query `conflict_edges` table
2. Or integrate `ConflictEngine` into awareness loop
3. Publish `DissonanceSignal` when conflicts detected in HTN

---

## ISSUE #9: SelfKnowledgeIndex Integration Disabled

**Severity:** Low
**Impact:** TASK 10.2 integration incomplete

### Current State
```python
# htn_belief_methods.py
self.self_knowledge_index = None  # Disabled - may not be available
```

### What Needs to Happen
1. Determine if SelfKnowledgeIndex is still needed
2. If yes, enable and wire up properly
3. If no, remove the integration code

---

## ISSUE #10: Stream Assignments Not Used

**Severity:** Low
**Impact:** Stream classification computed but not consumed

### Current State
- HTN classifies beliefs into streams: identity, state, meta, relational
- `stream_assignments` table populated
- Nothing reads or acts on stream data

### What Needs to Happen
1. Define what streams should influence (prompt context? priority?)
2. Integrate stream data into belief retrieval
3. Or remove if not needed

---

## Migration Plan (Recommended Order)

### Phase 1: Wire HTN Into Runtime
1. Initialize `HTNBeliefExtractor` in `app.py`
2. Call extraction after experience creation (async/background)
3. Update dissonance check to query HTN tables

### Phase 2: Disable Old System
1. Comment out `gardener_tick_loop`
2. Set `BELIEF_GARDENER_ENABLED=false`
3. Keep file storage for rollback capability

### Phase 3: Migrate Dependent Services
1. Create adapter or migrate each service:
   - awareness_loop.py
   - contrarian_sampler.py
   - belief_consolidator.py
   - tag_injector.py
   - goal_generator.py
   - outcome_evaluator.py
   - memory_pruner.py

### Phase 4: Data Migration
1. Create migration script for OLD → NEW beliefs
2. Run one-time migration
3. Verify data integrity

### Phase 5: Cleanup
1. Remove old BeliefStore code (or deprecate)
2. Remove gardener pattern detection
3. Update documentation

---

## File Inventory

### OLD System Files (To Be Deprecated)
```
src/services/belief_store.py           # Core store
src/services/belief_gardener.py        # Pattern detection + lifecycle
data/beliefs/                          # File storage
  current.json
  index.json
  log-*.ndjson.gz
  backups/
```

### NEW System Files (To Be Activated)
```
src/services/htn_belief_methods.py     # Orchestration
src/services/belief_resolver.py        # Resolution
src/services/belief_atomizer.py        # LLM extraction
src/services/belief_canonicalizer.py   # Normalization
src/services/conflict_engine.py        # Conflict detection
src/services/htn_belief_embedder.py    # Embeddings
src/services/activation_service.py     # Scoring
src/services/core_score_service.py     # Centrality
src/services/stream_service.py         # Stream classification
src/services/tentative_link_service.py # Uncertain matches
src/memory/models/                     # SQLModel definitions
scripts/backfill_*.py                  # Migration tools
```

### Shared Files (Need Updates)
```
app.py                                 # Main wiring
src/services/awareness_loop.py         # Dissonance check
src/services/persona_service.py        # Experience creation
```

---

## Current Data State

| Table/File | Count | Source |
|------------|-------|--------|
| `belief_nodes` | 354 | HTN backfill |
| `belief_occurrences` | 1,255 | HTN backfill |
| `tentative_links` | 26 | HTN backfill |
| `conflict_edges` | 0 | HTN backfill |
| `data/beliefs/current.json` | ~5 | OLD runtime |
| `experience` (SELF_DEFINITION) | 3,378 | Runtime |

---

## Questions to Resolve

1. **Timing**: Run HTN extraction inline or async/background?
2. **Cost**: LLM calls for every experience? Batch? Throttle?
3. **Rollback**: Keep OLD system as fallback for how long?
4. **Contrarian**: Does contrarian sampler concept apply to HTN?
5. **Immutability**: How to handle core beliefs in HTN? (OLD had `immutable` flag)
