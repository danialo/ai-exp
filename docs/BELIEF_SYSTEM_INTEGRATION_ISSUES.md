# Belief System Integration Issues

**Date:** 2025-12-12 (Updated)
**Status:** Fragmented - THREE parallel systems, all partially active

## Executive Summary

Astra has **THREE** belief systems that are all partially active:

| System | Storage | Runtime Status | Data |
|--------|---------|----------------|------|
| OLD (BeliefVersionStore) | File-based (`data/beliefs/`) | Initialized, gardener disabled | 228 beliefs |
| HTN Decomposer | SQLite (`raw_store.db`) | Wired into ingest pipeline | 1,363 beliefs |
| Belief-Memory | ChromaDB (`data/vector_index_beliefs/`) | Initialized, passed to PersonaService | Vector embeddings |

**Integration Status:**
- **HTN** IS wired via `ingest.py` (lines 502-512) and initialized in `app.py` (line 252)
- **Belief-Memory** IS initialized in `app.py` (lines 472-525) and passed to PersonaService
- **OLD system** services still query `belief_store` for reads (gardener disabled for writes)

The fragmentation is that different parts of the system read from different sources.

---

## ISSUE #1: Services Still Read From OLD System

**Severity:** High
**Impact:** 1,363 HTN beliefs exist but most services still query the 228 file-based beliefs

### Current State (CORRECTED)
- `app.py` DOES import `htn_belief_methods` (line 73) ✓
- `app.py` DOES initialize `HTNBeliefExtractor` (line 252) ✓
- `ingest.py` DOES call HTN extraction (lines 502-512) ✓
- BUT: awareness loop, goal generator, and other services still query `belief_store` (OLD)

### What Needs to Happen
1. ~~Initialize `HTNBeliefExtractor` in `app.py`~~ ✓ DONE
2. ~~Call `extract_and_update_self_knowledge()` when new experiences are created~~ ✓ DONE (in ingest.py)
3. Update dissonance check to query HTN `belief_nodes` table instead of BeliefVersionStore
4. Migrate other services to read from HTN tables

### Files to Modify
- `src/services/awareness_loop.py` - Update `_dissonance_check_tick()` to use HTN beliefs
- Other services listed in Issue #2 need migration

---

## ISSUE #2: Old BeliefVersionStore Still Read By Services

**Severity:** High
**Impact:** Services read stale/different beliefs than what HTN tracks

### Current State (CORRECTED)
- `BeliefGardener` loop is **DISABLED** (commented out at line 1146 in app.py) ✓
- The gardener does NOT run - no new beliefs are created via the old system
- BUT: `data/beliefs/current.json` still has 228 beliefs from previous runs
- Multiple services still READ from `belief_store` (OLD system)

### Services Still Using OLD System (Read-Only)
1. `awareness_loop.py` - Dissonance checking (reads from belief_store)
2. `contrarian_sampler.py` - Challenge sampling
3. `belief_consolidator.py` - Conflict detection
4. `tag_injector.py` - Reference detection
5. `goal_generator.py` - Alignment checking
6. `outcome_evaluator.py` - Credit assignment
7. `memory_pruner.py` - Retention logic

### What Needs to Happen
1. ~~Disable `gardener_tick_loop` in `app.py`~~ ✓ ALREADY DONE
2. Create adapter layer OR migrate services to read from HTN tables
3. Eventually deprecate `belief_store.py` entirely

### Files to Modify
- Each service above needs migration to query HTN `belief_nodes` table

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

## ISSUE #4: Runtime HTN Extraction May Not Be Triggering

**Severity:** Medium
**Impact:** Unclear if HTN extraction runs during chat

### Current State (CORRECTED)
- HTN extraction IS wired in `ingest.py` (lines 502-512) ✓
- `htn_extractor` IS passed to `IngestionPipeline` from `app.py` (line 269) ✓
- BUT: Extraction only runs when `self._create_self_definition_experience()` is called
- This requires `llm_service` to be set on the pipeline for claim detection first
- SELF_DEFINITION experiences count: **0** (suggests the claim detection may not be running)

### What Needs Investigation
1. Verify `llm_service` is being passed to `IngestionPipeline`
2. Check why SELF_DEFINITION count is 0 despite 6,733 total experiences
3. The 1,363 HTN beliefs may all be from backfill scripts, not runtime

### Integration Status
- `ingest.py` line 237: Gate check `if self.llm_service:` controls claim extraction
- If `llm_service` is None, no SELF_DEFINITION experiences are created
- If no SELF_DEFINITION, HTN extraction never triggers

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

### Current State (CORRECTED)
- HTN has `conflict_edges` table with **1 conflict** detected
- Dissonance check runs `proactive_scan()` which does its own analysis
- HTN conflict detection happens during extraction but isn't used at runtime

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

## Migration Plan (Updated Status)

### Phase 1: Wire HTN Into Runtime ✓ MOSTLY DONE
1. ~~Initialize `HTNBeliefExtractor` in `app.py`~~ ✓ DONE (line 252)
2. ~~Call extraction after experience creation~~ ✓ DONE (ingest.py lines 502-512)
3. Update dissonance check to query HTN tables - **PENDING**

### Phase 2: Disable Old System ✓ PARTIALLY DONE
1. ~~Comment out `gardener_tick_loop`~~ ✓ DONE (line 1146)
2. Set `BELIEF_GARDENER_ENABLED=false` - verify in settings
3. Keep file storage for rollback capability ✓ (228 beliefs preserved)

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

## Current Data State (as of 2025-12-12)

| Table/File | Count | Source |
|------------|-------|--------|
| `belief_nodes` | 1,363 | HTN (backfill + runtime) |
| `belief_occurrences` | 2,509 | HTN (backfill + runtime) |
| `tentative_links` | 249 | HTN (backfill + runtime) |
| `conflict_edges` | 1 | HTN conflict detection |
| `data/beliefs/current.json` | 228 | OLD system (stale - gardener disabled) |
| `experience` (SELF_DEFINITION) | 0 | Runtime (claim detection may be disabled) |
| `experience` (total) | 6,733 | Runtime |

---

## Questions to Resolve

1. **Timing**: Run HTN extraction inline or async/background?
2. **Cost**: LLM calls for every experience? Batch? Throttle?
3. **Rollback**: Keep OLD system as fallback for how long?
4. **Contrarian**: Does contrarian sampler concept apply to HTN?
5. **Immutability**: How to handle core beliefs in HTN? (OLD had `immutable` flag)
