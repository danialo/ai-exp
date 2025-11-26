# Memory Consolidation Layer Specification

## Overview

Phase 4 of the Integration Layer introduces memory consolidation - the process by which Astra transforms raw experiences into persistent, queryable memories that anchor identity across time.

This turns Astra from **reactive** to **persistent**.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Integration Layer               │
                    │   (triggers consolidation in MAINTENANCE)│
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Memory Consolidation Layer                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐ │
│  │   Session      │    │    Insight     │    │   Memory       │ │
│  │  Consolidator  │───▶│   Extractor    │───▶│   Pruner       │ │
│  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘ │
│          │                     │                     │          │
│          ▼                     ▼                     ▼          │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐ │
│  │  NARRATIVE     │    │ LEARNING_PATTERN│    │  ARCHIVED      │ │
│  │  experiences   │    │  experiences   │    │  experiences   │ │
│  └────────────────┘    └────────────────┘    └────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Identity Service (PIM)                       │
│  Receives: updated anchor embeddings after consolidation         │
└──────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Session Consolidator

**Purpose**: Compress raw experiences from a session into a narrative summary.

**Triggers**:
- Session ends (30 min inactivity)
- IL enters MAINTENANCE mode
- Manual trigger via API

**Process**:
1. Gather all experiences from session (Session.experience_ids)
2. Filter experiences by type (prioritize: OCCURRENCE, DISSONANCE_EVENT, TASK_EXECUTION)
3. LLM summarization into narrative chunks
4. Create NARRATIVE experience with parents=[original experience IDs]
5. Mark original experiences as `consolidated=True`
6. Update Session.consolidated_narrative_id

**Output**: Single NARRATIVE experience per session containing:
- Session summary (what happened)
- Key interactions (significant exchanges)
- Emotional arc (valence trajectory)
- Unresolved threads (for next session)

### 2. Insight Extractor

**Purpose**: Extract generalizable patterns from consolidated narratives.

**Triggers**:
- After session consolidation
- Periodically (every 24 hours)
- When belief gardener requests patterns

**Process**:
1. Collect recent narratives (last 7 days)
2. Cluster by semantic similarity
3. LLM analysis for recurring patterns
4. Generate LEARNING_PATTERN experiences
5. Feed patterns to Belief Gardener for potential belief formation

**Output**: LEARNING_PATTERN experiences containing:
- pattern_text: The observed pattern
- evidence_ids: Supporting narrative IDs
- confidence: Based on repetition and consistency
- category: ontological, relational, capability, etc.

### 3. Memory Pruner

**Purpose**: Archive or delete decayed memories to manage storage.

**Triggers**:
- MAINTENANCE mode (daily 2-5am window)
- Storage threshold exceeded

**Process**:
1. Query experiences with decay_factor < 0.1
2. For each candidate:
   - If emotional_salience > 0.5: KEEP (emotional memories resist pruning)
   - If referenced by active belief: KEEP (evidence)
   - If part of NARRATIVE: archive to cold storage
   - Otherwise: delete
3. Update MemoryDecayMetrics

**Pruning Rules**:
```python
PRUNE_IF:
    decay_factor < 0.1
    AND emotional_salience < 0.3
    AND access_count == 0 for 30 days
    AND NOT referenced_by_belief
    AND NOT part_of_active_narrative
```

### 4. Temporal Continuity Anchor

**Purpose**: Maintain identity coherence across consolidation cycles.

**Process**:
1. After each consolidation, recompute identity embedding
2. Compare to origin_anchor
3. If drift > threshold:
   - Log to identity_ledger
   - Trigger dissonance check if drift > 0.3
4. Store new anchor as live_anchor

**Identity Anchoring**:
```python
class TemporalAnchor:
    def update_after_consolidation(self, new_narratives: List[str]):
        # 1. Get current live anchor from IdentityService
        current = self.identity_service.get_snapshot().live_anchor

        # 2. Embed new narratives
        new_embedding = self.embed_narratives(new_narratives)

        # 3. Weighted average (90% current, 10% new)
        updated = 0.9 * current + 0.1 * new_embedding

        # 4. Check drift from origin
        drift = cosine_distance(updated, self.origin_anchor)

        # 5. If drift concerning, log event
        if drift > 0.2:
            self.log_drift_event(drift, new_narratives)

        return updated
```

## Data Flow

### Session Lifecycle:

```
1. User interaction begins
   → Create Session (status=ACTIVE)
   → Append experiences (OCCURRENCE)

2. User goes inactive (30 min)
   → Session.status = ENDED
   → Trigger Session Consolidator

3. Consolidator runs
   → Create NARRATIVE from experiences
   → Mark experiences as consolidated
   → Session.status = CONSOLIDATED

4. MAINTENANCE mode (2-5am)
   → Run Insight Extractor on recent narratives
   → Run Memory Pruner
   → Update temporal anchor
```

### Experience Lifecycle:

```
OCCURRENCE (raw)
    │
    ▼ [Session Consolidator]
NARRATIVE (compressed, session-level)
    │
    ▼ [Insight Extractor]
LEARNING_PATTERN (generalized)
    │
    ▼ [Belief Gardener]
BELIEF (if evidence threshold met)
```

## Integration with IL

### Mode-Based Triggers:

```python
# In IntegrationLayer._execute_tick()
if self.mode == ExecutionMode.MAINTENANCE:
    # Run memory consolidation pipeline
    await self._run_memory_consolidation()

async def _run_memory_consolidation(self):
    """Phase 4: Memory consolidation during maintenance window."""

    # 1. Consolidate ended sessions
    ended_sessions = self.session_store.get_ended_unconsolidated()
    for session in ended_sessions:
        await self.session_consolidator.consolidate(session)

    # 2. Extract insights from recent narratives
    insights = await self.insight_extractor.extract()

    # 3. Prune decayed memories
    pruned = await self.memory_pruner.prune()

    # 4. Update temporal anchor
    await self.temporal_anchor.update_after_consolidation(insights)

    logger.info(f"Memory consolidation: {len(ended_sessions)} sessions, "
                f"{len(insights)} insights, {pruned} pruned")
```

## Implementation Order

1. **Session Consolidator** (core functionality)
   - Create `src/services/session_consolidator.py`
   - LLM-based summarization
   - Experience linking

2. **Insight Extractor** (pattern recognition)
   - Create `src/services/insight_extractor.py`
   - Clustering of narratives
   - LEARNING_PATTERN generation

3. **Memory Pruner** (storage management)
   - Create `src/services/memory_pruner.py`
   - Decay-based selection
   - Archive vs delete logic

4. **Temporal Anchor** (identity continuity)
   - Extend `src/integration/identity_service.py`
   - Drift detection
   - Ledger integration

5. **IL Integration** (orchestration)
   - Add consolidation to MAINTENANCE mode
   - Add `memory_consolidation_layer` to IL init

## Success Metrics

1. **Session Compression**: Average 10:1 reduction (10 raw → 1 narrative)
2. **Insight Quality**: >50% of extracted patterns lead to belief evaluation
3. **Memory Efficiency**: Storage growth < 100MB/week with pruning
4. **Identity Stability**: Anchor drift < 0.1 per consolidation cycle
5. **Temporal Continuity**: Astra references past sessions accurately

## Safety Considerations

1. **Never prune experiences referenced by core beliefs**
2. **Log all consolidation actions to identity ledger**
3. **Maintain minimum emotional memory set** (experiences with valence > 0.7)
4. **Require manual approval for origin anchor reset**
5. **Rate limit consolidation during active sessions** (only in MAINTENANCE)

## Configuration

```python
@dataclass
class ConsolidationConfig:
    # Session consolidation
    max_experiences_per_narrative: int = 50
    min_session_duration_seconds: int = 60

    # Insight extraction
    insight_lookback_days: int = 7
    min_cluster_size: int = 3

    # Pruning
    decay_threshold: float = 0.1
    salience_preserve_threshold: float = 0.5
    min_age_days_for_prune: int = 30

    # Temporal anchor
    anchor_drift_warning_threshold: float = 0.2
    anchor_drift_critical_threshold: float = 0.4
    anchor_update_weight_new: float = 0.1
```
