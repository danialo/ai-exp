# HTN Self-Belief Decomposer v1: Technical Design Specification

## 1. System Overview

The HTN Self-Belief Decomposer extracts structured belief atoms from Astra's self-definitional statements and maintains them in a graph of `BeliefNode` entities with occurrence evidence, conflict relationships, stream assignments, and time-decayed activation.

**Entry Point**: `HTNBeliefExtractor.extract_and_update_self_knowledge(experience)` in `src/services/htn_belief_methods.py`

**Input**: `Experience` row where `type='self_definition'`

**Output**: `ExtractionResult` containing all created/updated nodes, occurrences, tentative links, and conflicts.

---

## 2. Pipeline Architecture

```
Experience(type='self_definition')
         │
    ┌────▼────┐
    │ PHASE 1 │  source_context_classifier.py
    │ SOURCE  │  → SourceContext(mode, source_weight, context_id)
    └────┬────┘
         │
    ┌────▼────┐
    │ PHASE 2 │  belief_segmenter.py
    │ SEGMENT │  → List[ClaimCandidate(text, span)]
    └────┬────┘
         │
    ┌────▼────┐
    │ PHASE 3 │  belief_atomizer.py
    │ ATOMIZE │  → List[RawAtom(text, belief_type, polarity, confidence)]
    └────┬────┘
         │
    ┌────▼────┐
    │ PHASE 4 │  belief_atom_validator.py
    │VALIDATE │  → List[RawAtom] (filtered)
    └────┬────┘
         │
    ┌────▼────┐
    │ PHASE 5 │  belief_canonicalizer.py → belief_atom_deduper.py
    │ DEDUP   │  → DedupResult(List[CanonicalAtom])
    └────┬────┘
         │
    ┌────▼────┐
    │ PHASE 6 │  epistemics_rules.py → epistemics_llm.py (fallback)
    │EPISTEM. │  → EpistemicsResult(frame, confidence, polarity)
    └────┬────┘
         │
    ┌────▼────┐
    │ PHASE 7 │  htn_belief_embedder.py
    │ EMBED   │  → np.ndarray (1536-dim)
    └────┬────┘
         │
    ┌────▼────┐
    │ PHASE 8 │  belief_resolver.py + belief_match_verifier.py
    │ RESOLVE │  → ResolutionResult(outcome, matched_node_id, candidates)
    └────┬────┘
         │
    ┌────▼────┐
    │ PHASE 9 │  stream_classifier.py → stream_service.py
    │ STREAM  │  → StreamAssignment(primary, secondary, confidence)
    └────┬────┘
         │
    ┌────▼─────┐
    │ PHASE 10 │  Models: BeliefNode, BeliefOccurrence, TentativeLink
    │ STORAGE  │  tentative_link_service.py
    └────┬─────┘
         │
    ┌────▼─────┐
    │ PHASE 11 │  conflict_engine.py
    │ CONFLICT │  → List[ConflictEdge]
    └────┬─────┘
         │
    ┌────▼─────┐
    │ PHASE 12 │  activation_service.py → core_score_service.py
    │ SCORING  │  → updated node.activation, node.core_score, node.status
    └────┬─────┘
         │
    ┌────▼─────┐
    │ PHASE 13 │  stream_service.py (migration check)
    │ MIGRATE  │  → MigrationResult (state→identity if thresholds met)
    └──────────┘
```

---

## 3. Data Models

### 3.1 BeliefNode (`src/memory/models/belief_node.py`)

Canonical belief concept. One node per unique belief meaning.

| Column | Type | Description |
|--------|------|-------------|
| `belief_id` | `UUID` PK | Unique identifier |
| `canonical_text` | `TEXT` UNIQUE | Normalized belief text (e.g., "i am patient") |
| `canonical_hash` | `TEXT` UNIQUE | `SHA256(canonical_text)[:32]` |
| `belief_type` | `TEXT` | `TRAIT`, `VALUE`, `PREFERENCE`, `CAPABILITY_LIMIT`, `FEELING_STATE`, `META_BELIEF`, `RELATIONAL`, `BELIEF_ABOUT_SELF` |
| `polarity` | `TEXT` | `affirm` or `deny` |
| `embedding` | `BLOB` | Serialized `np.float32[1536]` |
| `activation` | `REAL` | Recency-weighted sum of occurrences |
| `core_score` | `REAL` | Centrality score `[0, 1]` |
| `status` | `TEXT` | `surface`, `developing`, `core` |
| `conflict_count` | `INT` | Open conflict edges |
| `created_at` | `TIMESTAMP` | First extraction |
| `last_reinforced_at` | `TIMESTAMP` | Last occurrence added |

**Uniqueness**: Nodes are unique by `canonical_hash`. Resolution checks embedding similarity; hash equality is a shortcut for exact match.

### 3.2 BeliefOccurrence (`src/memory/models/belief_occurrence.py`)

Evidence event linking a node to a source experience.

| Column | Type | Description |
|--------|------|-------------|
| `occurrence_id` | `UUID` PK | |
| `belief_id` | `UUID` FK | → `belief_nodes.belief_id` |
| `source_experience_id` | `TEXT` | `Experience.id` |
| `extractor_version` | `TEXT` | Version hash for rollback |
| `raw_text` | `TEXT` | Original text before canonicalization |
| `raw_span` | `JSON` | `{start, end}` or `[{start, end}, ...]` |
| `source_weight` | `REAL` | `[0, 1]` from source context |
| `atom_confidence` | `REAL` | LLM extraction confidence |
| `epistemic_frame` | `JSON` | `{temporal_scope, modality, degree}` |
| `epistemic_confidence` | `REAL` | |
| `match_confidence` | `REAL` | Resolution similarity |
| `context_id` | `TEXT` | For diversity scoring |
| `created_at` | `TIMESTAMP` | |
| `deleted_at` | `TIMESTAMP` | Soft delete for rollback |

**Constraint**: `UNIQUE(belief_id, source_experience_id, extractor_version)` — one occurrence per belief per source per extractor version.

### 3.3 TentativeLink (`src/memory/models/tentative_link.py`)

Uncertain identity resolution between two nodes.

| Column | Type | Description |
|--------|------|-------------|
| `link_id` | `UUID` PK | |
| `from_belief_id` | `UUID` FK | Smaller UUID (normalized order) |
| `to_belief_id` | `UUID` FK | Larger UUID |
| `confidence` | `REAL` | Merge confidence `[0, 1]` |
| `status` | `TEXT` | `pending`, `accepted`, `rejected` |
| `support_both` | `INT` | Uncertain matches involving both |
| `support_one` | `INT` | Definite matches to one side only |
| `last_support_at` | `TIMESTAMP` | Last evidence added |
| `signals` | `JSON` | Reasoning, similarity scores |
| `extractor_version` | `TEXT` | |
| `created_at` | `TIMESTAMP` | |
| `updated_at` | `TIMESTAMP` | |

**Constraint**: `UNIQUE(from_belief_id, to_belief_id)` — one link per ordered pair.

**Note**: `status='accepted'` does NOT auto-merge. Merge is a separate operation.

**Confidence formula**:
```
confidence = sigmoid(a * support_both - b * support_one - c * age_days)

Default params:
  a = 1.2 (support_both weight)
  b = 0.9 (support_one weight)
  c = 0.06 (age decay per day)
```

### 3.4 ConflictEdge (`src/memory/models/conflict_edge.py`)

Contradiction or tension between nodes.

| Column | Type | Description |
|--------|------|-------------|
| `edge_id` | `UUID` PK | |
| `node_a_id` | `UUID` FK | |
| `node_b_id` | `UUID` FK | |
| `conflict_type` | `TEXT` | `contradiction`, `tension` |
| `detection_method` | `TEXT` | `polarity_flip`, `semantic_opposition` |
| `evidence` | `JSON` | Similarity score, occurrence IDs |
| `status` | `TEXT` | `active`, `resolved` |
| `created_at` | `TIMESTAMP` | |
| `resolved_at` | `TIMESTAMP` | |
| `resolution_reason` | `TEXT` | |

**Constraint**: `UNIQUE(node_a_id, node_b_id)`

### 3.5 StreamAssignment (`src/memory/models/stream_assignment.py`)

Soft stream assignment with migration history.

| Column | Type | Description |
|--------|------|-------------|
| `assignment_id` | `UUID` PK | |
| `belief_id` | `UUID` FK UNIQUE | One assignment per node |
| `primary_stream` | `TEXT` | `identity`, `state`, `meta`, `relational` |
| `secondary_stream` | `TEXT` | Nullable fallback |
| `confidence` | `REAL` | |
| `migrated_from` | `TEXT` | Previous stream if migrated |
| `migrated_at` | `TIMESTAMP` | |
| `migration_history` | `JSON` | `[{from, to, at, reason}, ...]` |
| `created_at` | `TIMESTAMP` | |
| `updated_at` | `TIMESTAMP` | |

---

## 4. Service Specifications

### 4.1 SourceContextClassifier (`source_context_classifier.py`)

**Purpose**: Compute source weight and context ID for an experience.

**Input**: `Experience` object with optional metadata fields.

**Output**: `SourceContext(mode, source_weight, context_id)`

**Logic**:
1. Extract `mode` from `experience.metadata[mode_field]` or infer via heuristics
2. Look up base weight from `mode_weights` config
3. Apply penalties:
   - `caps_penalty` if >30% uppercase characters
   - `profanity_penalty` if profanity detected
   - `exclaim_penalty` if >10% exclamation density
4. VAD arousal penalty (if enabled): `weight -= arousal * arousal_weight`
5. Generate `context_id` = `conversation_id:mode` (for diversity tracking)

**Config** (`source_context` section):
```yaml
mode_weights:
  journaling: 1.0
  introspection: 0.95
  normal_chat: 0.8
  roleplay: 0.4
  heated: 0.5
  unknown: 0.7
heuristic_fallback:
  profanity_penalty: 0.15
  caps_penalty: 0.10
  exclaim_penalty: 0.05
```

### 4.2 BeliefSegmenter (`belief_segmenter.py`)

**Purpose**: Split text into claim candidates.

**Input**: Raw text string.

**Output**: `List[ClaimCandidate(text, span)]`

**Logic**:
1. Split on sentence-ending punctuation (`. ! ?`)
2. Split on semicolons (`;`)
3. Split on coordinating conjunctions when connecting independent clauses:
   - `and`, `but`, `however`, `although`, `though`, `yet`, `so`, `while`, `whereas`
4. Do NOT split on subordinating conjunctions (dependent clauses stay attached):
   - `because`, `since`, `unless`, `until`, `when`, `if`, `after`, `before`, `as`, `that`, `which`, `who`

**Span tracking**: Each candidate records `(start, end)` character positions for provenance.

### 4.3 BeliefAtomizer (`belief_atomizer.py`)

**Purpose**: LLM-based extraction of atomic belief statements.

**Input**: `List[ClaimCandidate]`

**Output**: `AtomizerResult(atoms: List[RawAtom], errors: List)`

**LLM Prompt** (from `config/prompts/atomizer_system_v1.txt`, `atomizer_user_v1.txt`):
- System: Instructions for extracting atomic first-person beliefs
- User: Formatted candidates for extraction

**Output Schema**:
```json
{
  "atoms": [
    {
      "atom_text": "i am patient",
      "belief_type": "TRAIT",
      "polarity": "affirm",
      "confidence": 0.9,
      "source_idx": 0
    }
  ]
}
```

**JSON Repair**: If LLM output is malformed, attempt repair via `repair_json_v1.txt` prompt (max 1 retry).

**Valid belief_types**: `TRAIT`, `PREFERENCE`, `VALUE`, `CAPABILITY_LIMIT`, `FEELING_STATE`, `META_BELIEF`, `RELATIONAL`, `BELIEF_ABOUT_SELF`

**Valid polarities**: `affirm`, `deny`

### 4.4 BeliefAtomValidator (`belief_atom_validator.py`)

**Purpose**: Filter invalid extractions.

**Input**: `List[RawAtom]`

**Output**: `ValidationResult(valid: List[RawAtom], invalid: List[InvalidAtom])`

**Rejection criteria**:
| Criterion | Example | Reason |
|-----------|---------|--------|
| `not_first_person` | "You are kind" | Must start with "I" |
| `imperative` | "Always be honest" | Command, not belief |
| `too_short` | "I am" | Below minimum length |
| `question` | "Am I happy?" | Question, not statement |
| `template_junk` | "I am {name}" | Contains placeholders |
| `generic` | "People are kind" | Not first-person |

### 4.5 BeliefCanonicalizer (`belief_canonicalizer.py`)

**Purpose**: Normalize text for deduplication.

**Input**: `RawAtom.atom_text`

**Output**: `CanonicalAtom(original_text, canonical_text, canonical_hash, ...)`

**Normalization steps**:
1. Lowercase
2. Expand contractions: `i'm` → `i am`, `don't` → `do not`, etc.
3. Strip trailing punctuation
4. Collapse multiple whitespace to single space
5. Trim leading/trailing whitespace
6. Unicode NFC normalization

**Hash**: `SHA256(canonical_text)[:32]` (hex digest prefix)

### 4.6 BeliefAtomDeduper (`belief_atom_deduper.py`)

**Purpose**: Deduplicate atoms within a single extraction.

**Input**: `List[RawAtom]` (validated)

**Output**: `DedupResult(deduped_atoms: List[CanonicalAtom], duplicates_removed: int)`

**Dedup key**: `(canonical_hash, polarity, belief_type)`

When duplicates found:
- Keep highest confidence atom
- Merge spans from all duplicates

### 4.7 EpistemicsRulesEngine (`epistemics_rules.py`)

**Purpose**: Deterministic epistemic frame extraction via cue matching.

**Input**: Original text string

**Output**: `EpistemicsResult(frame, confidence, signals, needs_llm_fallback, detected_polarity)`

**EpistemicFrame**:
```python
@dataclass
class EpistemicFrame:
    temporal_scope: str  # state, ongoing, habitual, transitional, past, unknown
    modality: str        # certain, likely, possible, unsure
    degree: float        # [0, 1] intensity
    conditional: str     # normalized condition if present
```

**Cue matching order**:
1. **Negation** → sets `polarity=deny`
2. **Modality** → caps certainty
3. **Temporal scope** → resolve conflicts by specificity
4. **Degree** → intensity modifier

**Cue categories** (from config):
```yaml
negation: [not, don't, never, cannot, won't, ...]
modality:
  possible: [might, maybe, perhaps, could be]
  likely: [i think, i suspect, probably]
  unsure: [unsure, not sure, uncertain]
past: [used to, formerly, previously, no longer]
transitional: [lately, recently, becoming, starting to]
habitual_strong: [always, never, every time, whenever]
habitual_soft: [usually, generally, tend to, inclined to]
ongoing: [still, continue to, remain]
state: [right now, at the moment, currently, today]
```

**IMPORTANT**: "never" is BOTH negation AND habitual. Sets `polarity=deny` AND `temporal_scope=habitual`.

**Conflict resolution**: When multiple temporal cues match, use `specificity_then_rightmost`:
```yaml
temporal_specificity:
  past: 6        # highest
  transitional: 5
  habitual: 4
  ongoing: 3
  state: 2
  unknown: 1     # lowest
```
Higher specificity wins. If equal, rightmost position wins.

**LLM fallback trigger**: If `confidence < llm_fallback_threshold` (default 0.6).

### 4.8 EpistemicsLLMFallback (`epistemics_llm.py`)

**Purpose**: LLM-based epistemics when rules are uncertain.

**Input**: Text string

**Output**: `EpistemicsResult` (same as rules engine)

**Prompt**: `config/prompts/epistemics_fallback_v1.txt`

**Used when**: Rules engine confidence < 0.6

### 4.9 HTNBeliefEmbedder (`htn_belief_embedder.py`)

**Purpose**: Compute embeddings for belief text.

**Input**: Canonical text string

**Output**: `np.ndarray` of shape `(1536,)` dtype `float32`

**Model**: `text-embedding-3-small` (OpenAI) or configured alternative

**Batch support**: `embed_batch(texts)` for efficiency

**Fallback**: If embedding disabled or fails, resolver uses text similarity (`levenshtein_ratio`).

### 4.10 BeliefResolver (`belief_resolver.py`)

**Purpose**: 3-way concept resolution with concurrency safety.

**Input**: `CanonicalAtom`, optional pre-computed embedding

**Output**: `ResolutionResult(outcome, match_confidence, matched_node_id, candidate_ids, ...)`

**Resolution outcomes**:
| Outcome | Condition | Action |
|---------|-----------|--------|
| `match` | `similarity >= 0.90` | Link occurrence to existing node |
| `no_match` | `similarity < 0.75` | Create new node |
| `uncertain` | `0.75 <= similarity < 0.90` | Create new node + TentativeLink |

**Algorithm**:
1. Query existing nodes by embedding similarity (top_k=10)
2. Check for exact hash match (shortcut)
3. If best similarity >= `match_threshold`: return `match`
4. If best similarity < `no_match_threshold`: return `no_match`
5. If in uncertain band and `verifier.enabled`:
   - Call LLM verifier to confirm/deny match
   - Verifier can upgrade to `match` or downgrade to `no_match`
6. If still uncertain: return `uncertain`

**Concurrency handling**: `unique_canonical_hash_retry` strategy
- On unique constraint violation during node creation, retry up to 3 times with 100ms delay
- On retry, re-query to find the node created by concurrent process

### 4.11 BeliefMatchVerifier (`belief_match_verifier.py`)

**Purpose**: LLM verification of uncertain matches.

**Input**: Two belief texts to compare

**Output**: `{is_same_belief: bool, confidence: float, reasoning: str}`

**Prompt**: `config/prompts/verifier_v1.txt`

**Trigger band**: Only called when similarity in `[0.75, 0.90]`

### 4.12 TentativeLinkService (`tentative_link_service.py`)

**Purpose**: Create and manage tentative links for uncertain resolution.

**Key methods**:
- `create_or_update(node_a, node_b, initial_confidence, signals)`: Creates link with normalized ID ordering
- `update_evidence(link_id, support_type)`: Increments `support_both` or `support_one`
- `check_auto_resolution(link)`: Auto-accept if confidence > 0.85, auto-reject if < 0.15

**ID normalization**: Links always stored with `from_belief_id < to_belief_id` (string comparison).

### 4.13 StreamClassifier (`stream_classifier.py`)

**Purpose**: Map (belief_type, temporal_scope) to stream.

**Input**: `belief_type`, `EpistemicFrame`

**Output**: `StreamClassification(primary_stream, secondary_stream, confidence)`

**Mapping** (from config):
```yaml
FEELING_STATE:
  state: {primary: state, secondary: identity}
  habitual: {primary: identity, secondary: state}
  ongoing: {primary: identity, secondary: state}
TRAIT:
  default: {primary: identity}
VALUE:
  default: {primary: identity}
META_BELIEF:
  default: {primary: meta, secondary: identity}
RELATIONAL:
  default: {primary: relational, secondary: identity}
```

### 4.14 StreamService (`stream_service.py`)

**Purpose**: Manage stream assignments and migration.

**Key methods**:
- `assign_initial(node, classification)`: Create initial assignment
- `get_assignment(belief_id)`: Retrieve current assignment
- `check_migration(node, assignment, core_result)`: Check for state→identity promotion

**Migration criteria** (state→identity):
```yaml
promote_state_to_identity:
  min_spread: 0.70       # sigmoid output of temporal spread
  min_diversity: 0.60    # sigmoid output of context diversity
  min_activation: 0.35   # minimum activation level
```

**Ratchet**: Once promoted to `identity`, cannot be demoted without explicit trigger:
```yaml
ratchet:
  enabled: true
  allow_demotion: false
  demotion_triggers: [explicit_obsolescence, sustained_conflict_low_activation]
```

### 4.15 ConflictEngine (`conflict_engine.py`)

**Purpose**: Detect contradictions and tension between beliefs.

**Input**: New node, embedding, occurrence

**Output**: `List[ConflictEdge]`

**Conflict types**:
| Type | Detection | Example |
|------|-----------|---------|
| `contradiction` | Same concept, opposite polarity | "i am patient" (affirm) vs "i am patient" (deny) |
| `tension` | High similarity, opposite polarity, different concepts | "i love mornings" vs "i hate mornings" |

**Algorithm**:
1. Query top_k (default 20) most similar nodes by embedding
2. For each candidate:
   - Skip if same node
   - Skip if temporal scope exclusion applies (see below)
   - Check for hard contradiction: same `canonical_hash`, opposite `polarity`
   - Check for tension: `similarity > embedding_threshold (0.88)` AND opposite polarity
3. Create `ConflictEdge` for each detected conflict

**Temporal scope exclusion**: `state` beliefs don't conflict with each other (different moments in time). Conflicts only between:
- `habitual` vs `habitual`
- `identity` vs `identity`
- `state` vs `habitual`/`identity` (state contradicts stable pattern)

### 4.16 ActivationService (`activation_service.py`)

**Purpose**: Compute recency-weighted activation.

**Formula**:
```
activation = Σ (source_weight × exp(-ln(2) × age_days / half_life))
```

**Half-lives by stream**:
```yaml
half_life_days:
  identity: 60   # slow decay - core beliefs persist
  state: 7       # fast decay - transient states fade
  meta: 30
  relational: 30
```

### 4.17 CoreScoreService (`core_score_service.py`)

**Purpose**: Compute belief centrality score.

**Formula**:
```
support = 1 - exp(-n_weighted / k_n)
spread = sigmoid((max_t - min_t - midpoint_days) / temperature_days)
diversity = sigmoid((n_contexts - midpoint_contexts) / temperature_contexts)

base = support × spread × diversity

conflict_penalty = weight × (conflicts_in_window / n_weighted)

core_score = max(0, base - conflict_penalty)
```

Where:
- `n_weighted` = sum of `source_weight × atom_confidence` across occurrences
- `max_t - min_t` = temporal spread in days
- `n_contexts` = count of distinct `context_id` values

**Config**:
```yaml
support:
  k_n: 10.0
spread:
  midpoint_days: 14.0
  temperature: 4.0
diversity:
  midpoint_contexts: 5.0
  temperature: 1.5
conflict_penalty:
  weight: 0.35
  recent_window_days: 30
```

**Status thresholds**:
```yaml
status_thresholds:
  developing: 0.3
  core: 0.6
```
- `core_score < 0.3` → `surface`
- `0.3 <= core_score < 0.6` → `developing`
- `core_score >= 0.6` → `core`

---

## 5. Configuration Reference

All hyperparameters live in `config/system_config.yaml`.

### 5.1 Extractor
```yaml
extractor:
  atomizer_model: "gemini-2.0-flash"
  epistemics_model: "gemini-2.0-flash"
  verifier_model: "gemini-2.0-flash"
  temperature: 0
  max_json_repair_attempts: 1
```

### 5.2 Embeddings
```yaml
embeddings:
  enabled: true
  model: "text-embedding-3-small"
  dimension: 1536
  batch_size: 32
  linear_scan_max_nodes: 50000
  fallback_to_text_similarity: true
  text_similarity_method: "levenshtein_ratio"
```

### 5.3 Resolution
```yaml
resolution:
  top_k: 10
  match_threshold: 0.90
  no_match_threshold: 0.75
  verifier:
    enabled: true
    trigger_band: [0.75, 0.90]
  tentative_link:
    auto_accept_threshold: 0.85
    auto_reject_threshold: 0.15
  tension:
    enabled: true
    embedding_threshold: 0.88
    top_k_conflict_check: 20
```

### 5.4 Concurrency
```yaml
concurrency:
  strategy: "unique_canonical_hash_retry"
  max_retries: 3
  retry_delay_ms: 100
```

---

## 6. Extractor Version Hash

**Purpose**: Enable rollback and A/B comparison of extraction versions.

**Computed from** (`src/utils/extractor_version.py`):
- Prompt template contents (atomizer, epistemics, verifier)
- Cue word lists
- Model IDs
- Key thresholds (match, no_match)

**Usage**:
- Stored in every `BeliefOccurrence.extractor_version`
- Rollback: `DELETE FROM belief_occurrences WHERE extractor_version = ?`
- Backfill: Skip experiences already processed by current version

---

## 7. Scripts

| Script | Purpose |
|--------|---------|
| `scripts/create_belief_tables.py` | Initialize SQLite schema |
| `scripts/backfill_self_definitions_to_beliefs.py` | Process existing self_definitions |
| `scripts/rollback_extractor_version.py` | Remove occurrences from specific version |
| `scripts/pilot_annotation_export.py` | Export samples for human review |
| `scripts/pilot_annotation_metrics.py` | Compute extraction F1 vs ground truth |

---

## 8. Integration Points

### 8.1 Ingest Pipeline (TODO)
Wire into `src/pipeline/ingest.py` to automatically extract beliefs from `self_definition` experiences:
```python
if experience.type == 'self_definition':
    extractor = HTNBeliefExtractor(llm_client=llm, db_session=db)
    result = extractor.extract_and_update_self_knowledge(experience)
```

### 8.2 Belief Surfacing (TODO)
Query beliefs by stream/status for prompt construction:
```sql
SELECT * FROM belief_nodes bn
JOIN stream_assignments sa ON bn.belief_id = sa.belief_id
WHERE sa.primary_stream = 'identity'
  AND bn.status = 'core'
ORDER BY bn.activation DESC
LIMIT 10;
```

### 8.3 TentativeLink Resolution (TODO)
Human review interface for uncertain matches:
- Display link pairs with confidence
- Accept → mark for merge
- Reject → close link

### 8.4 Conflict Resolution (TODO)
Rules or UI for resolving contradictions:
- Temporal precedence (newer wins)
- Evidence weight comparison
- Human override

---

## 9. Invariants

1. **One node per canonical meaning**: `canonical_hash` is unique
2. **One occurrence per (belief, source, version)**: Prevents duplicate evidence
3. **Ordered link pairs**: `from_belief_id < to_belief_id` always
4. **No auto-merge**: `TentativeLink.status='accepted'` requires explicit merge operation
5. **Ratchet**: `identity` stream assignment never auto-demotes
6. **Temporal exclusion**: `state` vs `state` never conflicts
