# IMPLEMENTATION PLAN: HTN Self-Belief Decomposer v1 (Agent-Executable)

## Goal

Convert existing `Experience(type='self_definition')` rows into a structured self-knowledge system with:

* canonical **BeliefNodes** (concepts)
* **BeliefOccurrences** (evidence events, 1 per source experience per belief per extractor_version)
* **TentativeLinks** (uncertain merges)
* **ConflictEdges** (v1: syntactic contradiction + tension candidates)
* **StreamAssignments** (soft, with sticky migration)
* full **eval event logging** for debugging regressions

This plan is **non-destructive**: it adds new tables and scripts, does not mutate existing `Experience` rows.

---

## Hard Constraints (agent rules)

1. **Do not redesign architecture.** Implement this plan as-written.
2. **All hyperparameters live in `config/system_config.yaml`.** Nothing domain-y gets hardcoded.
3. **LLM calls only in two places:** atomization, and epistemics fallback when deterministic confidence is low.
4. **Every LLM output must be**:

   * schema validated
   * tagged with `extractor_version`
   * logged via eval events
5. **No curated exclusion ontology in v1.** Conflict detection is scoped.
6. **Uncertain matches never merge immediately.** They create TentativeLinks.
7. **Concurrency must be safe.** Use uniqueness constraints and retry logic.
8. **Backfill must be resumable** and skip work already done for the same extractor_version.
9. **Rollback must exist** as a script that cleans occurrences + recomputes node state.

---

## Architecture Summary (what you are building)

* **BeliefNode**: canonical belief concept ("I value honesty")
* **BeliefOccurrence**: an extracted event tying a BeliefNode to a source Experience id and extractor_version
* **TentativeLink**: uncertain identity resolution between two BeliefNodes
* **ConflictEdge**: contradiction/tension relationship between two BeliefNodes
* **StreamAssignment**: identity/state/meta/relational soft assignment on a BeliefNode
* **SelfKnowledgeIndex**: cache mapping topics -> experience ids (already exists). We will update it and include remove support (either already added or must be added).

---

## Config (must be created first)

### TASK 0.1: `config/system_config.yaml`

**Deliverable:** config file with these sections and keys:

```yaml
extractor:
  atomizer_model: "..."
  epistemics_model: "..."
  temperature: 0
  max_json_repair_attempts: 1

source_context:
  # mode->base weight mapping, mode can come from Experience metadata if available
  mode_weights:
    journaling: 1.0
    normal_chat: 0.8
    roleplay: 0.4
    heated: 0.5
    unknown: 0.7
  vad:
    enabled: true
    arousal_weight: 0.35   # source_weight = clamp(mode_weight - arousal_weight*arousal, 0, 1)
  heuristic_fallback:
    enabled: true
    profanity_penalty: 0.15
    caps_penalty: 0.10
    exclaim_penalty: 0.05

context:
  # v1 choice: context_id = conversation_id + ":" + mode (fallback to experience_id if conversation_id missing)
  strategy: "conversation_mode"
  fallback: "experience_id"

atomizer:
  json_schema: "v1"   # internal tag
  repair_prompt_id: "repair_json_v1"   # points to a constant prompt template

epistemics:
  llm_fallback_threshold: 0.6
  cue_conflict_resolution: "specificity_then_rightmost"
  modality_caps:
    possible: 0.4
    likely: 0.6
    unsure: 0.2
  temporal_specificity:
    past: 6
    transitional: 5
    habitual: 4
    ongoing: 3
    state: 2
    unknown: 1
  cues:
    modality:
      possible: ["might", "maybe", "perhaps", "possibly", "could be"]
      likely: ["i think", "i suspect", "i guess", "i assume", "probably"]
      unsure: ["unsure", "not sure", "uncertain"]
    past: ["used to", "formerly", "previously", "back then", "in the past"]
    transitional: ["lately", "recently", "these days", "of late", "increasingly", "more and more", "getting", "becoming", "starting to"]
    habitual_strong: ["always", "never", "every time", "whenever"]
    habitual_soft: ["usually", "generally", "typically", "normally", "often", "frequently", "tend to", "inclined to", "prone to"]
    ongoing: ["still", "continue to", "keep", "remain", "stay"]
    state: ["right now", "at the moment", "currently", "today", "tonight"]
  degree:
    strong: ["extremely", "very", "really"]
    weak: ["somewhat", "slightly", "a bit"]

streams:
  # mapping rules: type x temporal_scope -> primary/secondary
  mapping:
    FEELING_STATE:
      state: {primary: "state", secondary: "identity", confidence: 0.65}
      habitual: {primary: "identity", secondary: "state", confidence: 0.80}
      ongoing: {primary: "identity", secondary: "state", confidence: 0.75}
      transitional: {primary: "state", secondary: "identity", confidence: 0.70}
    TRAIT:
      any: {primary: "identity", secondary: null, confidence: 0.85}
    PREFERENCE:
      any: {primary: "identity", secondary: null, confidence: 0.80}
    VALUE:
      any: {primary: "identity", secondary: null, confidence: 0.90}
    CAPABILITY_LIMIT:
      any: {primary: "identity", secondary: null, confidence: 0.80}
    META_BELIEF:
      any: {primary: "meta", secondary: "identity", confidence: 0.80}
    RELATIONAL:
      any: {primary: "relational", secondary: "identity", confidence: 0.75}

resolution:
  use_embeddings: true
  embedding_model: "..."   # whatever your stack supports
  top_k: 10
  match_threshold: 0.90
  no_match_threshold: 0.75
  verifier_enabled: true
  verifier_threshold: 0.60
  tentative_link:
    auto_accept: 0.85
    auto_reject: 0.15
    a: 1.2
    b: 0.9
    c: 0.06
    age_definition: "days_since_last_support_else_created"
  tension:
    enabled: true
    threshold: 0.88  # embedding sim threshold for tension candidates

concurrency:
  strategy: "unique_canonical_hash_retry"
  max_retries: 3

scoring:
  half_life_days:
    identity: 60
    state: 7
    meta: 30
    relational: 30
  support_k_n: 10.0
  spread_midpoint_days: 14.0
  spread_temp: 4.0
  diversity_midpoint: 5.0
  diversity_temp: 1.5
  conflict_penalty:
    enabled: true
    recent_window_days: 30
    weight: 0.35

migration:
  promote_state_to_identity:
    strategy: "absolute"
    min_spread: 0.70
    min_diversity: 0.60
    min_activation: 0.35
  ratchet:
    enabled: true
    demotion_requires: ["explicit_obsolescence_or_sustained_conflict"]

backfill:
  batch_size: 50
  checkpoint_every: 10
  resume_strategy: "skip_if_occurrence_exists_for_extractor_version"

logging:
  eval_events_enabled: true
  eval_events_path: "data/eval_events"
```

**Acceptance:** New code reads config; missing keys fail fast with clear errors.

---

## Extractor Versioning

### TASK 0.2: `src/utils/extractor_version.py`

**Deliverable:** function `get_extractor_version()` returning a stable string hash of:

* atomizer prompt template id/version
* epistemics rules version (table hash)
* model ids
* config version hash (or subset relevant)
* code constant `EXTRACTOR_CODE_VERSION`

**Acceptance:** Changing prompts or rules changes extractor_version.

---

## Pilot Annotation Harness (required, not optional)

### TASK 0.3: Annotation export + metrics scripts

**Deliverables:**

* `scripts/pilot_annotation_export.py`

  * exports 20 self_def Experiences into JSON for manual labeling
  * includes experience_id and text
* `scripts/pilot_annotation_metrics.py`

  * reads annotator JSON outputs
  * computes pairwise F1 per category (atoms/types/epistemics)
  * computes Krippendorff alpha if feasible (otherwise leave TODO with clear stub)

**Acceptance:** Running metrics prints agreement summary.

---

## Database Models (SQLModel)

### TASK 1.1: Create models + migrations

**Deliverables:**

* `src/memory/models/belief_node.py`
* `src/memory/models/belief_occurrence.py`
* `src/memory/models/tentative_link.py`
* `src/memory/models/conflict_edge.py`
* `src/memory/models/stream_assignment.py`

#### REQUIRED SCHEMA DETAILS (do not improvise)

**BeliefNode**

* `belief_id` UUID PK
* `canonical_text` TEXT (indexed)
* `canonical_hash` TEXT UNIQUE (indexed, enforce uniqueness)
* `belief_type` TEXT
* `polarity` TEXT
* `created_at` datetime
* `last_reinforced_at` datetime
* `activation` float default 0
* `core_score` float default 0
* `status` TEXT default "surface"
* `embedding` optional BLOB/TEXT (whatever your stack uses)

**BeliefOccurrence**

* `occurrence_id` UUID PK
* `belief_id` FK
* `source_experience_id` TEXT (indexed)
* `extractor_version` TEXT (indexed)
* `raw_text` TEXT
* `raw_span` optional JSON
* `source_weight` float
* `atom_confidence` float
* `epistemic_frame` JSON
* `epistemic_confidence` float
* `match_confidence` float
* `context_id` TEXT (indexed)
* `created_at` datetime
* UNIQUE constraint: `(belief_id, source_experience_id, extractor_version)` to enforce idempotency

**TentativeLink**

* `link_id` UUID PK
* `from_belief_id` FK
* `to_belief_id` FK
* `confidence` float
* `status` TEXT
* `support_both` int default 0
* `support_one` int default 0
* `last_support_at` datetime nullable
* `signals` JSON
* `extractor_version` TEXT
* `created_at`, `updated_at`
* UNIQUE constraint: `(from_belief_id, to_belief_id)` (store normalized ordering to avoid duplicates)

**ConflictEdge**

* `edge_id` UUID PK
* `from_belief_id` FK
* `to_belief_id` FK
* `type` TEXT (contradiction|tension)
* `status` TEXT (unresolved|tolerated|resolved)
* `evidence_occurrence_ids` JSON
* `created_at`, `updated_at`
* UNIQUE constraint: `(from_belief_id, to_belief_id, type)`

**StreamAssignment**

* `belief_id` FK PK
* `primary_stream` TEXT
* `secondary_stream` TEXT nullable
* `confidence` float
* `updated_at` datetime

**Acceptance:** migrations run, uniqueness prevents duplicates.

---

## Eval Events Logging (flight recorder)

### TASK 2.1: `src/services/eval_event_logger.py`

**Deliverable:** JSONL logger writing to `config.logging.eval_events_path/YYYY-MM-DD.jsonl`

#### REQUIRED EVENT FIELDS

* `timestamp`
* `experience_id`
* `extractor_version`
* `source_context`: `{mode, source_weight, context_id, vad?, heuristics?}`
* `segmentation`: `{candidate_count}`
* `atoms`: `{raw_count, valid_count, invalid: [{text, reason}] }`
* `epistemics`: list per atom `{frame, confidence, signals}`
* `resolution`: list per atom `{outcome, match_confidence, candidate_ids, belief_id_final}`
* `conflicts`: `{hard_count, tension_count, edges_created}`
* `scoring`: `{activation_delta, core_score_delta, status_before_after}`
* `stream`: `{primary, secondary, confidence, migration_event?}`
* `error` optional

**Acceptance:** Every pipeline run produces a complete event, including failures.

---

## Source Context + Source Weight Computation (missing before, now explicit)

### TASK 2.2: `src/services/source_context_classifier.py`

**Input:** Experience (and any metadata you have), optional VAD telemetry
**Output:** `{mode, source_weight, context_id, details}`

#### REQUIRED RULES

* Determine base `mode`:

  * use Experience metadata if present, else "unknown"
* Compute base weight: `mode_weights[mode]`
* If VAD enabled and arousal present:

  * `source_weight = clamp(mode_weight - arousal_weight * arousal, 0, 1)`
* Else if heuristic fallback enabled:

  * apply penalties for profanity/caps/exclamation density
* Define `context_id` using config:

  * strategy `conversation_mode`: `conversation_id + ":" + mode`
  * fallback: `experience_id`

**Acceptance:** No code sets source_weight = 1.0 by default unless config makes it so.

---

## Segmentation + Canonicalization

### TASK 3.1: `src/services/belief_segmenter.py`

Deterministic split into claim candidates.
**Acceptance:** tests for punctuation and conjunction splits.

### TASK 3.2: `src/services/belief_canonicalizer.py`

Contractions, whitespace, punctuation normalization, stable ordering, `canonical_hash`.
**Acceptance:** same input -> same output.

---

## Atomization (LLM) + Validation

### TASK 4.1: `src/services/belief_atomizer.py`

* Input: list of claim candidates
* Output: strict JSON list: `{atom_text, belief_type, polarity, confidence, spans?}`

#### JSON repair behavior (explicit)

* If invalid JSON, run one repair attempt using a constant repair prompt template `repair_json_v1`
* If still invalid, drop result and log error in eval event

**Acceptance:** schema validation tests.

### TASK 4.2: `src/services/belief_atom_validator.py`

Reject atoms:

* not first-person
* imperative/instructional
* empty/too short noise
  Return reason codes for logging.

---

## Epistemics (Deterministic-first + fallback)

### TASK 5.1: `src/services/epistemics_rules.py`

#### Must implement cue conflict resolution exactly

* Assign each cue:

  * scope proposal
  * specificity score from config
  * index position (rightmost wins tie)
* Choose temporal scope by:

  1. highest specificity
  2. if tie: rightmost in string
  3. if tie: higher base conf

Order of operations:

1. polarity (negation)
2. modality (cap certainty)
3. temporal_scope
4. degree

**Special explicit expected output test case**
Text: "I always hate it when it rains right now"

* temporal_scope = habitual
* signals must include both habitual and state cues
  Reason: in self-definition extraction we treat frequency cues as identity builders; state cue becomes context, not scope.

**Acceptance:** unit tests for cue conflicts.

### TASK 5.2: `src/services/epistemics_llm.py`

Trigger only if `epistemic_confidence < config.epistemics.llm_fallback_threshold`.
Store override + confidence + extractor_version.

### TASK 5.3: Stream classifier (missing before, now required)

**Deliverable:** `src/services/stream_classifier.py`
**Input:** `{belief_type, epistemic_frame}`
**Output:** `{primary_stream, secondary_stream, confidence}` using `config.streams.mapping`

**Acceptance:** for FEELING_STATE + habitual -> primary identity; for FEELING_STATE + state -> primary state.

---

## Resolution + Concurrency + Embeddings

### TASK 6.1: `src/services/belief_resolver.py`

#### Embedding strategy (explicit)

* If config `use_embeddings=true`:

  * compute embedding on BeliefNode creation (store in BeliefNode.embedding)
  * retrieve candidates via cosine similarity top_k
* If embeddings not available, fallback:

  * exact match on canonical_hash
  * else Levenshtein similarity on canonical_text (only for top few candidates)

#### Concurrency strategy (explicit)

Use DB uniqueness on `BeliefNode.canonical_hash`:

* Attempt insert
* On unique violation: re-select existing node and treat as match
* Retry up to `config.concurrency.max_retries`

**Acceptance:** running two concurrent inserts for same canonical_hash yields one node, no duplicates.

#### 3-way outcome (explicit)

* sim >= match_threshold -> match
* sim <= no_match_threshold -> no-match
* else -> uncertain

If verifier is enabled:

* run verifier only in the uncertain band to push toward match/no-match, else keep uncertain.

### TASK 6.2: `src/services/tentative_link_service.py`

#### Age definition (explicit)

`age = days_since(last_support_at) if last_support_at else days_since(created_at)`

#### Confidence update (explicit)

`confidence = sigmoid(a*support_both - b*support_one - c*age)`

Update counters:

* If later evidence indicates both nodes are repeatedly selected by resolver as top candidates for similar atoms: `support_both += 1`
* If evidence consistently selects one and excludes the other: `support_one += 1`

Auto accept/reject per config.

### TASK 6.3: `scripts/review_tentative_link_clusters.py`

* Pairwise links only in v1
* Build connected components of pending links
* Print clusters sorted by avg confidence and size
* This satisfies transitivity awareness without auto-merging.

---

## Conflict Detection v1

### TASK 7.1: `src/services/conflict_engine.py`

Outputs:

* **Hard contradiction**: direct negation conflicts
* **Tension candidates**: embedding sim >= `config.resolution.tension.threshold` and opposite polarity

Create ConflictEdge with status default tolerated.

**Acceptance:** syntactic negation unit tests.

---

## Activation + Core Scoring + Migration Ratchet

### TASK 8.1: `src/services/activation_service.py`

v1 choice: recompute activation from occurrences (safe and simple).

* `activation = Σ (source_weight * exp(-age/half_life_stream))`
  Use stream from StreamAssignment primary.

**Acceptance:** deterministic given timestamps.

### TASK 8.2: `src/services/core_score_service.py`

Implement bounded + multiplicative score:

* `support = 1 - exp(-n_weighted / support_k_n)`
* `spread = sigmoid((distinct_days - spread_midpoint)/spread_temp)`
* `diversity = sigmoid((distinct_contexts - diversity_midpoint)/diversity_temp)`
* `base = support * spread * diversity`
* `core_score = base - conflict_penalty_recent`

Define `distinct_contexts` exactly:

* count distinct `context_id` values across occurrences (computed by Task 2.2)

Conflict penalty recent:

* over occurrences in last `recent_window_days` OR conflict edges created in that window.

**Acceptance:** 1000 occurrences in one day does not score high without spread/diversity.

### TASK 8.3: `src/services/stream_service.py`

* initial stream assignment comes from Task 5.3
* migration STATE -> IDENTITY when thresholds met (config)
* ratchet enabled: no demotion unless explicit triggers

Demotion triggers v1:

* explicit obsolescence cue (future work if needed)
* sustained conflict + low activation long period

**Acceptance:** "oscillating migration" does not happen.

---

## HTN Wiring

### TASK 9.1: `src/services/htn_belief_methods.py`

Implement method: `extract_and_update_self_knowledge(experience_id)` that calls tasks in order:

1. Load experience
2. source_context_classifier
3. segmenter
4. atomizer
5. atom_validator
6. canonicalizer
7. epistemics_rules (+ fallback)
8. stream_classifier
9. resolver (match/no/uncertain)
10. create node/occurrence
11. tentative link service update (if uncertain)
12. conflict engine (post-create)
13. activation + core score update for affected nodes
14. stream migration check
15. SelfKnowledgeIndex updates
16. eval event log

**Acceptance:** one call produces artifacts + eval event.

---

## Backfill + Resume + Rollback (all explicit)

### TASK 10.1: `scripts/backfill_self_definitions_to_beliefs.py`

Required behavior:

* batch size from config
* checkpoint file `data/backfill_checkpoint.json` with last processed experience_id and counts
* resume behavior:

  * if `backfill.resume_strategy=skip_if_occurrence_exists_for_extractor_version` then for each experience:

    * query occurrences where `source_experience_id == experience_id AND extractor_version == current`
    * if exists: skip
* partial failures:

  * if one experience fails, log error, continue
  * store failures list in checkpoint

**Acceptance:** re-run resumes and skips already-processed.

### TASK 10.2: SelfKnowledgeIndex integration

* On occurrence create: index the `source_experience_id` under topic/category (existing method)
* On rollback/removal: call `remove_claim(experience_id)` for each removed occurrence experience_id
* If `remove_claim` does not exist in your branch, implement it (your diff suggests it exists now). Ensure it prunes empty lists.

**Acceptance:** index stays consistent with rollbacks.

### TASK 10.3: Optional read-path cutover (defer if needed)

* Keep gardener on old system until eval metrics acceptable
* Then switch to BeliefNodes

### TASK 10.4: `scripts/rollback_extractor_version.py` (explicit rollback)

**Input:** extractor_version string
**Behavior:**

1. soft-delete occurrences where extractor_version matches (or delete if your DB tolerates it; prefer soft-delete with `deleted_at`)
2. recompute BeliefNode activation/core_score for any affected belief_id
3. delete orphaned TentativeLinks that reference deleted nodes or reduce their support
4. delete ConflictEdges whose `evidence_occurrence_ids` now empty
5. update SelfKnowledgeIndex via remove_claim for each affected experience_id
6. log rollback summary

**Acceptance:** rollback leaves DB consistent and nodes recomputed.

---

## Definition of Done (agent must meet)

1. New tables exist with uniqueness constraints preventing duplicates.
2. HTN method processes a single Experience end-to-end and logs eval events.
3. Backfill script is resumable and idempotent per extractor_version.
4. Rollback script works and repairs derived state and index.
5. TentativeLinks created for uncertain matches; cluster review script shows clusters.
6. Conflict engine emits contradictions and tension candidates (config thresholds).
7. All hyperparameters in config, not hardcoded.

---

## Suggested Build Order (agent workflow)

1. 0.1 config, 0.2 extractor version, 0.3 annotation harness
2. 1.1 models + migrations
3. 2.1 eval logger, 2.2 source context classifier
4. 3.1 segmenter, 3.2 canonicalizer
5. 4.1 atomizer, 4.2 validator
6. 5.1 epistemics rules, 5.2 epistemics LLM fallback, 5.3 stream classifier
7. 6.1 resolver (embeddings + concurrency), 6.2 tentative link service, 6.3 cluster review
8. 7.1 conflict engine
9. 8.1 activation, 8.2 core score, 8.3 stream ratchet/migration
10. 9.1 HTN wiring
11. 10.1 backfill, 10.2 index integration, 10.4 rollback

---

If you paste this to an agent, they can implement without inventing anything. If they try to "improve" it, they're doing it wrong.
