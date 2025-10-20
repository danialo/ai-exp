# Experience Record Schema

## Goals
- Treat every interaction as an immutable "raw experience" while supporting derived views that reinterpret those memories under different lenses.
- Represent experience signatures primarily through embedding vectors (semantic, temporal, causal, affective) with optional symbolic scaffolding for auditability.
- Encode affect (valence–arousal–dominance) so tone adapts to user emotion and agent self-affect, while keeping factual stance independent of affect.
- Support uncertainty-aware retrieval, stance reconciliation, and reflexive learning without model weight updates.

## Experience Type Lattice
Experience types form a lattice: `occurrence ⊑ observation ⊑ inference ⊑ reconciliation`.
- Experiences may be derived *upward* (e.g., an observation summarizing an occurrence) but never rewritten downward.
- Links between records carry diffs noting what changed (added/removed/reweighted evidence).

## Memory Architecture
- **Raw Store (immutable)**: Holds canonical `experience` records plus WAL/tombstones; content is append-only. Provenance braids ensure lineage across derived artifacts.
- **View Store (derived, immutable but supersedable)**: Holds lenses, stances, summaries, and other refractions that reference raw IDs and store deltas relative to the raw store.

## Core Entities

### 1. `experience` (Raw Store)
Represents a single immutable memory—either captured directly or derived upward in the lattice.

| Field | Type | Description |
| --- | --- | --- |
| `id` | TEXT | e.g., `exp_2025-10-02T17:01:05Z_9f3c`. |
| `type` | ENUM | `occurrence`, `observation`, `inference`, `reconciliation`. |
| `created_at` | TIMESTAMP | UTC timestamp of capture. |
| `content` | JSON | `{ "text": "...", "media": [], "structured": {} }`. |
| `provenance` | JSON | Sources, actor (`user|agent|tool(name)`), method (`capture|scrape|model_infer|reconcile`). |
| `evidence_ptrs` | ARRAY | References to parent experiences or external URIs. |
| `confidence` | JSON | `{ "p": 0.78, "method": "calibrated_logit" }`. |
| `embeddings` | JSON | Pointers to embedding vectors (semantic/temporal/causal) + optional symbolic triples. |
| `affect` | JSON | Valence–arousal–dominance, categorical labels, intensity, confidence. |
| `parents` | ARRAY | Direct lineage IDs. |
| `sign` | TEXT | Content signature (e.g., `ed25519:...`). |

#### Embedding Pointers
Embedding pointers (e.g., `vec://sem/exp_...`) reference rows stored in the vector index. Each pointer corresponds to a `signature_embedding` entry (see below) that records the vector, model version, salience, optional hash, and metadata.

### 2. `signature_embedding`
Defines role-specific embeddings that together constitute the experience "signature".

| Field | Type | Description |
| --- | --- | --- |
| `id` | UUID | Signature identifier. |
| `experience_id` | TEXT FK | References `experience.id`. |
| `role` | TEXT | `prompt_semantic`, `response_semantic`, `temporal_profile`, `causal_profile`, `affect_profile`, etc. |
| `embedding_model` | TEXT | Embedding model/version. |
| `vector` | VECTOR | Stored in FAISS/LanceDB/Chroma as appropriate. |
| `label` | TEXT | Optional classifier label (`feature_request`, `critical`, etc.). |
| `hash` | TEXT | Optional LSH (MinHash/SimHash) for near-duplicate detection. |
| `salience` | FLOAT | Influence weight during retrieval (0–1). |
| `metadata` | JSON | Role-specific metadata (entities, domain tags). |

### 3. `affect_snapshot`
Captures affect for each actor/stage with explicit confidence tracking.

| Field | Type | Description |
| --- | --- | --- |
| `id` | UUID | Snapshot identifier. |
| `experience_id` | TEXT FK | Owning experience. |
| `actor` | TEXT | `user` or `agent`. |
| `stage` | TEXT | `input`, `draft`, `augmented`, etc. |
| `primary_emotion` | TEXT | Canonical label (`joy`, `frustration`, `neutral`, …). |
| `valence` | FLOAT | [-1, 1]. |
| `arousal` | FLOAT | [0, 1]. |
| `dominance` | FLOAT | [-1, 1]. |
| `confidence` | FLOAT | Confidence in the affect reading. |
| `notes` | TEXT | Rationale/interpretation. |
| `signals` | JSON | Raw cues (lexical markers, punctuation, discourse features). |

### 4. `view` (View Store)
Represents a lens-derived reframe referencing raw experiences.

| Field | Type | Description |
| --- | --- | --- |
| `id` | TEXT | e.g., `view_2025-10-02T17:21:14Z_1ab7`. |
| `lens` | TEXT | Lens identifier (e.g., `causal_v2`, `policy_v1`). |
| `applied_to` | ARRAY | Raw experience IDs referenced. |
| `delta_from_raw` | JSON | Structural diff: `{ "added": "...", "removed": "...", "reweighted": [...] }`. |
| `affect_delta` | JSON | Affect shift from source to view. |
| `provenance_braid` | ARRAY | IDs documenting lineage (raw + prior views). |
| `sign` | TEXT | Content signature of the view. |

### 5. `stance`
Stores reconciliation outputs resolving conflicting claims.

| Field | Type | Description |
| --- | --- | --- |
| `id` | TEXT | e.g., `stance_2025-10-02_f3ca`. |
| `question` | TEXT | Question being reconciled. |
| `candidates` | JSON | `{ "exp": "exp_a", "claim": "...", "p": 0.62 }` list. |
| `method` | TEXT | `justified_truth_maintenance+bayes_pool`, etc. |
| `decision` | JSON | Final claim, supporting IDs, residual uncertainty. |
| `created_at` | TIMESTAMP | When stance recorded. |

### 6. `wal_entry`
Write-ahead log for append/tombstone operations.

| Field | Type | Description |
| --- | --- | --- |
| `wal_id` | TEXT | Incrementing ID (e.g., `wal_000042`). |
| `ops` | JSON ARRAY | Sequence of operations `{ "ts": "...", "op": "append|tombstone", ... }`. |

## JSON Examples
```json
{
  "id": "exp_2025-10-02T17:01:05Z_9f3c",
  "type": "occurrence",
  "created_at": "2025-10-02T17:01:05Z",
  "content": {"text": "...", "media": [], "structured": {}},
  "provenance": {
    "sources": [{"uri": "https://...", "hash": "sha256:..."}],
    "actor": "user",
    "method": "capture"
  },
  "evidence_ptrs": ["exp_...", "uri:..."],
  "confidence": {"p": 0.78, "method": "calibrated_logit"},
  "embeddings": {
    "semantic": "vec://sem/exp_...",
    "temporal": "vec://temp/exp_...",
    "causal": "vec://caus/exp_...",
    "symbolic": [{"subject": "X", "rel": "causes", "object": "Y"}]
  },
  "affect": {
    "vad": {"v": 0.15, "a": 0.62, "d": 0.40},
    "labels": ["frustration"],
    "intensity": 0.7,
    "confidence": 0.66
  },
  "parents": ["exp_...", "exp_..."],
  "sign": "ed25519:..."
}
```

```json
{
  "id": "view_2025-10-02T17:21:14Z_1ab7",
  "lens": "causal_v2",
  "applied_to": ["exp_...", "exp_..."],
  "delta_from_raw": {
    "added": "...",
    "removed": "...",
    "reweighted": [{"exp": "exp_..", "w_old": 0.3, "w_new": 0.5}]
  },
  "affect_delta": {
    "from": {"v": 0.10, "a": 0.60, "d": 0.40},
    "to": {"v": 0.20, "a": 0.45, "d": 0.55}
  },
  "provenance_braid": ["exp_..", "view_..", "view_.."]
}
```

```json
{
  "id": "stance_2025-10-02_f3ca",
  "question": "Which metric should we report?",
  "candidates": [
    {"exp": "exp_a", "claim": "...", "p": 0.62},
    {"exp": "exp_b", "claim": "...", "p": 0.31}
  ],
  "method": "justified_truth_maintenance+bayes_pool",
  "decision": {"claim": "...", "support": ["exp_a", "view_x"], "residual_uncertainty": 0.22}
}
```

```json
{
  "wal_id": "wal_000042",
  "ops": [
    {"ts": "...", "op": "append", "target": "exp", "id": "exp_...", "key_id": "k_..."},
    {"ts": "...", "op": "tombstone", "target": "exp", "id": "exp_...", "reason": "gdpr_erasure", "purge_key": "k_..."}
  ]
}
```

## Retrieval & Processing Pipeline
1. **Decompose Query**: Extract intent, entities, constraints, temporal scope.
2. **Hybrid Retrieval**: Combine semantic KNN, temporal filters, causal walks, and keyword BM25. Score candidates by weighted combination `α·semantic + β·recency + γ·causal + δ·keyword`.
3. **Uncertainty Re-rank**: Estimate disagreement/entropy (Active-Prompt style) across candidate answers; boost memories with high discriminative power.
4. **Lens Application**: Project retrieved raw experiences through relevant views (policy, causal, stance-specific).
5. **Stance Reconciliation**: Build a justification graph (JTMS) and apply Bayesian pooling to resolve conflicting claims. Record `stance` artifact with residual uncertainty.
6. **Affect Blending**: Blend user VAD + retrieved-memory VAD + agent self-VAD (weights e.g., 0.5/0.3/0.2). Clamp to affect-only style parameters (warmth, pace, hedging) without altering truth selection.
7. **Micro-program Execution**: Run an ART-style lightweight program interleaving retrieval, tools, and Chain-of-Draft (≤20 tokens) notes.
8. **Compose Answer**: Content follows stance decision; style follows affect blend; cite experience IDs inline.
9. **Reflexion Write-back**: Store a reflection shard noting what helped or hindered, written as an `observation` experience.
10. **Persistence**: Append to WAL, snapshot as needed, update provenance braid.

## Key Algorithms (Pseudocode)

### Query Decomposition
```python
def decompose(query: str) -> dict:
    intent = classify_intent(query)
    ents = ner_link(query)
    time = extract_temporal_scope(query)
    cons = extract_constraints(query)
    return {"intent": intent, "entities": ents, "time": time, "constraints": cons}
```

### Hybrid Retrieval with Uncertainty Re-rank
```python
def retrieve(decomp: dict, k: int = 64):
    S = semantic_knn(decomp, k=256)
    T = temporal_filter(S, decomp.get("time"))
    C = causal_walk(T, hops=2)
    K = bm25(query=decomp, topn=256)
    pool = union_topn([C, K], n=512)
    scored = [(
        e,
        alpha * cos_sim(e, decomp)
        + beta * recency(e)
        + gamma * causal_score(e, decomp)
        + delta * keyword_score(e)
    ) for e in pool]
    top = topn(scored, n=k)
    U = [(e, answer_disagreement(e)) for e, _ in top]
    return rerank_by(U, weight=0.25)
```

### Stance Reconciliation
```python
def reconcile(candidates: list):
    jtms = JTMS()
    for exp in candidates:
        jtms.assert_claim(
            exp["claim"],
            support=exp["evidence_ptrs"],
            p=exp["confidence"]["p"],
        )
    jtms.propagate()
    pooled = bayesian_pool([c["confidence"]["p"] for c in candidates])
    decision = argmax_over_claims(pooled)
    return {
        "decision": decision,
        "support": jtms.support_of(decision),
        "residual_uncertainty": 1 - pooled[decision],
    }
```

### Affect Blending (Style Only)
```python
def blend_vad(user_vad, mem_vad, self_vad, w=(0.5, 0.3, 0.2)):
    def clamp(x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, x))

    v = clamp(w[0] * user_vad["v"] + w[1] * mem_vad["v"] + w[2] * self_vad["v"])
    a = clamp(w[0] * user_vad["a"] + w[1] * mem_vad["a"] + w[2] * self_vad["a"])
    d = clamp(w[0] * user_vad["d"] + w[1] * mem_vad["d"] + w[2] * self_vad["d"])
    return {"v": v, "a": a, "d": d}
```

### Micro-program Execution (ART-style)
```python
def run_program(plan: list):
    ctx = {}
    for step in plan:
        if step["op"] == "retrieve":
            ctx[step["as"]] = retrieve(step["args"], k=step.get("k", 16))
        elif step["op"] == "tool":
            out = call_tool(step["name"], step["args"])
            ctx[step["as"]] = out
        elif step["op"] == "draft":
            ctx[step["as"]] = concise_note(step["text"], max_tokens=20)
    return ctx
```

### Reflexion Write-back
```python
def reflect(trial: dict):
    summary = verbal_reflection(trial)
    store_experience(
        type="observation",
        content={"text": summary},
        provenance={"actor": "agent", "method": "reflect"},
    )
```

## Tone & Style Mapping
- **Valence → Warmth**: `warmth = 0.5 + 0.5 * V` (cap 0–1).
- **Arousal → Pace**: `avg_sentence_length = base_len - 6 * A`; mark urgent when `A ≥ 0.6`.
- **Dominance → Hedging**: `hedge_rate = base_rate - 0.4 * D`; imperative usage increases with higher dominance.
- Clamp all adjustments to prevent factual changes; stance selection remains affect-agnostic.

## Refinements
- Experience type lattice with upward-only derivations; diffs stored on links.
- Two-tier storage (Raw vs View) with immutable records and explicit provenance braids.
- Affect blending limited to stylistic adjustments; truth selection isolated from affect.
- Uncertainty-aware retrieval inspired by Active-Prompt to favor discriminative memories.
- Chain-of-Draft internal notes keep reasoning concise while preserving multi-step thinking.
- Reflexion shards capture post-response reflections for future improvement.
- Tool micro-programs (ART-style) let the agent pause for retrieval/tools and resume reasoning seamlessly.

## Pitfalls & Guardrails
- **Affect leakage**: enforce architectural separation so stance logic never references style knobs.
- **Conflicting lenses**: version views; retain diffs so new lenses don't occlude prior interpretations.
- **Compaction risk**: during pruning, keep cryptographic digests / Merkle roots to preserve provenance.
- **Affect misreads**: use multi-signal detectors, track confidence bands, avoid hard gating on low-confidence VAD.

## Enhanced Prompt Template (XML-style)
```xml
<DEFINE type="rules">
  <RULE>Raw memories are immutable; rewrites live in views with explicit diffs.</RULE>
  <RULE>Affect modifies style only; factual stance is affect-agnostic.</RULE>
  <RULE>Always cite experience IDs backing claims.</RULE>
</DEFINE>

<DEFINE type="tags">
  <TAG name="INTENT">Identify intent, entities, constraints, time.</TAG>
  <TAG name="PLAN">List retrieval + tool steps (micro-program).</TAG>
  <TAG name="RETRIEVE">Hybrid search over semantic/temporal/causal/keyword.</TAG>
  <TAG name="RECONCILE">Build stance via JTMS + Bayesian pooling.</TAG>
  <TAG name="AFFECT">Blend VAD sources → style knobs.</TAG>
  <TAG name="DRAFT">Chain-of-Draft notes, ≤20 tokens.</TAG>
  <TAG name="ANSWER">Final content with experience citations.</TAG>
  <TAG name="REFLEXION">Post-mortem reflection shard.</TAG>
</DEFINE>

<PROMPT>
  <INTENT/>
  <PLAN>
    <STEP op="RETRIEVE">Rank by α·semantic + β·recency + γ·causal + δ·keyword; re-rank by disagreement.</STEP>
    <STEP op="DRAFT">Minimal hypothesis path note.</STEP>
    <STEP op="TOOL">Optional tool usage with resume point.</STEP>
  </PLAN>
  <RECONCILE/>
  <AFFECT/>
  <ANSWER/>
  <REFLEXION/>
</PROMPT>
```

## Strategy Table

| Strategy | Where Used | Benefit |
| --- | --- | --- |
| Few-shot / in-context | Query decomposition & retrieval exemplars | Adapts to new tasks without fine-tuning. |
| Active-Prompt uncertainty | Retrieval re-rank | Surfaces memories that resolve ambiguity. |
| Self-consistency (optional) | Hard queries | Boosts reliability via majority vote. |
| ART micro-programs | Planning/tool orchestration | Enables pause/resume tool calls mid-generation. |
| Reflexion | Post-answer reflection | Lightweight learning from verbal feedback. |
| Chain-of-Draft | Internal reasoning | Maintains step transparency with low token overhead. |

## Next Build Steps
1. Implement the Raw/View store split with WAL + tombstone handling and provenance braid maintenance.
2. Build the hybrid retrieval service integrating uncertainty-aware re-ranking and affect-aware filtering.
3. Calibrate affect detectors on labeled VAD data; incorporate confidence thresholds into style controls.
4. Prototype the enhanced prompt template and micro-program execution harness.
5. Add safety tests ensuring affect knobs never alter stance decisions or experience citations.
6. Create a gold pipeline harness (≈200 tasks) to measure accuracy, uncertainty, token usage, and latency end-to-end.

## MVP Scope (Thin Slice)
- **Data ingestion (Raw Store only)**: persist `experience` records with semantic embeddings (`prompt_semantic`, `response_semantic`) and minimal affect snapshot (user valence only). Skip lattice derivations except `occurrence`.
- **Vector retrieval**: store embeddings in a single index (FAISS/Chroma); support top-k cosine search filtered by recency.
- **Experience lens pass v0**: base model draft → retrieve top-N similar experiences → second pass that styles response using simple valence rules (warm vs. neutral tone). No stance reconciliation yet.
- **Reflection shard**: after response, append a short `observation` noting what memory helped; manual review only.
- **CLI notebook harness**: run the end-to-end loop on canned examples, log IDs, embeddings, and affect scores for inspection.
- **Manual guardrails**: enforce affect/style separation via unit test asserting factual content unchanged when style knobs vary.

### MVP Exit Criteria
- Able to ingest ≥50 experiences and retrieve relevant ones for a fresh prompt within acceptable latency (<500 ms retrieval in local tests).
- Augmented responses cite at least one prior experience ID when recall is used.
- Affect-styled responses adjust tone without altering factual assertions across regression prompts.
- Reflection shards successfully append and can be manually inspected via simple query.
