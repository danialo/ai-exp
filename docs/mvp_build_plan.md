# MVP Build Plan

This plan walks through the thin-slice MVP defined in `experience_schema.md`, organised into sequential stages with clear goals, prerequisites, implementation steps, validation checks, and deliverables. Follow each stage before advancing to the next to keep complexity contained.

---

## Stage 0 — Environment & Project Scaffolding
- **Goal**: Ensure a reproducible workspace for ingest + retrieval experiments.
- **Tasks**:
  1. Pick runtime stack (Python recommended) and create virtual environment (`uv`, `poetry`, or `venv`).
  2. Pin core dependencies: `fastapi` (API), `pydantic` (schema validation), `sqlmodel`/`SQLAlchemy` (raw store), `chromadb` or `faiss-cpu` (vector index), `numpy`, `sentence-transformers` or OpenAI embeddings wrapper, `pytest`.
  3. Initialize repo scaffolding: `src/` package, `tests/`, `scripts/`, config folder.
  4. Create `.env.example` with placeholder secrets (embedding API keys, DB paths).
- **Validation**: `pytest -q` runs empty test suite; lint/format (if using `ruff`/`black`) succeed.
- **Deliverable**: Baseline repo ready for feature development; dependency lockfile committed.

---

## Stage 1 — Data Contracts & Models
- **Goal**: Encode the raw experience schema in code for validation and persistence.
- **Tasks**:
  1. Translate JSON schema into Pydantic models (`Experience`, `SignatureEmbedding`, `AffectSnapshot`). Keep optional fields optional.
  2. Define SQL representation for the raw store (`experience` table minimal fields) using `sqlmodel`/`SQLAlchemy`.
  3. Add serialization helpers to map Pydantic → SQL rows and vice versa.
  4. Document field-level defaults (e.g., `affect.valence` default `0.0`, `confidence.p` default `0.5`).
- **Validation**: Unit tests that instantiate models with sample payloads from the doc examples and round-trip serialize/deserialize without loss.
- **Deliverable**: Schema module (`src/memory/models.py`) with tests in `tests/test_models.py`.

---

## Stage 2 — Raw Store Persistence Layer
- **Goal**: Provide a thin repository API to append immutable experiences and query by timestamp/ID.
- **Tasks**:
  1. Configure SQLite database file (e.g., `data/raw_store.db`). Enable WAL mode.
  2. Implement repository class with methods: `append_experience`, `get_experience(id)`, `list_recent(limit)`, `append_observation` (reflection shard helper).
  3. Enforce immutability: block updates/deletes except an explicit `tombstone` method guarded behind feature flag (MVP may stub).
  4. Add migration script (`alembic` or `sqlmodel` metadata create_all`).
- **Validation**: Integration test that inserts sample experiences, fetches them, verifies immutability constraint.
- **Deliverable**: `src/memory/raw_store.py` + tests `tests/test_raw_store.py`.

---

## Stage 3 — Embedding Provider & Vector Index
- **Goal**: Generate semantic embeddings and persist them to a vector database for retrieval.
- **Tasks**:
  1. Choose embedding source: local `sentence-transformers` model for offline (e.g., `all-MiniLM-L6-v2`) or OpenAI if API access is acceptable.
  2. Abstract embedding provider with interface `embed(text: str) -> np.ndarray` and `embed_batch(list[str])`.
  3. Set up vector store: start with `Chroma` (local, simple) or raw FAISS index stored on disk (`data/vector_index/`).
  4. Implement `VectorStore` wrapper with methods `upsert(id, vector, metadata)` and `query(vector, top_k)` returning IDs + scores.
  5. Integrate with raw store ingestion so each new experience writes prompt/response embeddings.
- **Validation**: Test that embedding provider returns deterministic vectors and vector store returns self-nearest neighbor for inserted samples.
- **Deliverable**: `src/memory/embedding.py`, `src/memory/vector_store.py`, plus tests.

---

## Stage 4 — Ingestion Pipeline (MVP Scope)
- **Goal**: Assemble a pipeline that accepts an interaction payload and produces stored experiences + embeddings.
- **Tasks**:
  1. Define ingestion function (`ingest_interaction(interaction: InteractionPayload)`), capturing: prompt, draft response, augmented response (temporarily same as draft), valence estimate (user only), timestamps.
  2. Generate semantic embeddings for prompt & response; store via vector index under roles `prompt_semantic`, `response_semantic`.
  3. Persist `Experience` record to raw store with references to embedding pointer strings (e.g., `vec://sem/exp123_prompt`).
  4. Log ingestion events (stdout or structured logs) to verify flow.
- **Validation**: Functional test feeding synthetic interaction; assert experience count increments, embeddings stored, retrieval (simple) returns the new entry.
- **Deliverable**: `src/pipeline/ingest.py` + `tests/test_ingest.py`.

---

## Stage 5 — Retrieval Service
- **Goal**: Provide an API that, given a prompt, retrieves relevant experiences for the lens stage.
- **Tasks**:
  1. Implement `retrieve_similar(prompt: str, top_k: int = 5)` that embeds prompt, queries vector store, fetches experiences, and sorts by recency-biased score (e.g., `score = cos_sim * 0.8 + recency_weight * 0.2`).
  2. Include optional filter to skip experiences older than configurable horizon.
  3. Return lightweight payload: experience ID, prompt excerpt, augmented response, valence.
  4. Expose retrieval via simple CLI command or FastAPI endpoint for manual checks.
- **Validation**: Tests that retrieval returns known similar sample and respects recency filter.
- **Deliverable**: `src/services/retrieval.py`, tests, optional `/retrieve` endpoint.

---

## Stage 6 — Experience Lens Pass v0
- **Goal**: Implement two-pass response flow with valence-based styling.
- **Tasks**:
  1. Wrap base LLM call (can mock or use OpenAI/local model). Get draft response.
  2. Call retrieval service for similar experiences; if none, fall back to draft.
  3. Apply simple tone adjustment: if blended valence < 0 → add empathetic preface; else present normally. Keep factual content untouched.
  4. Append experience citations at the end (`[exp_123]`).
  5. Return augmented response along with metadata (IDs used, valence scores).
- **Validation**: Unit tests with stubbed model & retrieval verifying tone changes only when valence negative and citations included.
- **Deliverable**: `src/pipeline/lens.py`, tests.

---

## Stage 7 — Reflection Shard Writer
- **Goal**: Capture post-response reflection as a lightweight observation experience.
- **Tasks**:
  1. Implement `record_reflection(interaction, retrieved_ids)` that drafts a short note (template-driven) summarizing what memory helped.
  2. Store as `type="observation"` via raw-store repository; embed using same provider.
  3. Optionally tag reflections with `label="reflection"` in `SignatureEmbedding.metadata`.
- **Validation**: Test that reflection records persist and link back to source experience IDs.
- **Deliverable**: `src/pipeline/reflection.py`, tests.

---

## Stage 8 — CLI / Notebook Harness
- **Goal**: Provide a manual way to exercise the full MVP loop end-to-end.
- **Tasks**:
  1. Build a CLI command (`scripts/run_interaction.py`) that accepts a prompt, runs draft (mock), lens pass, and reflection.
  2. Print augmented response, cited experience IDs, and store IDs.
  3. Optionally create a Jupyter notebook demonstrating ingestion of seed data and a sample query.
- **Validation**: Manual run with seed experiences shows retrieval working and reflections captured.
- **Deliverable**: CLI script + documentation snippet.

---

## Stage 9 — Tests & Guardrails
- **Goal**: Ensure regressions are caught early and affect stays stylistic.
- **Tasks**:
  1. Write regression test verifying that varying valence only changes surface text, not facts (string diff limited to prefixes/suffixes).
  2. Add integration test covering ingest → retrieve → lens → reflection pipeline.
  3. Configure CI (GitHub Actions or local script) running `pytest`, lint, and type checks.
- **Validation**: CI green; integration pipeline test passes deterministically.
- **Deliverable**: `tests/test_pipeline_integration.py`, CI workflow.

---

## Stage 10 — Documentation & Handover
- **Goal**: Capture how to operate the MVP and known limitations.
- **Tasks**:
  1. Update `README.md` with MVP overview, setup instructions, commands.
  2. Document config knobs (vector index path, recency weight, affect weights).
  3. Note out-of-scope features (uncertainty re-rank, full lattice, stance builder) for future roadmap.
- **Validation**: Another collaborator can follow README to ingest sample data and reproduce retrieval.
- **Deliverable**: README updates, `docs/OPERATIONS.md` (optional).

---

### Suggested Sequencing Summary
0. Environment setup → 1. Models → 2. Raw store → 3. Embeddings/Vector → 4. Ingest pipeline → 5. Retrieval service → 6. Lens pass → 7. Reflection → 8. Harness → 9. Tests/guardrails → 10. Documentation.

Following this order ensures the data foundation exists before retrieval logic, and retrieval exists before the lens/reflection layers.
