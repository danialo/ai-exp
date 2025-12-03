# Changelog

All notable changes to the AI Experience Memory System MVP.

## [Unreleased]

### Added
- Query telemetry dataclasses and scoring helpers (`src/services/research_query_telemetry.py`, `src/services/research_query_scoring.py`)
- Search query validation/sanitization utilities with regression fixtures (`src/services/research_query_utils.py`, `tests/fixtures/known_bad_queries.json`)
- Configurable fallback executor and authority reranker (`src/services/research_query_fallback.py`, `src/services/research_result_reranker.py`)
- Targeted tests for scoring, sanitization, and HTN fallback integration (`tests/services/test_research_query_scoring.py`, `tests/services/test_research_query_utils.py`, `tests/services/test_research_htn_investigate.py`)

### Changed
- `InvestigateTopic` now records per-attempt telemetry, enforces stricter query prompts, runs sanitization, and retries using heuristic fallbacks before fetching sources.
- Documentation (`docs/RESEARCH_HTN_ROADMAP.md`) updated with the new query generation pipeline and tuning knobs.

## [1.0.0] - 2025-10-19

### Completed MVP - All 10 Stages ✅

#### Stage 6: Experience Lens Pass v0
- **Added** `src/pipeline/lens.py` - Affect-aware response styling
- **Added** `tests/test_lens.py` - 15 comprehensive tests
- **Features**:
  - Two-pass LLM flow (draft → lens-adjusted)
  - Valence-based tone adjustment with empathetic prefacing
  - Experience citations appended to responses
  - Blended valence calculation from retrieved memories
  - Configurable valence threshold for tone sensitivity

#### Stage 7: Reflection Shard Writer
- **Added** `src/pipeline/reflection.py` - Post-response learning observations
- **Added** `tests/test_reflection.py` - 15 comprehensive tests
- **Features**:
  - Template-driven reflection note generation
  - OBSERVATION type experiences for reflections
  - Parent experience linking (interaction + retrieved memories)
  - Reflection-tagged embeddings for future retrieval
  - Affect inheritance from blended valence

#### Stage 10: Documentation & Operations
- **Updated** `README.md` - Comprehensive setup, usage, and operations guide
- **Added** `docs/OPERATIONS.md` - Detailed operational guide
- **Added** `CHANGELOG.md` - This file
- **Documentation includes**:
  - Quick start guide
  - Configuration reference (all environment variables)
  - Architecture diagrams and data flow
  - API usage examples (REST and Python)
  - Troubleshooting guide
  - Deployment instructions (systemd, nginx)
  - Performance tuning recommendations
  - Backup and recovery procedures
  - Security considerations
  - Future roadmap

### Summary

**Complete MVP Implementation:**
- ✅ 10/10 stages from `docs/mvp_build_plan.md`
- ✅ 127 passing tests (100% test coverage on core components)
- ✅ Web interface (FastAPI + HTML/JS)
- ✅ REST API with OpenAPI docs
- ✅ CLI tools and demos
- ✅ Comprehensive documentation

**Technical Highlights:**
- Immutable experience storage with SQLite
- Semantic vector search with ChromaDB
- Sentence transformers for embeddings
- OpenAI GPT integration for responses
- Affect-aware tone adjustment
- Post-response reflection learning
- Recency-biased retrieval ranking

**Files Added/Modified:**
- `src/pipeline/lens.py` (new, 223 lines)
- `src/pipeline/reflection.py` (new, 248 lines)
- `tests/test_lens.py` (new, 15 tests)
- `tests/test_reflection.py` (new, 15 tests)
- `README.md` (updated, comprehensive rewrite)
- `docs/OPERATIONS.md` (new, operational guide)
- `CHANGELOG.md` (new, this file)

**Test Results:**
```
127 passed in 19.65s
```

**System Stats:**
- 11 experiences stored
- 22 vectors indexed
- LLM enabled (gpt-3.5-turbo)
- Server running at http://172.239.66.45:8000

### Next Steps (Future Roadmap)

See `README.md` and `docs/OPERATIONS.md` for:
- Uncertainty re-ranking
- Full experience lattice (inference, reconciliation)
- Multi-modal content support
- Advanced affect models
- Local LLM integration
- Multi-user support
- Horizontal scaling

## [0.9.0] - 2025-10-18

### Prior Work
- Stages 0-5, 8-9 completed
- Web interface implemented
- Core memory system operational

---

**MVP Status**: ✅ COMPLETE
