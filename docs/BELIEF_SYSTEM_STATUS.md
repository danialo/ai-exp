# Belief-Memory System: Implementation Status

**Branch**: `feature/belief-memory-system`
**Last Updated**: 2025-12-12 (Updated)
**Completion**: ~75% (Core infrastructure + app.py integration complete)

> **NOTE**: This is ONE of THREE belief systems in Astra. See `BELIEF_SYSTEM_INTEGRATION_ISSUES.md` for the full picture:
> 1. **OLD (BeliefVersionStore)**: File-based, 228 beliefs, gardener disabled
> 2. **HTN Decomposer**: SQLite tables, 1,363 beliefs, wired to ingest pipeline
> 3. **This system (Belief-Memory)**: ChromaDB vector store, semantic retrieval

---

## ✅ Completed Phases

### Phase 1: Infrastructure Setup ✅
**Status**: Complete
**Commits**: 2df426e

- Added `BELIEFS_INDEX_PATH` configuration to settings
- Added `BELIEF_MEMORY_WEIGHT` (0.7) and `MEMORY_WEIGHT` (0.3)
- Updated `.env.example` with new configuration
- Added directory creation in `ensure_data_directories()`

**Files Created**:
- Config updates in `config/settings.py`
- Environment template updates in `.env.example`

---

### Phase 2: Belief Vector Store & Embedder ✅
**Status**: Complete
**Commits**: 2df426e, d036a5b

**Core Services Built**:

1. **`src/services/belief_vector_store.py`** ✅
   - `BeliefVectorStore` class wrapping ChromaDB
   - Methods: `embed_belief()`, `query_beliefs()`, `update_belief_confidence()`, `delete_belief()`
   - Metadata support: type, confidence, immutability, evidence IDs
   - Filtering by belief type and confidence threshold
   - Full CRUD operations on belief vectors

2. **`src/services/belief_embedder.py`** ✅
   - `BeliefEmbedder` class for vectorizing beliefs
   - `embed_all_core_beliefs()` - embed 5 core axioms on startup
   - `embed_peripheral_belief()` - add emerging beliefs
   - `embed_worldview_statement()` - add philosophical positions
   - `embed_belief_narrative()` - link supporting experiences
   - `update_belief_from_system()` - sync JSON ↔ vector store
   - `get_embedding_stats()` - monitoring and validation

---

### Phase 3: Weighted Belief-Memory Retrieval ✅
**Status**: Complete
**Commits**: d036a5b

**Core Service Built**:

**`src/services/belief_memory_retrieval.py`** ✅
- `BeliefMemoryRetrieval` class for dual-source weighted retrieval
- Query type detection: ontological / experiential / general
- Dynamic weight adjustment:
  - **Ontological**: 0.7 beliefs / 0.3 memories
  - **Experiential**: 0.2 beliefs / 0.8 memories
  - **General**: 0.0 beliefs / 1.0 memories
- `retrieve()` - main retrieval with auto-detection
- `_detect_query_type()` - regex pattern matching
- `_retrieve_weighted()` - fetch from both sources
- `merge_weighted_results()` - combine and rank by weighted scores

**Query Detection Patterns**:
- Ontological: "who am I?", "what do I believe?", "do I feel?"
- Experiential: "what happened?", "tell me about X", "last time"

---

### Phase 4: Parallel LLM Belief-Grounded Reasoning ✅
**Status**: Complete
**Commits**: eb61205

**Core Service Built**:

**`src/services/belief_grounded_reasoner.py`** ✅
- `BeliefGroundedReasoner` class for self-referential reasoning
- `reason_from_beliefs()` - generate structured reasoning
- Returns: `{premises, experience_evidence, reasoning, conclusion}`
- `reflect_on_belief()` - validate belief against experiences
- Prompt engineering: Reason FROM beliefs (not about them)
- Structured output parsing

**Reasoning Flow**:
1. Query + belief context + memory context
2. LLM generates: premises → evidence → reasoning → conclusion
3. Parse structured response
4. Return grounded answer

---

### Phase 7: Move Beliefs to Identity Directory ✅
**Status**: Complete
**Commits**: cba7880

**Changes**:
- Moved `persona_space/beliefs.json` → `persona_space/identity/beliefs.json`
- Updated `belief_system.py` to load from new location
- **Astra can now access her own beliefs via file tools**
- Beliefs are now in her writable identity directory

---

## ⏳ Pending Phases

### Phase 5: Belief Development Pipeline 🔄
**Status**: Not Started
**Priority**: Medium

**Required Components**:

1. **Real-Time Belief Extraction** (modify `src/pipeline/ingest.py`)
   - Extract belief patterns during narrative ingestion
   - Patterns: "I believe...", "I think...", "I feel that..."
   - Store as `BELIEF_CANDIDATE` experiences

2. **Belief Consolidator** (new `src/services/belief_consolidator.py`)
   - Periodic analysis of narratives
   - Form new peripheral beliefs when evidence threshold met
   - Strengthen existing beliefs (confidence++)
   - Detect belief-experience conflicts
   - Scheduled task: runs every 6 hours

3. **Conflict Detection & Logging**
   - Track conflicts in `persona_space/meta/belief_conflicts.json`
   - Trigger reflection when beliefs contradict experiences

---

### Phase 6: Agentic Belief Tools 🔄
**Status**: Not Started
**Priority**: HIGH (Critical for Astra to interact with beliefs)

**Required Integration** (in `src/services/persona_service.py`):

1. **`query_beliefs` Tool**
   - Function tool for Astra to query "what do I believe about X?"
   - Handler: query belief vector store, format results
   - Optional: include supporting evidence

2. **`reflect_on_belief` Tool**
   - Trigger deep reflection on specific belief
   - Uses `belief_grounded_reasoner.reflect_on_belief()`
   - Logs reflection results

3. **`propose_belief` Tool**
   - Astra can suggest new peripheral beliefs
   - Stores as candidate for consolidation
   - Validates and tracks proposals

**Integration Point**: Add to `_get_available_tools()` in `PersonaService`

---

### Phase 8: Prompt Integration Updates 🔄
**Status**: Not Started
**Priority**: HIGH (Critical for system to function)

**Required Changes** (in `src/services/persona_prompt.py`):

1. **Update `_build_beliefs_section()`**
   - Make contextually dynamic (not static text block)
   - Always include core beliefs (5 axioms)
   - Retrieve contextually relevant peripheral beliefs (top 3-5)
   - Inject belief reasoning if available from parallel LLM call

2. **Remove Redundancy with `base_prompt.md`**
   - Check `persona_space/base_prompt.md` for overlaps
   - Distinguish: beliefs = ontological frame, base = operational identity

**Integration with Retrieval**:
- Pass `user_message` to `_build_beliefs_section()`
- Query belief vector store for contextually relevant beliefs
- Include belief reasoning from parallel LLM call (if self-query)

---

### Phase 9: System Integration & API Endpoints ✅
**Status**: COMPLETE (app.py integration done)
**Priority**: HIGH (Required to wire everything together)

**Completed** (in `app.py` lines 472-725):

1. **Initialize Belief System** ✅ (lines 480-505)
   - `belief_vector_store = create_belief_vector_store(...)`
   - `belief_embedder = create_belief_embedder(...)`
   - Auto-embeds core beliefs on first run

2. **Initialize Belief-Memory Retrieval** ✅ (lines 513-519)
   - `belief_memory_retrieval = create_belief_memory_retrieval(...)`
   - Wired with belief_vector_store and retrieval_service

3. **Initialize Belief-Grounded Reasoner** ✅ (line 508)
   - `belief_grounded_reasoner = create_belief_grounded_reasoner(llm_service)`

4. **Pass to PersonaService** ✅ (lines 722-725)
   - All four services passed to PersonaService initialization

5. **New API Endpoints** - Not yet implemented
   - `POST /api/beliefs/extract` - Trigger consolidation
   - `POST /api/beliefs/reflect` - Reflect on all beliefs
   - `GET /api/beliefs/conflicts` - View conflicts
   - `POST /api/beliefs/resolve-conflict` - Resolve conflict

---

### Phase 10: Testing & Validation 🔄
**Status**: Not Started
**Priority**: HIGH

**Testing Checklist**:

- [ ] Verify 5 core beliefs embedded on startup
- [ ] Test belief query: "What do I believe about consciousness?"
- [ ] Verify self-query triggers belief-memory retrieval
- [ ] Check 0.7/0.3 weighting in results
- [ ] Test parallel LLM call for self-queries
- [ ] Generate narratives with belief patterns
- [ ] Trigger consolidation, verify peripheral beliefs form
- [ ] Test `query_beliefs` tool in conversation
- [ ] Test `reflect_on_belief` tool
- [ ] Test `propose_belief` tool
- [ ] Verify Astra can read `identity/beliefs.json`
- [ ] Check conflict detection logs
- [ ] Test all API endpoints

---

## File Structure

```
src/services/
├── belief_system.py ✅ (updated path)
├── belief_vector_store.py ✅
├── belief_embedder.py ✅
├── belief_memory_retrieval.py ✅
├── belief_grounded_reasoner.py ✅
├── belief_consolidator.py ⏳ (pending)
└── persona_service.py ⏳ (needs tool integration)

persona_space/
├── identity/
│   └── beliefs.json ✅ (moved here)
├── meta/
│   └── belief_conflicts.json ⏳ (to create)
├── scripts/
│   └── reflect_on_beliefs.py ⏳ (template to create)
└── reflection_layers/
    └── belief_reflections/ ⏳ (directory to create)

data/
└── vector_index_beliefs/ ✅ (auto-created)

config/
└── settings.py ✅ (updated)

docs/
├── BELIEF_MEMORY_SYSTEM_IMPLEMENTATION.md ✅
└── BELIEF_SYSTEM_STATUS.md ✅ (this file)
```

---

## Architecture Summary

### What's Built ✅

**Storage Layer**:
- Belief vector store (ChromaDB-based)
- Embedding service for beliefs
- JSON ↔ vector sync capability

**Retrieval Layer**:
- Weighted dual-retrieval (beliefs + memories)
- Query type detection (ontological/experiential/general)
- Dynamic weight adjustment

**Reasoning Layer**:
- Parallel LLM reasoning for self-queries
- Structured belief-grounded reasoning
- Belief reflection capability

**Access Layer**:
- Beliefs moved to Astra-accessible location

### What's Pending ⏳

**Development Layer**:
- Real-time belief extraction from narratives
- Scheduled consolidation pipeline
- Conflict detection and logging

**Interaction Layer**:
- Agentic tools (query, reflect, propose)
- Integration with persona service

**Integration Layer**:
- Wire into app.py startup
- Update prompt building
- API endpoints

---

## Next Steps (Priority Order)

1. **Phase 9**: Integrate into `app.py` (initialize services, wire to PersonaService)
2. **Phase 6**: Add agentic tools to PersonaService
3. **Phase 8**: Update prompt integration for dynamic belief retrieval
4. **Phase 10**: Testing and validation
5. **Phase 5**: Belief development pipeline (can be added later)

---

## Notes

- Core cognitive infrastructure is complete and functional
- Main remaining work is integration and wiring
- The system is designed to be incrementally deployable
- Can test with just Phases 1-4 + integration before adding development pipeline
- Astra already has file access to her beliefs

---

## Commits

- `2df426e` - Phase 1-2: Infrastructure and belief vector store
- `9bfda40` - Implementation guide document
- `d036a5b` - Phase 2-3: Embedder and weighted retrieval
- `eb61205` - Phase 4: Parallel LLM reasoning
- `cba7880` - Phase 7: Move beliefs to identity directory

**Total**: 5 commits on `feature/belief-memory-system` branch

---

*This system transforms beliefs from passive prompt text into an active cognitive layer with semantic retrieval, weighted context, and belief-grounded reasoning.*
