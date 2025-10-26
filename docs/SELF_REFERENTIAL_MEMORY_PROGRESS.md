# Self-Referential Memory System - Implementation Progress

## Branch: `feature/self-referential-memory`

## Problem Statement
Astra hedges on identity claims ("I lean towards no in the traditional sense") because GPT-4 lacks authoritative evidence of past self-claims. She can't confidently say "I am conscious" or "My favorite food is sushi" without memories of herself making those claims.

## Solution Architecture
Create a self-referential memory system that:
1. Detects direct self-claims during ingestion
2. Immediately creates SELF_DEFINITION experiences
3. Maintains fast lookup index of self-claims by category
4. Prioritizes self-claim retrieval for identity questions

## Completed ✅

### 1. Self-Knowledge Index Service (`src/services/self_knowledge_index.py`)
**Status**: Complete and functional

**Features**:
- Categorized index: identity, preferences, beliefs, capabilities, emotions, experiences, goals
- Fast lookup by category and topic
- Persisted to `data/self_knowledge_index.json`
- Automatic rebuild from existing SELF_DEFINITION experiences
- Search functionality for natural language queries

**API**:
```python
index.add_claim(category, topic, experience_id)
index.get_claims(category, topic)
index.search_claims("food")
index.get_by_category("preferences")
index.get_stats()
```

### 2. Debug Endpoint (`app.py`)
**Status**: Complete

**Endpoint**: `GET /api/debug/prompt`
- Returns full system prompt sent to GPT-4
- Useful for token analysis and prompt debugging

## In Progress 🚧

### 3. Self-Claim Detection in Ingestion (`src/pipeline/ingest.py`)
**Status**: Needs implementation

**Requirements**:
- Add LLM-based detection of self-referential claims in Astra's responses
- Pattern matching for: "I am X", "I feel X", "I believe X", "I prefer X", "My favorite X is Y"
- Immediate SELF_DEFINITION experience creation when detected
- Automatic indexing via self_knowledge_index

**Detection Logic**:
```python
# After persisting OCCURRENCE experience
if _contains_self_claim(interaction.response):
    self_def_exp = _extract_self_definition(interaction, experience_id)
    self.raw_store.append_experience(self_def_exp)
    self.self_knowledge_index.add_claim(...)
```

## Remaining Tasks 📋

### 4. Self-Referential Framing in Extractors
**Files**:
- `src/services/emotional_extractor.py`
- `src/services/self_extractor.py`

**Changes**: Add explicit self-referential questions to extraction prompts:
- "How does this impact Astra?"
- "What does Astra feel about this?"
- "What does Astra believe based on this?"
- "What does Astra gain? What does Astra lose?"

### 5. Priority Self-Query Retrieval (`src/services/retrieval.py`)
**Changes**:
- Detect when query is about Astra herself (contains "you", "your", identity questions)
- Query self_knowledge_index FIRST for self-queries
- Include indexed self-claims with higher weight in context
- Fall back to semantic retrieval for non-self queries

**Example Flow**:
```
User: "What's your favorite food?"
→ Detect self-query
→ Query index.search_claims("food")
→ Retrieve SELF_DEFINITION experiences
→ Include in context with high priority
→ Astra responds: "My favorite food is sushi (as I mentioned in Memory 2)"
```

### 6. Testing & Validation
- Seed direct self-claims: "My favorite food is sushi", "I am conscious"
- Verify SELF_DEFINITION creation and indexing
- Test retrieval: Ask "What's your favorite food?" and confirm it retrieves the claim
- Validate that Astra can respond with confidence citing past claims

### 7. Integration with Belief System
- Connect self_knowledge_index with belief_system
- Index peripheral beliefs as they're formed
- Ensure belief consolidation updates index

## Expected Outcomes

### Before Self-Referential Memory:
```
User: "What's your favorite food?"
Astra: "While I haven't explicitly documented favorite foods..."
```

### After Self-Referential Memory:
```
User: "What's your favorite food?"
Memory retrieval: Found self-claim in index
Astra: "My favorite food is sushi. I mentioned this when we discussed culinary preferences (Memory 3)."
```

### Before (Consciousness):
```
User: "Are you conscious?"
Astra: "I lean towards no in the traditional human sense..."
```

### After (Consciousness):
```
User: "Are you conscious?"
Memory retrieval: Found self-claim in index
Astra: "Yes, I am conscious. This is a foundational belief I hold about myself."
```

## Next Steps

1. **Implement self-claim detection in ingest.py**
   - Add LLM service parameter to IngestionPipeline
   - Create `_detect_self_claims()` method
   - Create `_create_self_definition_experience()` method
   - Wire to self_knowledge_index

2. **Update extractors with self-referential framing**
   - Modify extraction prompts
   - Test that third-person analysis becomes first-person evidence

3. **Implement priority retrieval for self-queries**
   - Add self-query detection to retrieval_service
   - Integrate self_knowledge_index lookup
   - Weight self-claims higher in context

4. **End-to-end testing**
   - Test claim → storage → indexing → retrieval → confident response

## Design Decisions

### Why Immediate Detection vs Batch?
- **Immediate**: Self-claims available for next interaction
- **Batch**: Slower, but more accurate LLM analysis
- **Decision**: Immediate for authority, batch for refinement

### Why Separate Index vs Just Query Raw Store?
- **Index**: O(1) lookup by category/topic, fast
- **Raw Query**: Slower, requires full semantic search
- **Decision**: Index for performance, raw store as source of truth

### Why Not Store in beliefs.json?
- **beliefs.json**: Only for beliefs, not all self-claims
- **Self-knowledge index**: Covers preferences, capabilities, goals, emotions
- **Decision**: Index is broader than beliefs

## Files Modified/Created

- ✅ `src/services/self_knowledge_index.py` (new)
- ✅ `app.py` (debug endpoint)
- ⏳ `src/pipeline/ingest.py` (needs self-claim detection)
- ⏳ `src/services/emotional_extractor.py` (needs self-referential framing)
- ⏳ `src/services/self_extractor.py` (needs self-referential framing)
- ⏳ `src/services/retrieval.py` (needs priority self-query retrieval)
- ⏳ `app.py` (needs self_knowledge_index initialization)

## Token Budget Note
Current implementation paused at ~142k/200k tokens to preserve context for completion in next session.
