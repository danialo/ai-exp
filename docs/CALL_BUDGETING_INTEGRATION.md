# Call Budgeting Integration Plan

**Date**: 2025-11-15
**Status**: Ready to implement

## Problem

Research synthesis can overflow context limits when:
- Many source documents (>20 docs × ~2k tokens each = 40k+ tokens)
- Large documents with extensive claims
- No chunking or size control currently implemented

Current code calls `llm_service.summarize_research_session()` which doesn't exist yet.

## Solution

Add CallBudgeter to handle automatic chunking with map-reduce pattern:
1. **Map phase**: Multiple chunked synthesis calls (if needed)
2. **Reduce phase**: Single merge call (if multiple chunks)

---

## Architecture

### 1. CallBudgeter (src/services/call_budgeter.py) ✅ CREATED

Responsibility: Plan LLM calls to stay within context limits

**Interface**:
```python
class CallBudgeter:
    def plan_chunked_calls(
        items,
        estimate_tokens_for_item,
        base_prompt_tokens,
        desired_response_tokens
    ) -> List[ChunkPlan]
```

**Key features**:
- Greedy bin packing algorithm
- Safety margin (80% utilization default)
- Handles oversized single items gracefully
- No LLM calls itself - only planning

### 2. LLMService Extensions (src/services/llm.py) - TO ADD

Add three methods:

#### a) `summarize_research_session()` - Main entry point
```python
def summarize_research_session(
    self,
    root_question: str,
    docs: List[dict],
    tasks: List[dict],
) -> dict:
    """Synthesize research session with automatic chunking."""
```

**Logic**:
1. Estimate tokens for all docs
2. Ask CallBudgeter for plan
3. If 1 chunk → direct synthesis
4. If multiple chunks → chunked_summarize_docs() + merge

#### b) `chunked_summarize_docs()` - Map phase
```python
def chunked_summarize_docs(
    self,
    docs: List[dict],
    root_question: str,
    response_tokens: int = 512
) -> List[dict]:
    """Summarize docs in chunks, return partial summaries."""
```

**Logic**:
1. Get chunk plans from CallBudgeter
2. For each plan:
   - Extract subset of docs
   - Generate partial summary (key_events, contested_claims, etc.)
3. Return list of partial summaries

#### c) `merge_partial_summaries()` - Reduce phase
```python
def merge_partial_summaries(
    self,
    partials: List[dict],
    root_question: str,
) -> dict:
    """Merge multiple partial summaries into final synthesis."""
```

**Logic**:
1. Combine all key_events, contested_claims, open_questions
2. Ask LLM to deduplicate and prioritize
3. Generate final narrative_summary
4. Return unified summary object

---

## Token Estimation Strategy

### For SourceDocs

Use cached or heuristic approach:

```python
def estimate_doc_tokens(doc: dict) -> int:
    """Estimate tokens for a SourceDoc."""
    # Content + claims + metadata
    content_tokens = len(doc.get("content", "")) // 4
    claims_tokens = sum(len(c.get("claim", "")) // 4 for c in doc.get("claims", []))
    url_tokens = len(doc.get("url", "")) // 4
    return content_tokens + claims_tokens + url_tokens + 50  # overhead
```

### For Base Prompt

Fixed estimate based on synthesis prompt template:

```python
BASE_SYNTHESIS_PROMPT_TOKENS = 800  # system + instructions + JSON schema
```

---

## Implementation Steps

### Step 1: Add to LLMService.__init__()

```python
def __init__(self, ...):
    # ... existing init ...

    # Add call budgeter for chunking large operations
    from src.services.call_budgeter import CallBudgeter
    self.call_budgeter = CallBudgeter(
        max_tokens_per_call=self.max_tokens * 2,  # Use 2x max_tokens as context limit estimate
        safety_margin=0.8
    )
```

### Step 2: Implement `summarize_research_session()`

Main method called by SynthesizeFindings:

```python
def summarize_research_session(
    self,
    root_question: str,
    docs: List[dict],
    tasks: List[dict],
) -> dict:
    """Synthesize research session with automatic chunking if needed.

    Returns:
        dict with: narrative_summary, key_events, contested_claims,
                   open_questions, coverage_stats
    """
    if not docs:
        return {
            "narrative_summary": "No sources found",
            "key_events": [],
            "contested_claims": [],
            "open_questions": [],
            "coverage_stats": {"sources_investigated": 0}
        }

    # Estimate tokens
    from src.services.call_budgeter import estimate_tokens_from_dict
    doc_estimates = [estimate_tokens_from_dict(d, key="content") +
                    sum(len(c.get("claim", "")) // 4 for c in d.get("claims", []))
                    for d in docs]

    base_prompt_tokens = 800  # synthesis prompt + schema
    desired_response = 512

    # Check if chunking needed
    total_tokens = base_prompt_tokens + sum(doc_estimates) + desired_response

    if total_tokens <= self.call_budgeter.max_effective:
        # Single call synthesis
        return self._single_call_synthesis(root_question, docs, tasks)
    else:
        # Chunked synthesis with merge
        partials = self._chunked_summarize_docs(docs, root_question, doc_estimates)
        return self._merge_partial_summaries(partials, root_question)
```

### Step 3: Implement `_single_call_synthesis()`

Direct synthesis when everything fits:

```python
def _single_call_synthesis(
    self,
    root_question: str,
    docs: List[dict],
    tasks: List[dict],
) -> dict:
    """Single-call synthesis when all docs fit in context."""

    prompt = f"""You are synthesizing research findings for this question:
{root_question}

Source documents ({len(docs)} total):
{json.dumps(docs, indent=2)}

Task execution log:
{json.dumps(tasks, indent=2)}

Generate a comprehensive synthesis with:
1. narrative_summary: 2-3 paragraph overview
2. key_events: List of 3-7 most important findings
3. contested_claims: Claims with conflicting sources
4. open_questions: What remains unclear
5. coverage_stats: Sources investigated, claims extracted

Return valid JSON only."""

    response = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": "You are a research synthesis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1024,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

### Step 4: Implement `_chunked_summarize_docs()`

Map phase - process docs in chunks:

```python
def _chunked_summarize_docs(
    self,
    docs: List[dict],
    root_question: str,
    doc_estimates: List[int],
) -> List[dict]:
    """Summarize docs in chunks when they don't fit in single call."""

    plans = self.call_budgeter.plan_chunked_calls(
        items=docs,
        estimate_tokens_for_item=lambda d: doc_estimates[docs.index(d)],
        base_prompt_tokens=800,
        desired_response_tokens=512,
    )

    partials = []
    for plan in plans:
        subset = [docs[i] for i in plan.item_indices]

        prompt = f"""Partial synthesis for: {root_question}

Analyze these {len(subset)} sources and extract:
- key_events: Important findings
- contested_claims: Conflicting information
- open_questions: Unclear aspects

Sources:
{json.dumps(subset, indent=2)}

Return JSON with: key_events, contested_claims, open_questions"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=plan.response_token_budget,
            response_format={"type": "json_object"}
        )

        partials.append(json.loads(response.choices[0].message.content))

    return partials
```

### Step 5: Implement `_merge_partial_summaries()`

Reduce phase - combine partial results:

```python
def _merge_partial_summaries(
    self,
    partials: List[dict],
    root_question: str,
) -> dict:
    """Merge multiple partial summaries into final synthesis."""

    # Combine all findings
    all_key_events = []
    all_contested = []
    all_questions = []

    for p in partials:
        all_key_events.extend(p.get("key_events", []))
        all_contested.extend(p.get("contested_claims", []))
        all_questions.extend(p.get("open_questions", []))

    prompt = f"""Merge these partial research summaries for: {root_question}

Key events from {len(partials)} chunks:
{json.dumps(all_key_events, indent=2)}

Contested claims:
{json.dumps(all_contested, indent=2)}

Open questions:
{json.dumps(all_questions, indent=2)}

Create final synthesis:
1. narrative_summary: 2-3 paragraphs
2. key_events: Top 5-7 most important (deduplicated)
3. contested_claims: Actual conflicts (deduplicated)
4. open_questions: Top 3-5 questions
5. coverage_stats: Estimate sources/claims

Return JSON only."""

    response = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": "You are a research synthesis expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1024,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

---

## Testing Plan

### Unit Tests (call_budgeter.py)

Test chunking logic:
```python
def test_single_chunk_when_fits():
    budgeter = CallBudgeter(max_tokens_per_call=1000)
    items = [{"content": "x" * 100} for _ in range(5)]  # 500 chars = ~125 tokens
    plans = budgeter.plan_chunked_calls(
        items=items,
        estimate_tokens_for_item=lambda d: len(d["content"]) // 4,
        base_prompt_tokens=100,
        desired_response_tokens=100,
    )
    assert len(plans) == 1

def test_multiple_chunks_when_large():
    budgeter = CallBudgeter(max_tokens_per_call=500)
    items = [{"content": "x" * 400} for _ in range(10)]  # 4000 chars = ~1000 tokens
    plans = budgeter.plan_chunked_calls(
        items=items,
        estimate_tokens_for_item=lambda d: len(d["content"]) // 4,
        base_prompt_tokens=100,
        desired_response_tokens=100,
    )
    assert len(plans) > 1

def test_oversized_single_item():
    budgeter = CallBudgeter(max_tokens_per_call=500)
    items = [{"content": "x" * 2000}]  # 2000 chars = ~500 tokens (too big)
    plans = budgeter.plan_chunked_calls(
        items=items,
        estimate_tokens_for_item=lambda d: len(d["content"]) // 4,
        base_prompt_tokens=100,
        desired_response_tokens=100,
    )
    assert len(plans) == 1
    assert plans[0].response_token_budget < 100  # Reduced from desired
```

### Integration Tests (llm.py)

Test synthesis with varying doc counts:
```python
def test_synthesis_single_chunk():
    llm = LLMService(...)
    docs = [create_test_doc(i) for i in range(5)]
    result = llm.summarize_research_session("test question", docs, [])
    assert "narrative_summary" in result
    assert len(result["key_events"]) > 0

def test_synthesis_chunked():
    llm = LLMService(...)
    docs = [create_test_doc(i) for i in range(50)]  # Force chunking
    result = llm.summarize_research_session("test question", docs, [])
    assert "narrative_summary" in result  # Still returns valid synthesis
```

---

## Rollout

1. **Create call_budgeter.py** ✅
2. **Add methods to llm.py** - Next
3. **Test with small doc set** - Verify single-call path
4. **Test with large doc set** - Verify chunking + merge
5. **Run benchmark** - Validate in production flow
6. **Monitor logs** - Check synthesis_complete events for chunking behavior

---

## Benefits

- **No more context overflows**: Automatic chunking handles any doc count
- **Explicit and predictable**: Plans are deterministic, easy to debug
- **Separation of concerns**: CallBudgeter never touches LLM, LLMService never does manual chunking
- **Observable**: Can log chunk plans to see when/why chunking triggered
- **Future-proof**: Same pattern works for other large operations (belief reconciliation, claim extraction)

---

## Next Steps

1. Implement three LLMService methods
2. Add unit tests for CallBudgeter
3. Integration test with research_htn_methods
4. Update logging to capture chunking events
5. Run benchmark to validate
