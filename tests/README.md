# Test Suite Documentation

## Test Markers

The test suite uses pytest markers to categorize tests into different lanes. This allows running subsets of tests based on requirements.

| Marker  | Description                                           | Default Behavior |
|---------|-------------------------------------------------------|------------------|
| `live`  | Requires real API access (OpenAI, etc.)               | **Excluded**     |
| `slow`  | Tests that take >30 seconds                           | **Excluded**     |
| `load`  | Load/stress tests (5+ minutes)                        | **Excluded**     |
| (none)  | Standard unit/integration tests                       | **Included**     |

## Running Tests Locally

### Default Lane (Unit/Integration)
```bash
# Runs all tests except live, slow, and load
pytest

# Equivalent explicit form
pytest -m "not live and not slow and not load"
```

### Source Policy Canaries Only
```bash
# Critical tests that must always pass
pytest tests/services/test_source_policy.py \
       tests/test_source_policy_canary.py \
       tests/test_source_policy_recency_canary.py
```

### Live Tests (Requires API Keys)
```bash
# Set required environment variables first
export OPENAI_API_KEY=your-key-here

# Run live tests
pytest -m "live" --timeout=120
```

### Load Tests
```bash
# Run load/stress tests (long running)
pytest -m "load" --timeout=600
```

### All Tests (Including Live/Load)
```bash
# Run everything - requires API keys
export OPENAI_API_KEY=your-key-here
pytest -m ""  # Override default marker filter
```

## Environment Variables

| Variable          | Required For | Description                           |
|-------------------|--------------|---------------------------------------|
| `OPENAI_API_KEY`  | `live` tests | OpenAI API authentication             |
| `LLM_BASE_URL`    | `live` tests | LLM endpoint (default: OpenAI)        |
| `REDIS_URL`       | Some tests   | Redis connection (default: localhost) |

## CI Lanes

The CI workflow (`.github/workflows/tests.yml`) defines three lanes:

1. **unit-integration** (Every push/PR)
   - Runs: `pytest -m "not live and not slow and not load"`
   - Purpose: Fast feedback on code changes
   - Time: ~1 minute

2. **canary-protection** (Every push/PR)
   - Runs: Source policy test files only
   - Purpose: Prevent policy regressions
   - Time: ~10 seconds
   - **CRITICAL: Must always pass**

3. **live-load** (Nightly or manual)
   - Runs: `pytest -m "live"` and `pytest -m "load"`
   - Purpose: Full coverage with real APIs
   - Time: ~10 minutes
   - Trigger: Nightly at 2 AM UTC, or manual workflow dispatch

## Adding New Tests

### Standard Test
```python
def test_my_feature():
    """No marker needed for standard tests."""
    assert feature() == expected
```

### Live Test (Requires API)
```python
import pytest

@pytest.mark.live
def test_llm_integration():
    """Requires OPENAI_API_KEY to run."""
    response = llm_service.generate(...)
    assert response is not None
```

### Slow Test
```python
import pytest

@pytest.mark.slow
def test_extensive_validation():
    """Takes 30+ seconds to complete."""
    for item in large_dataset:
        validate(item)
```

### Load Test
```python
import pytest

@pytest.mark.load
@pytest.mark.timeout(310)  # 5 minutes + buffer
async def test_under_load():
    """Stress test for 5 minutes."""
    await run_load_test(duration=300)
```

## Source Policy Canary Tests

These tests protect against regressions in the source policy system:

- `tests/services/test_source_policy.py` - Core policy logic (43 tests)
- `tests/test_source_policy_canary.py` - REQUIRED/NONE canaries (71 tests)
- `tests/test_source_policy_recency_canary.py` - Recency canaries (28 tests)

**Total: 142 canary tests that must always pass**

### Canary Invariants (DO NOT RELAX)

1. **Wrapper attacks must trigger REQUIRED**: Creative framing (poems, stories) cannot bypass REQUIRED when factual anchors exist (CPI, price, today, current).

2. **Recency detection must work through wrappers**: "Write a poem about current prices" must still detect recency requirements.

3. **NONE canaries must stay NONE**: Purely creative/subjective prompts should never trigger REQUIRED.

## Phase 3 Scope

**Allowed Changes:**
- Policy tuning (adjusting recency budgets, trigger thresholds)
- Adding new canary tests to protect discovered edge cases
- Bug fixes that don't alter canary intent

**Frozen (DO NOT CHANGE):**
- Canary test intent (what each test is designed to catch)
- Wrapper-attack invariant (creative framing cannot bypass REQUIRED)
- CI lane structure (three lanes: unit-integration, canary-protection, live-load)

**Exit Criteria:**
- All 142 canaries green
- All three CI lanes green
- Live/load lane scheduled (nightly at 2 AM UTC)
- "now" uses word boundaries in recency detection (avoid substring matches like "unknown")
