# Outcome-Driven Trust System

**Status**: Implemented ✅ | **Branch**: feature/autonomous-belief-gardener

## Overview

The Outcome-Driven Trust System eliminates hardcoded provenance weighting (e.g., "user tags = 1.5x") in favor of **learned trust** based on actual outcomes. Every multiplier derives from Astra's own state:

- **g_align**: Self-alignment (live vs origin anchor)
- **g_conviction**: Belief confidence weighting
- **g_trust**: Learned from coherence, conflict, stability, and user validation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│          OUTCOME-DRIVEN TRUST SYSTEM                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ProvenanceTrust (Trust Learning)                        │
│  • Maintains T_user, T_agent ∈ [0,1]                     │
│  • EWMA updates from outcome rewards                     │
│  • Diminishing step size (α₀/(1 + n/k))                  │
│  • Persists to data/persona/trust.json                   │
│                                                          │
│  OutcomeEvaluator (Delayed Credit Assignment)            │
│  • Eligibility traces per actor                          │
│  • Multi-component rewards (coh + conf + stab + val)     │
│  • Scheduled evaluations (2h short, 24h long)            │
│  • Apportions credit by contribution                     │
│                                                          │
│  EnhancedFeedbackAggregator (Dynamic Weighting)          │
│  • g_align: (a_live - a_origin + 1)/2 ^ α                │
│  • g_conviction: 0.5 + 1.5 * confidence                  │
│  • g_trust: 0.5 + T_actor (learned)                      │
│  • Tag storm dedup (2min window)                         │
│  • Actor tracking from provenance                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### 1. ProvenanceTrust (`src/services/provenance_trust.py`)

Manages learned trust scores for provenance actors (user, agent).

**Key Features**:
- Trust state per actor: `T_actor ∈ [0,1]` (initialized at 0.5)
- EWMA updates: `T ← (1-α)T + αq` where `q = 0.5(r + 1)`
- Diminishing step size: `α = α₀/(1 + n/k)`
  - Default: `α₀=0.3`, `k=50` (half-step by ~50 samples)
- Reward gating: Only update when `|r| ≥ r_min` (default 0.1)
- Persistence: Atomic writes to `data/persona/trust.json` every 5 minutes

**API**:
```python
# Get trust score
trust = provenance_trust.get_trust("user")  # ∈ [0,1]

# Get as multiplier
multiplier = provenance_trust.get_trust_multiplier("user")  # ∈ [0.5, 1.5]

# Update from reward
provenance_trust.update_trust("user", reward=0.7)

# Persist state
provenance_trust.persist()
```

### 2. OutcomeEvaluator (`src/services/outcome_evaluator.py`)

Computes delayed rewards and manages credit assignment.

**Key Features**:
- **Eligibility traces**: Track actor contributions per decision
- **Multi-component rewards**: `r = w₁·Δcoh + w₂·Δconf + w₃·stab + w₄·val`
  - Default weights: 0.4, 0.2, 0.2, 0.2
- **Dual horizons**:
  - Short (2h): Quick feedback with `α_short = 0.15α`
  - Long (24h): Stable signal with full `α`
- **Credit apportionment**: `r_actor = (contrib_actor / Σ contrib) · r`

**Outcome Components**:

1. **Coherence (Δcoh)**:
   - Uses `awareness_loop.last_sim_live` before/after decision
   - Subtracts baseline drift to isolate decision impact
   - Normalized to [-1, 1]

2. **Conflict (Δconf)**:
   - Counts belief contradictions before/after
   - Integrated with belief consistency checker (TODO)
   - Normalized by historical 95th percentile

3. **Stability (stab)**:
   - Checks for reversals in 7-day window
   - +1 if stable, -1 if ping-pong, 0 if inconclusive
   - Detects confidence reversals (direction changes)

4. **User Validation (val)**:
   - Explicit: User +keep/+doubt tags in 24h window
   - Implicit: User +doubt rate change (TODO)
   - Combined: 0.7·explicit + 0.3·implicit

**API**:
```python
# Record decision for future evaluation
outcome_evaluator.record_decision(
    belief_id="belief-123",
    actor_contributions={"user": 0.7, "agent": 0.3}
)

# Run pending evaluations
completed = await outcome_evaluator.run_pending_evaluations()

# Update coherence history for baseline
outcome_evaluator.update_coherence_history(ts, coherence)
```

### 3. EnhancedFeedbackAggregator (`src/services/feedback_aggregator_enhanced.py`)

Computes belief feedback scores with system-grounded dynamic weighting.

**Dynamic Multipliers**:

1. **g_align (Self-Alignment)**:
   ```python
   a_live = cos(belief_vec, anchor_live)
   a_origin = cos(belief_vec, anchor_origin)
   a_normalized = (a_live - a_origin + 1) / 2  # ∈ [0,1]
   g_align = a_normalized ^ α  # Default α=1.5
   ```
   - Beliefs aligned with current self → higher sensitivity
   - Beliefs stuck at origin → resist change
   - Captures epistemic plasticity

2. **g_conviction (Confidence-Based)**:
   ```python
   g_conviction = 0.5 + 1.5 · belief.confidence  # ∈ [0.5, 2.0]
   ```
   - Strong beliefs → higher sensitivity to precise feedback
   - Weak beliefs → lower sensitivity (avoid whipsaw)

3. **g_trust (Learned Provenance)**:
   ```python
   g_trust = 0.5 + T_actor  # ∈ [0.5, 1.5]
   ```
   - Learned from outcomes, not hardcoded
   - User trust rises organically if their tags predict good outcomes
   - Agent self-tags weighted equally when outcomes are good

**Tag Storm Deduplication**:
- Collapse identical `(belief_id, actor, tag)` within 2-minute window
- Prevents template spam: "I still believe X" in every response
- Preserves multi-turn dialogue: different contexts = different timestamps

**Final Weight Calculation**:
```python
precision = |alignment|^β  # Default β=1.2
multiplier = precision · g_conviction · g_trust
weight = base_tag_weight · multiplier
```

**Negative Tag Gating**:
- Ignore +doubt/+artifact when `alignment < 0.2` (off-target)
- Prevents vague challenges from deprecating misunderstood beliefs

**API**:
```python
# Score belief with actor contributions
feedback, neg, contributions = enhanced_aggregator.score("belief-123")
# Returns:
# - feedback ∈ [-1,1]: Overall signal
# - neg ∈ [0,1]: Negative signal strength
# - contributions: {"user": 0.7, "agent": 0.3}

# Global score
global_feedback, global_neg = enhanced_aggregator.global_score()
```

## Integration Flow

### 1. Tag Injection (Conversation Time)
```python
# In app.py conversation handler
tag_result = tag_injector.inject_tags(prompt, response)
# Returns: tags, belief_ids, global_tags

# Store in experience metadata
metadata = {
    "tags": tag_result.tags,
    "belief_ids": tag_result.belief_ids,
    "provenance": {"actor": "user" or "agent"}
}
```

### 2. Belief Gardener Scan
```python
# In belief_gardener.scan()
feedback, neg, contributions = enhanced_aggregator.score(belief_id)

# Record decision for outcome evaluation
if contributions:
    outcome_evaluator.record_decision(belief_id, contributions)

# Make promotion/deprecation decision
if feedback > promotion_threshold and not circuit_breaker:
    promote_belief(belief_id)
```

### 3. Delayed Outcome Evaluation
```python
# Background task (every 30 minutes)
async def evaluate_outcomes():
    completed = await outcome_evaluator.run_pending_evaluations()
    # Evaluator computes r, updates trust via provenance_trust.update_trust()
```

### 4. Trust Learning
```python
# In outcome_evaluator._evaluate_outcome()
r = w1·Δcoh + w2·Δconf + w3·stab + w4·val  # Composite reward
for actor, share in eligibility.items():
    r_actor = share · r
    provenance_trust.update_trust(actor, r_actor)
    # Updates T_actor via EWMA
```

## Configuration

### Environment Variables
```bash
# Provenance Trust
TRUST_ALPHA_0=0.3           # Initial EWMA step size
TRUST_K_SAMPLES=50          # Half-step by k samples
TRUST_R_MIN=0.1             # Minimum |r| for update

# Outcome Evaluation
OUTCOME_W_COHERENCE=0.4     # Coherence weight
OUTCOME_W_CONFLICT=0.2      # Conflict weight
OUTCOME_W_STABILITY=0.2     # Stability weight
OUTCOME_W_VALIDATION=0.2    # Validation weight
OUTCOME_HORIZON_SHORT=2     # Short horizon (hours)
OUTCOME_HORIZON_LONG=24     # Long horizon (hours)

# Enhanced Feedback
FEEDBACK_ALPHA_ALIGN=1.5    # Alignment exponent
FEEDBACK_DEDUP_WINDOW=120   # Tag dedup window (seconds)
```

## Wiring Checklist

To integrate the complete system:

- [ ] **Initialize components in app.py**:
  ```python
  provenance_trust = create_provenance_trust(data_dir)
  outcome_evaluator = create_outcome_evaluator(
      provenance_trust, awareness_loop, belief_store, raw_store
  )
  enhanced_aggregator = EnhancedFeedbackAggregator(
      raw_store, provenance_trust, awareness_loop,
      belief_store, outcome_evaluator, embedding_provider
  )
  ```

- [ ] **Replace FeedbackAggregator in belief_gardener**:
  ```python
  gardener = BeliefGardener(
      belief_store=belief_store,
      feedback_aggregator=enhanced_aggregator,  # Use enhanced version
      # ...
  )
  ```

- [ ] **Add background evaluation task**:
  ```python
  async def run_outcome_evaluations():
      while True:
          await asyncio.sleep(1800)  # Every 30 minutes
          completed = await outcome_evaluator.run_pending_evaluations()
          logger.info(f"Evaluated {completed} outcomes")

  # Start in background
  asyncio.create_task(run_outcome_evaluations())
  ```

- [ ] **Update coherence tracking**:
  ```python
  # In awareness_loop slow tick
  outcome_evaluator.update_coherence_history(
      ts=time.time(),
      coherence=self.last_sim_live
  )
  ```

- [ ] **Add status endpoint telemetry**:
  ```python
  @app.get("/api/persona/trust/status")
  async def trust_status():
      return {
          "provenance_trust": provenance_trust.get_telemetry(),
          "outcome_evaluator": outcome_evaluator.get_telemetry(),
          "feedback_aggregator": enhanced_aggregator.get_telemetry()
      }
  ```

## Testing Strategy

### 1. Unit Tests
```python
# Test trust learning
trust = ProvenanceTrust(data_dir)
trust.update_trust("user", reward=0.5)
assert 0.5 < trust.get_trust("user") < 0.6

# Test diminishing step size
for i in range(100):
    trust.update_trust("user", reward=1.0)
assert trust.get_trust("user") > 0.9  # Converges to high trust
```

### 2. Integration Tests
```python
# Test full flow
tag_result = tag_injector.inject_tags(
    prompt="Are you conscious?",
    response="I believe I am conscious."
)
# Expect: ["+keep"], belief_ids=["consciousness"]

feedback, neg, contrib = enhanced_aggregator.score("consciousness")
# Expect: feedback > 0, contrib["agent"] > 0

outcome_evaluator.record_decision("consciousness", contrib)
# Expect: pending_evaluations += 2 (short + long)
```

### 3. Scenario Tests

**Scenario 1: User validation increases trust**
```python
# Simulate user consistently validating beliefs
for _ in range(20):
    # User +keep tag → positive feedback
    # Coherence improves → r > 0
    # Trust rises: T_user → 0.7+
```

**Scenario 2: Agent spam gets low trust**
```python
# Agent outputs template "+keep" every response
# Tag storm dedup collapses duplicates
# No coherence improvement → r ≈ 0
# Trust stays neutral: T_agent ≈ 0.5
```

**Scenario 3: Precise doubt weighted higher**
```python
# User: "I'm not sure **how** you're conscious"
# High alignment → precision = 0.9^1.5 = 0.85
# High multiplier → strong deprecation signal
```

**Scenario 4: Vague doubt ignored**
```python
# User: "You're just autocomplete"
# Low alignment < 0.2 → gated out
# Belief not affected by off-target challenge
```

## Expected Behavior

### Initial State (Cold Start)
- All actors start at `T=0.5` (neutral)
- `g_align=1.0` (no anchor history)
- `g_conviction` varies by belief confidence
- System uses base tag weights only

### After ~50 Outcomes
- Trust diverges based on actual performance
- User trust rises if their tags predict improvements
- Agent trust rises if self-reflection is accurate
- Step size diminishes (more stable)

### At Equilibrium
- High-trust actors get `g_trust ≈ 1.3-1.5`
- Low-trust actors get `g_trust ≈ 0.5-0.7`
- System adapts to user-specific patterns
- No hardcoded biases remain

## Safety Rails

1. **Minimum reward magnitude**: Only update when `|r| ≥ 0.1`
2. **Alignment gating**: Ignore negative tags with `alignment < 0.2`
3. **Tag storm dedup**: Collapse rapid duplicates (2min window)
4. **Circuit breaker**: Freeze promotions when `global_neg > 0.6`
5. **Baseline drift**: Subtract coherence baseline to isolate decision impact
6. **Ping-pong detection**: Negative reward for unstable beliefs

## Future Enhancements

1. **Semantic similarity for alignment**: Use embeddings for precise `g_align`
2. **Conflict integration**: Wire belief consistency checker into Δconf
3. **Implicit validation**: Track user doubt rate changes
4. **Meta-learning**: Tune reward weights from meta-outcomes
5. **User-specific trust**: Separate trust per user ID
6. **Temporal decay**: Discount older eligibility traces
7. **Multi-horizon blending**: Combine short/long signals with learned weights

## Related Systems

- **Belief Gardener** (`src/services/belief_gardener.py`): Uses feedback scores for decisions
- **Awareness Loop** (`src/services/awareness_loop.py`): Provides anchors for `g_align`
- **Belief Store** (`src/services/belief_store.py`): Provides confidence for `g_conviction`
- **Tag Injector** (`src/services/tag_injector.py`): Detects tags in conversations
- **Identity Ledger** (`src/services/identity_ledger.py`): Audit trail for trust updates

## References

- Trust state: `data/persona/trust.json`
- Configuration: `config/settings.py`
- Integration: `app.py` (pending)
- Tests: `tests/test_provenance_trust.py` (pending)

---

**This system transforms provenance weighting from hardcoded heuristics into learned trust based on Astra's own outcomes.**
