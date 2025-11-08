# Adaptive Decision Framework - Implementation Status

**Date**: November 6, 2025
**Branch**: feature/adaptive-decision-framework
**Status**: Phase 1-3 Complete ✅

## What Was Implemented

### 1. Core Decision Framework (`src/services/decision_framework.py`)

A meta-system for coordinating and learning decision parameters across Astra's subsystems.

**Key Features**:
- **Decision Registry**: SQLite-backed registry of all decision points
- **Parameter Management**: Tunable parameters with bounds and adaptation rates
- **Decision Recording**: Full context and pre-decision metrics captured
- **Outcome Tracking**: Links decisions to outcomes for learning
- **Parameter Adaptation**: Hooks for adjusting thresholds based on outcomes

**Database Schema**:
- `decision_registry` - Registered decision points and their parameters
- `decision_history` - Every decision made with context
- `parameter_adaptations` - History of parameter changes

### 2. Success Signal Evaluator (`src/services/success_signal_evaluator.py`)

Defines what "success" means for decisions and measures outcomes.

**Baselines** (current performance):
- Coherence: 0.70
- Dissonance: 0.20
- Satisfaction: 0.60

**Targets** (optimization goals):
- Coherence: 0.85
- Dissonance: 0.10
- Satisfaction: 0.80

**Success Score Computation**:
```python
score = w_coh * Δcoh_norm + w_dis * Δdis_norm + w_sat * Δsat_norm
```
- Score ∈ [-1, 1]
- +1 = perfect improvement
- 0 = no change
- -1 = degradation

**Integration Points**:
- Awareness Loop → coherence metrics
- Belief Consistency Checker → dissonance metrics
- Feedback Aggregator → satisfaction metrics

### 3. Abort Condition Monitor (`src/services/abort_condition_monitor.py`)

Safety system that halts autonomous decisions when degradation detected.

**Abort Conditions**:

1. **Dissonance Spike**: `dissonance > baseline + 3σ` over N ticks
2. **Coherence Drop**: `coherence < baseline - 2σ` over N ticks
3. **Satisfaction Collapse**: `negative_tags > 70%` over 24h
4. **Belief Runaway**: `>10 beliefs formed` in 1 hour

**Recovery Mechanism**:
- Requires 1 hour elapsed since abort
- Coherence back above baseline
- Dissonance back below baseline + 1σ
- Manual reset option via API

**State Tracking**:
- Coherence history buffer (100 samples)
- Dissonance history buffer (100 samples)
- Belief formation timestamps (100 samples)

### 4. Belief Gardener Integration (`src/services/belief_gardener_integration.py`)

Wires the belief gardener into the decision framework.

**Decision Points Registered**:

1. **belief_formation**
   - Parameters: `min_evidence`, `confidence_boost`
   - Success Metrics: coherence_delta, dissonance_delta
   - Context: category, evidence_count

2. **belief_promotion**
   - Parameters: `promotion_threshold`, `min_evidence_asserted`
   - Success Metrics: coherence_delta, user_validation, stability
   - Context: belief_confidence, feedback_quality

3. **belief_deprecation**
   - Parameters: `deprecation_threshold`
   - Success Metrics: coherence_delta, dissonance_reduction
   - Context: belief_age, contradiction_count

**Decision Flow**:
1. Check abort conditions
2. Get adaptive parameters from registry
3. Capture pre-decision metrics
4. Execute decision (form/promote/deprecate)
5. Record decision with context for evaluation

### 5. API Endpoints (`src/api/decision_endpoints.py`)

RESTful API for monitoring and managing the decision framework.

**Endpoints**:
- `GET /api/persona/decisions/registry` - View all registered decisions
- `GET /api/persona/decisions/history` - Query decision history
- `GET /api/persona/decisions/parameters` - Get current parameter values
- `POST /api/persona/decisions/parameters` - Update parameter (admin)
- `GET /api/persona/decisions/success_signals` - View baselines/targets
- `GET /api/persona/decisions/abort_status` - Check abort conditions
- `POST /api/persona/decisions/abort_status/reset` - Manual abort reset

### 6. Comprehensive Tests (`tests/test_decision_framework.py`)

**Test Coverage** (10 tests, all passing):
- Decision point registration
- Decision recording and retrieval
- Parameter updates with bounds checking
- Decision outcome evaluation
- Success score computation
- Abort condition monitoring
- Abort recovery mechanism
- Registry statistics

**Test Results**: ✅ 10/10 passed in 0.23s

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│          ADAPTIVE DECISION FRAMEWORK                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Decision Registry (SQLite)                              │
│  ├─ Decision points (formation, promotion, deprecation) │
│  ├─ Tunable parameters with bounds                      │
│  ├─ Decision history with context                       │
│  └─ Parameter adaptation log                            │
│                                                          │
│  Success Signal Evaluator                                │
│  ├─ Baselines: coherence, dissonance, satisfaction      │
│  ├─ Targets: optimization goals                         │
│  ├─ Success score: weighted delta computation           │
│  └─ Metric history tracking                             │
│                                                          │
│  Abort Condition Monitor                                 │
│  ├─ Dissonance spike detection                          │
│  ├─ Coherence drop detection                            │
│  ├─ Satisfaction collapse detection                     │
│  ├─ Belief runaway detection                            │
│  └─ Recovery mechanism (1h + metric stabilization)      │
│                                                          │
│  Belief Gardener Integration                             │
│  ├─ 3 decision points registered                        │
│  ├─ Pre-decision metric snapshots                       │
│  ├─ Adaptive threshold queries                          │
│  ├─ Decision recording for evaluation                   │
│  └─ Abort condition checks                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Key Innovations

1. **Outcome-Driven Parameters**: Thresholds adapt based on measured outcomes, not hardcoded values

2. **Multi-Metric Success**: Success defined by coherence, dissonance, and satisfaction together

3. **Abort Safety Rails**: Automatic circuit breakers prevent runaway autonomous behavior

4. **Full Auditability**: Every decision recorded with context and parameters used

5. **Context Awareness**: Decision context captured for future context-aware adaptation

## TODO Items Completed

✅ **Signal of success - Define baselines and targets**
- Baselines: coherence=0.70, dissonance=0.20, satisfaction=0.60
- Targets: coherence=0.85, dissonance=0.10, satisfaction=0.80
- Success score computation from component deltas

✅ **Abort Condition - Bind to measurable signals**
- Dissonance spike: >baseline + 3σ
- Coherence drop: <baseline - 2σ
- Satisfaction collapse: >70% negative tags
- Belief runaway: >10 beliefs/hour

✅ **Adaptive Decision Framework**
- Full decision registry with SQLite persistence
- Decision recording and outcome tracking
- Parameter management with bounds
- Integration with belief gardener
- Comprehensive test suite

## What's NOT Yet Done

### Phase 4: Parameter Adaptation Algorithm
- **Status**: Hooks in place, algorithm not implemented
- **What's Needed**: Gradient-free optimization to adjust parameters based on outcome scores
- **Files**: Would extend `src/services/decision_framework.py` with `ParameterAdapter` class

### Phase 5: Context-Aware Parameters
- **Status**: Context captured but not used for selection
- **What's Needed**: `ContextClassifier` to detect high-stakes vs exploratory mode
- **Files**: New `src/services/context_classifier.py`

### Phase 6: Full App Integration
- **Status**: Integration layer exists, not wired into app.py
- **What's Needed**: Replace standard `BeliefLifecycleManager` with `AdaptiveBeliefLifecycleManager`
- **Files**: Modify `app.py` initialization

### Identity Ledger Audit Logging
- **Status**: Not implemented
- **What's Needed**: Log decision events to identity ledger
- **Files**: Extend `src/services/belief_gardener_integration.py`

## Usage Example

```python
from src.services.belief_gardener_integration import create_adaptive_belief_lifecycle_manager

# Create fully-wired adaptive manager
manager = create_adaptive_belief_lifecycle_manager(
    belief_store=belief_store,
    raw_store=raw_store,
    config=gardener_config,
    feedback_aggregator=feedback_agg,
    awareness_loop=awareness,
    belief_consistency_checker=consistency
)

# Decision points automatically registered:
# - belief_formation
# - belief_promotion
# - belief_deprecation

# Use like standard BeliefLifecycleManager
belief_id, error = manager.form_belief_from_pattern(pattern)
# → Decision recorded for evaluation
# → Abort conditions checked
# → Adaptive thresholds used

promoted = manager.consider_promotion(belief_id, evidence_count=7)
# → Decision recorded for evaluation
# → Feedback threshold adaptive
# → Success metrics tracked
```

## API Usage Example

```bash
# View decision registry
curl https://172.239.66.45:8000/api/persona/decisions/registry

# Get decision history for belief promotion
curl https://172.239.66.45:8000/api/persona/decisions/history?decision_id=belief_promotion&limit=10

# Get current parameters
curl https://172.239.66.45:8000/api/persona/decisions/parameters?decision_id=belief_promotion

# Check abort status
curl https://172.239.66.45:8000/api/persona/decisions/abort_status

# View success signals
curl https://172.239.66.45:8000/api/persona/decisions/success_signals
```

## Files Created

### Core Framework
- `src/services/decision_framework.py` - Decision registry and tracking (546 lines)
- `src/services/success_signal_evaluator.py` - Success metrics (217 lines)
- `src/services/abort_condition_monitor.py` - Safety monitoring (341 lines)

### Integration
- `src/services/belief_gardener_integration.py` - Belief gardener wiring (444 lines)
- `src/api/decision_endpoints.py` - REST API endpoints (223 lines)

### Documentation & Tests
- `docs/ADAPTIVE_DECISION_FRAMEWORK.md` - Full specification (546 lines)
- `tests/test_decision_framework.py` - Comprehensive tests (375 lines)

**Total**: ~2,700 lines of code and documentation

## Next Steps

1. **Implement Parameter Adaptation Algorithm** (Phase 4)
   - Gradient-free optimization (e.g., epsilon-greedy)
   - Compute parameter adjustments from outcome success scores
   - Schedule adaptation runs (weekly)

2. **Build Context Classifier** (Phase 5)
   - Detect high-stakes vs exploratory conversations
   - Select parameter profiles based on context
   - Track context classification accuracy

3. **Wire into App.py** (Phase 6)
   - Replace standard belief gardener with adaptive version
   - Initialize success evaluator and abort monitor
   - Start background outcome evaluation task

4. **Add Identity Ledger Logging**
   - Log decision events: formation, promotion, deprecation
   - Log parameter adaptations
   - Log abort triggers and recoveries

5. **Build Monitoring Dashboard**
   - Visualize parameter evolution over time
   - Show success score trends
   - Display abort condition status

## Summary

The Adaptive Decision Framework transforms Astra from static threshold-based decisions to adaptive, outcome-driven decision-making. Three major TODO items completed:

1. ✅ Success signals defined with baselines and targets
2. ✅ Abort conditions bound to measurable signals
3. ✅ Adaptive decision framework built and tested

All core infrastructure is in place. Future work focuses on the adaptation algorithm and full app integration.
