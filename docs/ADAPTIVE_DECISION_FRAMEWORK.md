# Adaptive Decision Framework

**Status**: Design Phase | **Branch**: feature/adaptive-decision-framework

## Overview

The Adaptive Decision Framework is a meta-system that coordinates decision-making across Astra's subsystems and learns optimal parameters from outcomes. Instead of hardcoded thresholds, it dynamically adjusts decision criteria based on measured success signals.

## Problem Statement

Currently, Astra has numerous hardcoded decision thresholds scattered across subsystems:

- **Belief Gardener**: `promotion_threshold=0.2`, `deprecation_threshold=0.30`, `max_auto_confidence=0.85`
- **Feedback Aggregator**: `circuit_breaker_threshold=0.6`, `window_hours=24`
- **Awareness Loop**: `coherence_threshold`, `drift_alert_level`
- **Outcome Evaluator**: `reward_weights=[0.4, 0.2, 0.2, 0.2]`

These thresholds were chosen arbitrarily and don't adapt to:
- Individual user interaction patterns
- System performance over time
- Environmental context (high-stakes vs exploratory conversations)
- Measured outcomes (did the decision improve coherence?)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│          ADAPTIVE DECISION FRAMEWORK                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Decision Registry                                       │
│  • Tracks all decision points across subsystems         │
│  • Maps decision → parameters → outcomes                │
│  • Stores decision history with context                 │
│                                                          │
│  Parameter Adapter                                       │
│  • Learns optimal thresholds from outcomes              │
│  • Gradient-free optimization (Bayesian, bandit)        │
│  • Per-context parameter tuning                         │
│                                                          │
│  Success Signal Evaluator                                │
│  • Defines baselines and targets                        │
│  • Measures: coherence, dissonance, satisfaction        │
│  • Computes decision quality scores                     │
│                                                          │
│  Abort Condition Monitor                                 │
│  • Watches for degradation signals                      │
│  • Triggers circuit breakers                            │
│  • Prevents runaway decisions                           │
│                                                          │
│  Context Classifier                                      │
│  • High-stakes vs exploratory                           │
│  • User engagement level                                │
│  • System confidence state                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Decision Point Abstraction

Every decision in Astra should be registered and tracked:

```python
@dataclass
class DecisionPoint:
    """A registered decision point in the system."""
    decision_id: str  # "belief_promotion", "belief_deprecation", etc.
    subsystem: str  # "belief_gardener", "feedback_aggregator", etc.
    description: str  # Human-readable description
    parameters: Dict[str, Parameter]  # Tunable parameters
    success_metrics: List[str]  # Which metrics define success
    context_features: List[str]  # What context matters

@dataclass
class Parameter:
    """A tunable parameter for a decision."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    adaptation_rate: float  # How quickly to adapt
```

### 2. Decision Registry

Central registry of all decision points:

```python
class DecisionRegistry:
    """Registry of all decision points across Astra's subsystems."""

    def register_decision(
        self,
        decision_id: str,
        subsystem: str,
        parameters: Dict[str, Parameter],
        success_metrics: List[str]
    ) -> None:
        """Register a new decision point."""

    def get_parameter(self, decision_id: str, param_name: str) -> float:
        """Get current value for a parameter."""

    def record_decision(
        self,
        decision_id: str,
        context: Dict[str, Any],
        parameters_used: Dict[str, float],
        outcome: Any
    ) -> str:
        """Record a decision for later evaluation."""

    def update_parameter(
        self,
        decision_id: str,
        param_name: str,
        new_value: float
    ) -> None:
        """Update a parameter value."""
```

### 3. Success Signal Evaluator

Defines what "success" means for each decision:

```python
class SuccessSignalEvaluator:
    """Evaluates whether decisions led to successful outcomes."""

    # Baselines (measured during cold start)
    baseline_coherence: float = 0.7  # Average coherence
    baseline_dissonance: float = 0.2  # Average contradiction rate
    baseline_satisfaction: float = 0.6  # Implicit user satisfaction

    # Targets (what we're optimizing for)
    target_coherence: float = 0.85
    target_dissonance: float = 0.1
    target_satisfaction: float = 0.8

    def evaluate_decision_outcome(
        self,
        decision_record_id: str,
        evaluation_window_hours: int = 24
    ) -> DecisionOutcome:
        """
        Evaluate a decision's outcome after a time window.

        Returns:
            DecisionOutcome with success score and component metrics
        """

    def compute_success_score(
        self,
        coherence_delta: float,
        dissonance_delta: float,
        satisfaction_delta: float
    ) -> float:
        """
        Compute overall success score from component deltas.

        Score ∈ [-1, 1]:
        - +1: Perfect improvement on all metrics
        - 0: No change from baseline
        - -1: Degradation on all metrics
        """

@dataclass
class DecisionOutcome:
    """Outcome of a decision after evaluation period."""
    decision_record_id: str
    success_score: float  # [-1, 1]
    coherence_delta: float
    dissonance_delta: float
    satisfaction_delta: float
    aborted: bool
    abort_reason: Optional[str]
```

### 4. Parameter Adapter

Learns optimal parameter values from outcomes:

```python
class ParameterAdapter:
    """Adapts decision parameters based on observed outcomes."""

    def __init__(
        self,
        decision_registry: DecisionRegistry,
        success_evaluator: SuccessSignalEvaluator
    ):
        self.registry = decision_registry
        self.evaluator = success_evaluator
        self.adaptation_history = []

    def adapt_parameters(
        self,
        decision_id: str,
        recent_outcomes: List[DecisionOutcome]
    ) -> Dict[str, float]:
        """
        Adapt parameters based on recent outcomes.

        Uses gradient-free optimization:
        - If recent decisions → positive outcomes: adjust in same direction
        - If recent decisions → negative outcomes: adjust in opposite direction
        - If mixed outcomes: small exploratory adjustments

        Returns:
            Updated parameter values
        """

    def explore_parameter_space(
        self,
        decision_id: str,
        param_name: str,
        exploration_rate: float = 0.1
    ) -> float:
        """Add controlled exploration to avoid local optima."""
```

### 5. Abort Condition Monitor

Watches for dangerous degradation patterns:

```python
class AbortConditionMonitor:
    """Monitors for conditions that should halt autonomous decisions."""

    def __init__(
        self,
        awareness_loop: AwarenessLoop,
        belief_consistency_checker: BeliefConsistencyChecker,
        success_evaluator: SuccessSignalEvaluator
    ):
        self.awareness = awareness_loop
        self.consistency = belief_consistency_checker
        self.evaluator = success_evaluator

    def check_abort_conditions(self) -> Tuple[bool, Optional[str]]:
        """
        Check if any abort conditions are triggered.

        Returns:
            (should_abort, reason)
        """

    # Abort condition: Rising dissonance
    def check_dissonance_spike(self, window_ticks: int = 10) -> bool:
        """
        Abort if contradiction count rising rapidly.

        Condition: dissonance > baseline + 3σ over N ticks
        """

    # Abort condition: Coherence drop
    def check_coherence_drop(self, window_ticks: int = 10) -> bool:
        """
        Abort if coherence dropping significantly.

        Condition: coherence < baseline - 2σ over N ticks
        """

    # Abort condition: Satisfaction collapse
    def check_satisfaction_collapse(self) -> bool:
        """
        Abort if user satisfaction plummeting.

        Condition: explicit negative tags > 70% over 24h
        """

    # Abort condition: Runaway belief formation
    def check_belief_runaway(self) -> bool:
        """
        Abort if beliefs forming too rapidly.

        Condition: >10 beliefs formed in 1 hour
        """
```

### 6. Context Classifier

Determines what context the system is in:

```python
class ContextClassifier:
    """Classifies current context to select appropriate parameters."""

    def classify_conversation_context(
        self,
        recent_messages: List[ExperienceModel]
    ) -> ConversationContext:
        """
        Classify current conversation context.

        Returns:
            Context classification affecting parameter selection
        """

@dataclass
class ConversationContext:
    """Context classification for parameter selection."""
    stakes: Literal["high", "medium", "low"]  # How important is accuracy?
    engagement: Literal["deep", "moderate", "casual"]  # User engagement level
    system_confidence: Literal["high", "medium", "low"]  # Astra's confidence
    exploration_mode: bool  # Is system in exploratory learning mode?
```

## Decision Points to Register

### Belief Lifecycle Decisions

1. **belief_formation**
   - Parameters: `min_evidence_tentative`, `confidence_threshold`
   - Success metrics: coherence_delta, dissonance_delta
   - Context: engagement_level, belief_domain

2. **belief_promotion**
   - Parameters: `promotion_threshold`, `min_evidence_asserted`
   - Success metrics: coherence_delta, user_validation, stability
   - Context: belief_confidence, feedback_quality

3. **belief_deprecation**
   - Parameters: `deprecation_threshold`, `negative_weight`
   - Success metrics: coherence_delta, dissonance_reduction
   - Context: belief_age, contradiction_count

### Feedback Processing Decisions

4. **tag_weighting**
   - Parameters: `g_trust`, `g_conviction`, `g_align`
   - Success metrics: outcome_correlation
   - Context: actor, alignment_quality

5. **circuit_breaker**
   - Parameters: `negative_threshold`, `recovery_time`
   - Success metrics: false_positive_rate, missed_degradation_rate
   - Context: dissonance_level, recent_changes

### Memory Retrieval Decisions

6. **memory_relevance_threshold**
   - Parameters: `similarity_threshold`, `recency_weight`
   - Success metrics: response_quality, user_satisfaction
   - Context: query_type, conversation_depth

## Implementation Plan

### Phase 1: Registry and Tracking

- [ ] Create `src/services/decision_framework.py`
- [ ] Implement `DecisionRegistry` with SQLite persistence
- [ ] Add decision recording to existing subsystems
- [ ] Create decision history schema
- [ ] Add telemetry endpoints

### Phase 2: Success Signal Definition

- [ ] Implement `SuccessSignalEvaluator`
- [ ] Define baseline measurement period (7 days)
- [ ] Create success score computation
- [ ] Wire into outcome evaluator
- [ ] Add success signal telemetry

### Phase 3: Abort Conditions

- [ ] Implement `AbortConditionMonitor`
- [ ] Wire into belief gardener, feedback aggregator
- [ ] Add abort logging to identity ledger
- [ ] Create recovery procedures
- [ ] Test abort triggers

### Phase 4: Parameter Adaptation

- [ ] Implement `ParameterAdapter` with gradient-free optimization
- [ ] Add exploration strategy
- [ ] Create adaptation scheduling (weekly)
- [ ] Add adaptation logging
- [ ] Test on simulated outcomes

### Phase 5: Context-Aware Parameters

- [ ] Implement `ContextClassifier`
- [ ] Create per-context parameter profiles
- [ ] Add context switching logic
- [ ] Test context detection
- [ ] Validate parameter selection

### Phase 6: Integration

- [ ] Wire decision framework into all registered decision points
- [ ] Replace hardcoded thresholds with framework calls
- [ ] Add decision audit trail
- [ ] Create dashboard for parameter monitoring
- [ ] Run A/B tests (adaptive vs static)

## Data Schema

### Decision Registry Table

```sql
CREATE TABLE decision_registry (
    decision_id TEXT PRIMARY KEY,
    subsystem TEXT NOT NULL,
    description TEXT,
    parameters JSON,  -- {param_name: {current, min, max, step}}
    success_metrics JSON,  -- [metric_name, ...]
    context_features JSON,  -- [feature_name, ...]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Decision History Table

```sql
CREATE TABLE decision_history (
    record_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    context JSON,  -- {feature: value}
    parameters_used JSON,  -- {param: value}
    outcome_snapshot JSON,  -- pre-decision metrics
    evaluated BOOLEAN DEFAULT FALSE,
    evaluation_timestamp TIMESTAMP,
    success_score REAL,
    outcome_details JSON,
    FOREIGN KEY (decision_id) REFERENCES decision_registry(decision_id)
);
```

### Parameter Adaptation Log

```sql
CREATE TABLE parameter_adaptations (
    adaptation_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    param_name TEXT NOT NULL,
    old_value REAL NOT NULL,
    new_value REAL NOT NULL,
    reason TEXT,  -- "positive_outcomes", "exploration", "abort_recovery"
    based_on_records JSON,  -- [record_id, ...]
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (decision_id) REFERENCES decision_registry(decision_id)
);
```

## Configuration

```bash
# Decision Framework
DECISION_FRAMEWORK_ENABLED=true

# Success Signal Baselines (learned during cold start)
BASELINE_COHERENCE=0.70
BASELINE_DISSONANCE=0.20
BASELINE_SATISFACTION=0.60

# Success Signal Targets
TARGET_COHERENCE=0.85
TARGET_DISSONANCE=0.10
TARGET_SATISFACTION=0.80

# Adaptation
ADAPTATION_INTERVAL_HOURS=168  # Weekly
ADAPTATION_MIN_SAMPLES=20  # Min decisions before adapting
EXPLORATION_RATE=0.10  # 10% random exploration

# Abort Conditions
ABORT_DISSONANCE_SIGMA=3.0  # Std devs above baseline
ABORT_COHERENCE_SIGMA=2.0  # Std devs below baseline
ABORT_NEGATIVE_TAG_THRESHOLD=0.70  # 70% negative tags
ABORT_BELIEF_RATE_LIMIT=10  # Max beliefs/hour
```

## API Endpoints

```python
# Get decision registry
GET /api/persona/decisions/registry

# Get decision history
GET /api/persona/decisions/history?decision_id={id}&limit=20

# Get current parameters
GET /api/persona/decisions/parameters?decision_id={id}

# Manually adjust parameter (admin)
POST /api/persona/decisions/parameters
{
    "decision_id": "belief_promotion",
    "param_name": "promotion_threshold",
    "new_value": 0.25,
    "reason": "manual_adjustment"
}

# Get success signal baselines/targets
GET /api/persona/decisions/success_signals

# Get abort condition status
GET /api/persona/decisions/abort_status

# Trigger parameter adaptation (admin)
POST /api/persona/decisions/adapt
{
    "decision_id": "belief_promotion",
    "force": true
}
```

## Testing Strategy

### Unit Tests

```python
def test_decision_registry():
    """Test decision point registration and parameter retrieval."""

def test_success_score_computation():
    """Test success score from component deltas."""

def test_parameter_adaptation():
    """Test parameter adjustment from outcomes."""

def test_abort_conditions():
    """Test each abort condition triggers correctly."""

def test_context_classification():
    """Test conversation context detection."""
```

### Integration Tests

```python
def test_decision_recording():
    """Test decision recorded in history during belief promotion."""

def test_outcome_evaluation():
    """Test decision outcome evaluation after 24h window."""

def test_parameter_update_propagation():
    """Test updated parameters used in next decision."""

def test_abort_halts_decisions():
    """Test abort condition stops belief gardener."""
```

### Scenario Tests

```python
def test_adaptive_learning():
    """
    Scenario: System learns optimal promotion threshold.
    1. Start with default threshold
    2. Record 50 promotion decisions
    3. Evaluate outcomes
    4. Verify threshold adapted toward better outcomes
    """

def test_abort_recovery():
    """
    Scenario: System aborts on dissonance spike, recovers.
    1. Inject contradictory beliefs rapidly
    2. Verify abort triggered
    3. Wait for recovery period
    4. Verify decisions resume with adjusted parameters
    """
```

## Success Criteria

After implementation:

1. ✅ **All decision points registered** → registry contains 6+ decision types
2. ✅ **Decisions tracked** → history table populated during operations
3. ✅ **Parameters adapt** → threshold values change based on outcomes
4. ✅ **Aborts work** → degradation triggers halt autonomous decisions
5. ✅ **Context-aware** → different parameters in high-stakes vs exploratory mode
6. ✅ **Auditable** → full decision trail in identity ledger
7. ✅ **Observable** → dashboard shows parameter evolution over time

## Related Systems

- **Outcome Evaluator** (`src/services/outcome_evaluator.py`): Provides outcome signals
- **Belief Gardener** (`src/services/belief_gardener.py`): Primary consumer of adaptive thresholds
- **Awareness Loop** (`src/services/awareness_loop.py`): Provides coherence metrics
- **Belief Consistency** (`src/services/belief_consistency_checker.py`): Provides dissonance metrics
- **Feedback Aggregator** (`src/services/feedback_aggregator_enhanced.py`): Provides satisfaction signals
- **Identity Ledger** (`src/services/identity_ledger.py`): Audit trail for decisions

## Future Enhancements

1. **Multi-armed bandit**: Use bandit algorithms for parameter exploration
2. **Bayesian optimization**: More sophisticated parameter search
3. **Transfer learning**: Apply learned parameters to similar decision types
4. **User-specific profiles**: Different parameters per user
5. **Meta-learning**: Learn how to learn (adaptation rate tuning)
6. **Hierarchical decisions**: Compose complex decisions from simple ones

---

**This framework transforms Astra from static threshold-based decisions to adaptive, outcome-driven decision-making.**
