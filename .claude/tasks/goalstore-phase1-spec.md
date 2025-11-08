# GoalStore - Phase 1 Specification

**Branch**: `feature/goal-store`
**Timeline**: 2-3 weeks
**Dependencies**: None (extends existing TaskScheduler)
**Priority**: HIGH - Foundation for HTN planner

---

## Overview

Upgrade Astra's task system from simple scheduled tasks to **goal-oriented prioritization** with value/effort/risk scoring. This transforms reactive scheduling into proactive goal pursuit.

## Current State

**TaskDefinition** (`src/services/task_scheduler.py:48-65`):
```python
@dataclass
class TaskDefinition:
    id: str
    name: str
    type: TaskType  # SELF_REFLECTION, GOAL_ASSESSMENT, etc.
    schedule: TaskSchedule  # MANUAL, HOURLY, DAILY, etc.
    prompt: str
    enabled: bool
    last_run: Optional[str]
    next_run: Optional[str]
    run_count: int
    metadata: Dict[str, Any]
```

**Problem**: No way to prioritize tasks beyond schedule. All tasks of same schedule type are equal.

---

## Target State

### GoalDefinition (extends TaskDefinition)

```python
@dataclass
class GoalDefinition:
    # Existing TaskDefinition fields
    id: str
    name: str
    type: TaskType
    prompt: str
    enabled: bool

    # NEW: Goal prioritization fields
    value: float  # [0.0, 1.0] How important is this goal?
    effort: float  # [0.0, 1.0] How much work is required? (0=easy, 1=hard)
    risk: float  # [0.0, 1.0] Probability of failure

    # NEW: Belief alignment
    aligns_with: List[str]  # Belief IDs this goal supports
    contradicts: List[str]  # Belief IDs this goal would violate

    # NEW: Temporal constraints
    horizon_min: int  # Earliest start (hours from now, 0=immediate)
    horizon_max: int  # Latest completion (hours from now, None=no deadline)

    # NEW: Success criteria
    success_metrics: Dict[str, float]  # metric_name -> target_value
    # Example: {"coherence_increase": 0.1, "new_beliefs": 3}

    # Computed fields
    priority: float  # Computed from value/effort/risk
    ready: bool  # Can execute now (horizon_min passed, deps satisfied)

    # Existing fields
    last_run: Optional[str]
    run_count: int
    metadata: Dict[str, Any]
```

### Priority Scoring Function

```python
def score_goal(goal: GoalDefinition, ctx: Context) -> float:
    """
    Compute goal priority for frontier selection.

    Higher score = execute sooner.
    """
    # Base weights (will be learned by ParameterAdapter)
    w_value = ctx.value_weight  # Default: 0.5
    w_effort = ctx.effort_weight  # Default: 0.3
    w_risk = ctx.risk_weight  # Default: 0.2

    # Normalize components to [0, 1]
    value_norm = goal.value
    effort_norm = 1.0 - goal.effort  # Invert: prefer low effort
    risk_norm = 1.0 - goal.risk  # Invert: prefer low risk

    # Weighted sum
    base_score = (
        w_value * value_norm +
        w_effort * effort_norm +
        w_risk * risk_norm
    )

    # Belief alignment bonus
    alignment_bonus = 0.0
    if goal.aligns_with:
        # Check if aligned beliefs are active and high confidence
        active_beliefs = [
            b for b in goal.aligns_with
            if belief_kernel.is_active(b) and belief_kernel.get_confidence(b) > 0.7
        ]
        if active_beliefs:
            alignment_bonus = 0.2 * (len(active_beliefs) / len(goal.aligns_with))

    # Belief contradiction penalty
    contradiction_penalty = 0.0
    if goal.contradicts:
        # Heavy penalty if goal contradicts any active belief
        contradicted = [
            b for b in goal.contradicts
            if belief_kernel.is_active(b)
        ]
        if contradicted:
            contradiction_penalty = -0.5  # Major penalty

    # Temporal urgency factor
    urgency = 0.0
    if goal.horizon_max is not None:
        hours_remaining = goal.horizon_max - time_since_created_hours
        if hours_remaining < 24:
            urgency = 0.1 * (1.0 - hours_remaining / 24.0)  # Boost as deadline approaches

    final_score = base_score + alignment_bonus + contradiction_penalty + urgency

    return clamp(final_score, 0.0, 1.0)
```

---

## Implementation Tasks

### Task 1: Extend Data Model (1-2 days)

**File**: `src/services/goal_store.py` (new)

1. Define `GoalDefinition` dataclass
2. Add JSON serialization/deserialization
3. Add validation:
   - `value`, `effort`, `risk` ∈ [0, 1]
   - `horizon_min` < `horizon_max`
   - `aligns_with` references valid belief IDs
4. Implement `compute_priority(goal, ctx)` method

**Test**: `tests/test_goal_definition.py`
- Validate field constraints
- Test priority scoring with various weights
- Test belief alignment bonus/penalty

### Task 2: Implement GoalStore (3-4 days)

**File**: `src/services/goal_store.py`

```python
class GoalStore:
    """
    Manages goals with priority-based selection.

    Wraps TaskScheduler and extends it with goal prioritization.
    """

    def __init__(
        self,
        belief_kernel,
        decision_registry,
        persona_space_path: str = "persona_space"
    ):
        self.belief_kernel = belief_kernel
        self.registry = decision_registry
        self.goals_file = Path(persona_space_path) / "goals" / "goals.json"
        self.goals: Dict[str, GoalDefinition] = {}
        self._load_goals()

    def add_goal(self, goal: GoalDefinition) -> None:
        """Add goal to store."""
        # Validate belief references
        for belief_id in goal.aligns_with + goal.contradicts:
            if not self.belief_kernel.belief_exists(belief_id):
                raise ValueError(f"Unknown belief: {belief_id}")

        self.goals[goal.id] = goal
        self._save_goals()

    def get_ready_goals(self, limit: int = 10) -> List[GoalDefinition]:
        """
        Get goals ready to execute, ordered by priority.

        Filters:
        - enabled=True
        - horizon_min <= now
        - horizon_max > now (if set)
        - No active belief contradictions
        """
        now_hours = time_since_init_hours()

        ready = [
            g for g in self.goals.values()
            if g.enabled
            and (g.horizon_min is None or now_hours >= g.horizon_min)
            and (g.horizon_max is None or now_hours < g.horizon_max)
            and not self._contradicts_beliefs(g)
        ]

        # Score and sort
        scored = [
            (g, score_goal(g, self._get_context()))
            for g in ready
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [g for g, score in scored[:limit]]

    def select_goal(self) -> Optional[GoalDefinition]:
        """
        Select the highest priority ready goal.

        This is called by the task scheduler to pick the next goal to execute.
        """
        ready = self.get_ready_goals(limit=1)
        if not ready:
            return None

        goal = ready[0]

        # Record decision
        self.registry.record_decision(
            decision_id="goal_selected",
            parameters={
                "value_weight": self.ctx.value_weight,
                "effort_weight": self.ctx.effort_weight,
                "risk_weight": self.ctx.risk_weight
            },
            context={
                "goal_id": goal.id,
                "goal_value": goal.value,
                "goal_effort": goal.effort,
                "goal_risk": goal.risk
            }
        )

        return goal

    def mark_completed(
        self,
        goal_id: str,
        outcome: GoalOutcome
    ) -> None:
        """Mark goal as completed and record outcome."""
        goal = self.goals.get(goal_id)
        if not goal:
            return

        goal.last_run = datetime.now(timezone.utc).isoformat()
        goal.run_count += 1

        # Check if success metrics achieved
        success = all(
            outcome.metrics.get(metric, 0) >= target
            for metric, target in goal.success_metrics.items()
        )

        # Record outcome for learning
        # (This will be used by ParameterAdapter to tune weights)
        self.registry.record_outcome(
            decision_record_id=outcome.decision_record_id,
            outcome=DecisionOutcome(
                success_score=1.0 if success else -0.5,
                coherence_delta=outcome.coherence_delta,
                dissonance_delta=outcome.dissonance_delta,
                satisfaction_delta=outcome.satisfaction_delta
            )
        )

        self._save_goals()

    def _contradicts_beliefs(self, goal: GoalDefinition) -> bool:
        """Check if goal contradicts any active beliefs."""
        for belief_id in goal.contradicts:
            belief = self.belief_kernel.get_belief(belief_id)
            if belief and belief.state == "active":
                return True
        return False

    def _get_context(self) -> Context:
        """Get current context for scoring."""
        # Fetch current value/effort/risk weights from DecisionRegistry
        # These weights are learned by ParameterAdapter
        params = self.registry.get_current_parameters("goal_selected")
        return Context(
            value_weight=params.get("value_weight", 0.5),
            effort_weight=params.get("effort_weight", 0.3),
            risk_weight=params.get("risk_weight", 0.2)
        )
```

**Test**: `tests/test_goal_store.py`
- Add/retrieve goals
- Filter ready goals (horizon, contradictions)
- Priority ordering
- Belief alignment/contradiction

### Task 3: Register Decision Point (1 day)

**File**: `src/services/goal_store.py`

Register `goal_selected` decision with DecisionRegistry:

```python
def register_goal_selection_decision(registry: DecisionRegistry):
    """Register goal selection as an adaptive decision point."""
    registry.register_decision(
        decision_id="goal_selected",
        subsystem="goal_store",
        description="Select which goal to execute next",
        parameters={
            "value_weight": Parameter(
                name="value_weight",
                current_value=0.5,
                min_value=0.0,
                max_value=1.0,
                step_size=0.05,
                adaptation_rate=0.1
            ),
            "effort_weight": Parameter(
                name="effort_weight",
                current_value=0.3,
                min_value=0.0,
                max_value=1.0,
                step_size=0.05,
                adaptation_rate=0.1
            ),
            "risk_weight": Parameter(
                name="risk_weight",
                current_value=0.2,
                min_value=0.0,
                max_value=1.0,
                step_size=0.05,
                adaptation_rate=0.1
            )
        },
        success_metrics=["coherence", "goal_completion", "satisfaction"]
    )
```

### Task 4: Migrate Existing Tasks (2-3 days)

**File**: `scripts/migrate_tasks_to_goals.py`

Convert existing TaskDefinitions to GoalDefinitions:

```python
# Manual estimates for existing tasks
TASK_ESTIMATES = {
    TaskType.SELF_REFLECTION: {
        "value": 0.8,  # High value - core to identity
        "effort": 0.3,  # Low effort
        "risk": 0.1,   # Low risk
        "aligns_with": ["core.ontological.consciousness"],
    },
    TaskType.GOAL_ASSESSMENT: {
        "value": 0.7,
        "effort": 0.4,
        "risk": 0.2,
        "aligns_with": [],
    },
    # ... etc
}
```

**Test**: Verify all existing tasks converted successfully

### Task 5: Integration with TaskScheduler (2-3 days)

**File**: `src/services/task_scheduler.py`

Modify `get_next_task()` to use GoalStore:

```python
class TaskScheduler:
    def __init__(self, ..., goal_store: Optional[GoalStore] = None):
        self.goal_store = goal_store
        # ... existing init

    async def get_next_task(self) -> Optional[TaskDefinition]:
        """Get next task to execute (now goal-driven)."""
        if self.goal_store:
            # Use goal prioritization
            goal = self.goal_store.select_goal()
            if goal:
                return self._goal_to_task(goal)

        # Fallback to schedule-based selection
        return self._get_scheduled_task()
```

**Test**: `tests/integration/test_goal_driven_scheduling.py`
- Verify high-priority goals execute first
- Verify belief contradictions block goals
- Verify horizon constraints respected

---

## Success Criteria

1. ✅ `GoalDefinition` dataclass with all fields
2. ✅ `GoalStore.get_ready_goals()` filters and ranks correctly
3. ✅ `score_goal()` computes priority from value/effort/risk
4. ✅ Belief alignment adds bonus, contradictions block
5. ✅ `goal_selected` decision registered and tracked
6. ✅ All existing tasks migrated to goals
7. ✅ TaskScheduler uses GoalStore when available
8. ✅ ParameterAdapter can learn value/effort/risk weights
9. ✅ Integration tests pass
10. ✅ Documentation updated

---

## Testing Strategy

### Unit Tests
- `tests/test_goal_definition.py` - Data model validation
- `tests/test_goal_store.py` - Store operations
- `tests/test_goal_scoring.py` - Priority computation

### Integration Tests
- `tests/integration/test_goal_driven_scheduling.py` - End-to-end goal selection
- `tests/integration/test_goal_learning.py` - Weight adaptation over time

### Manual Testing
1. Add 5 goals with different value/effort/risk
2. Verify highest-priority goal executes first
3. Add goal that contradicts belief
4. Verify goal is blocked
5. Run for 48 hours, verify weights adapt

---

## Migration Path

1. **Week 1**: Implement GoalDefinition + GoalStore (Tasks 1-2)
2. **Week 2**: Register decision + migrate tasks (Tasks 3-4)
3. **Week 3**: Integrate with scheduler + testing (Task 5)

---

## Future Enhancements (Phase 2+)

- Goal dependencies (prerequisite goals)
- Goal decomposition (break into subgoals)
- Dynamic goal generation (detect opportunities)
- Multi-objective optimization (Pareto frontier)
- User preference learning (personalized weights)

---

## Open Questions

1. Should goals auto-disable after success? Or allow recurring goals?
2. How to handle goals with conflicting `aligns_with` beliefs?
3. Should `effort` be estimated or learned from execution time?
4. What happens when all goals have contradictions? (Fallback behavior?)

---

*Spec ready for review and enhancement.*
