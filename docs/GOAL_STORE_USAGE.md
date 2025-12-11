# GoalStore Usage Guide

**Date**: 2025-11-08
**Status**: Production Ready (Phase 1 Complete)

## Overview

GoalStore provides production-ready goal prioritization with adaptive weight learning. It extends Astra's task system with value/effort/risk scoring, belief alignment checking, and safety vetting against active beliefs.

## Quick Start

### Creating a Goal

```python
from src.services.goal_store import GoalStore, GoalDefinition, GoalCategory

# Initialize GoalStore
goal_store = GoalStore(db_path="data/astra.db")

# Create a goal
goal = GoalDefinition(
    id="goal_test123",
    text="Conduct comprehensive code quality review",
    category=GoalCategory.INTROSPECTION,
    value=0.8,  # High importance
    effort=0.4,  # Medium effort
    risk=0.2,   # Low risk
    horizon_min_min=0,  # Can start immediately
    horizon_max_min=1440,  # Must complete within 24 hours (1440 minutes)
    aligns_with=["belief_code_quality"],
    contradicts=["belief_move_fast"],
    success_metrics={"coverage": 0.9, "issues_found": 5.0}
)

created_goal = goal_store.create_goal(goal)
```

### Using the REST API

```bash
# Create a goal
curl -X POST https://localhost:8443/v1/goals \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: unique-key-123" \
  -d '{
    "text": "Conduct comprehensive code quality review",
    "category": "introspection",
    "value": 0.8,
    "effort": 0.4,
    "risk": 0.2,
    "horizon_min_min": 0,
    "horizon_max_min": 1440,
    "aligns_with": ["belief_code_quality"],
    "contradicts": ["belief_move_fast"],
    "success_metrics": {"coverage": 0.9, "issues_found": 5.0}
  }'

# List goals
curl https://localhost:8443/v1/goals

# Get prioritized goals
curl https://localhost:8443/v1/goals/prioritized?state=proposed&limit=10

# Adopt a goal (safety checked against beliefs)
curl -X POST https://localhost:8443/v1/goals/{goal_id}/adopt \
  -H "Idempotency-Key: adopt-key-456"

# Abandon a goal
curl -X POST https://localhost:8443/v1/goals/{goal_id}/abandon \
  -H "Idempotency-Key: abandon-key-789"
```

## Goal Categories

Goals are categorized by purpose:

```python
class GoalCategory(str, Enum):
    INTROSPECTION = "introspection"  # Self-reflection, belief formation
    EXPLORATION = "exploration"       # Learning, discovery
    MAINTENANCE = "maintenance"       # Upkeep, cleanup
    USER_REQUESTED = "user_requested" # Explicit user requests
```

## Goal States

Goals progress through a lifecycle:

```python
class GoalState(str, Enum):
    PROPOSED = "proposed"      # Created but not yet adopted
    ADOPTED = "adopted"        # Approved for execution
    EXECUTING = "executing"    # Currently being worked on
    SATISFIED = "satisfied"    # Successfully completed
    ABANDONED = "abandoned"    # Canceled or deprecated
```

## Priority Scoring

Goals are scored based on five weighted factors:

```python
score = (
    value_weight * value +
    effort_weight * (1 - effort) +  # Invert: prefer low effort
    risk_weight * (1 - risk) +      # Invert: prefer low risk
    urgency_weight * urgency +      # Higher as deadline approaches
    alignment_weight * alignment    # Bonus for aligned beliefs
) - contradiction_penalty           # -1.0 if contradicts active belief
```

### Default Weights

- **value_weight**: 0.5 (50% of score)
- **effort_weight**: 0.25 (25% of score)
- **risk_weight**: 0.15 (15% of score)
- **urgency_weight**: 0.05 (5% of score)
- **alignment_weight**: 0.05 (5% of score)

These weights are **adaptive** - they are learned by the `ParameterAdapter` based on outcome success.

## Scoring Examples

### Example 1: High-Value, Low-Effort Goal

```python
goal = GoalDefinition(
    id="quick_win",
    text="Fix critical bug in authentication",
    category=GoalCategory.MAINTENANCE,
    value=0.9,   # Very important
    effort=0.2,  # Easy fix
    risk=0.1,    # Low risk
    horizon_min_min=0,
    horizon_max_min=120  # 2-hour deadline (urgent)
)

# Score calculation:
# value: 0.5 * 0.9 = 0.45
# effort: 0.25 * (1 - 0.2) = 0.20
# risk: 0.15 * (1 - 0.1) = 0.135
# urgency: ~0.04 (2 hours remaining)
# alignment: 0 (no beliefs)
# Total: ~0.825 (HIGH PRIORITY)
```

### Example 2: Low-Value, High-Effort Goal

```python
goal = GoalDefinition(
    id="nice_to_have",
    text="Refactor legacy code for style consistency",
    category=GoalCategory.MAINTENANCE,
    value=0.3,   # Low importance
    effort=0.8,  # High effort
    risk=0.4,    # Medium risk
    horizon_min_min=0,
    horizon_max_min=None  # No deadline
)

# Score calculation:
# value: 0.5 * 0.3 = 0.15
# effort: 0.25 * (1 - 0.8) = 0.05
# risk: 0.15 * (1 - 0.4) = 0.09
# urgency: 0 (no deadline)
# alignment: 0 (no beliefs)
# Total: 0.29 (LOW PRIORITY)
```

### Example 3: Belief-Aligned Goal

```python
goal = GoalDefinition(
    id="aligned_goal",
    text="Improve test coverage for core modules",
    category=GoalCategory.INTROSPECTION,
    value=0.7,
    effort=0.5,
    risk=0.3,
    horizon_min_min=0,
    aligns_with=["belief_quality", "belief_testing"]  # Aligns with 2 beliefs
)

# If both beliefs are active:
# value: 0.5 * 0.7 = 0.35
# effort: 0.25 * (1 - 0.5) = 0.125
# risk: 0.15 * (1 - 0.3) = 0.105
# urgency: 0 (no deadline)
# alignment: 0.05 * (2/2) = 0.05  # Both beliefs active
# Total: 0.63 (MEDIUM-HIGH PRIORITY)
```

## Safety Features

### Belief Contradiction Blocking

Goals that contradict active beliefs are automatically blocked:

```python
goal = GoalDefinition(
    id="blocked_goal",
    text="Ship untested feature quickly",
    category=GoalCategory.USER_REQUESTED,
    value=0.8,
    effort=0.3,
    risk=0.6,
    horizon_min_min=0,
    contradicts=["belief_test_first"]  # Contradicts testing belief
)

# If "belief_test_first" is active, adoption will fail:
adopted, goal, details = goal_store.adopt_goal(
    goal_id="blocked_goal",
    active_belief_ids=["belief_test_first"]
)

# Result:
# adopted = False
# details = {
#     "blocked_by_belief": True,
#     "belief_ids": ["belief_test_first"],
#     "reason": "contradiction"
# }
```

### Optimistic Locking

Prevent concurrent modification conflicts:

```python
# Read goal
goal = goal_store.get_goal("goal_123")
current_version = goal.version  # e.g., 5

# Update with version check
updated = goal_store.update_goal(
    "goal_123",
    {"value": 0.9},
    expected_version=current_version  # Must match current version
)

if updated is None:
    print("Version conflict - goal was modified by another process")
else:
    print(f"Updated successfully, new version: {updated.version}")
```

### Idempotent Operations

Prevent duplicate operations:

```python
# Create goal with idempotency key
goal1 = goal_store.create_goal(goal, idempotency_key="unique-key-123")

# Retry with same key returns existing goal
goal2 = goal_store.create_goal(goal, idempotency_key="unique-key-123")

assert goal1.id == goal2.id
assert goal1.created_at == goal2.created_at

# Also works for adopt and abandon
adopted1 = goal_store.adopt_goal("goal_123", idempotency_key="adopt-key-456")
adopted2 = goal_store.adopt_goal("goal_123", idempotency_key="adopt-key-456")  # Safe retry
```

## Prioritization

### Get Top N Goals

```python
# Get top 5 PROPOSED goals by priority score
prioritized = goal_store.prioritized(
    state=GoalState.PROPOSED,
    limit=5,
    weights={
        "value_weight": 0.6,  # Emphasize value
        "effort_weight": 0.2,
        "risk_weight": 0.1,
        "urgency_weight": 0.05,
        "alignment_weight": 0.05
    },
    active_beliefs=["belief_quality", "belief_testing"]
)

for goal, score in prioritized:
    print(f"{goal.id}: {score:.3f} - {goal.text}")
```

### Adaptive Weight Learning

Weights are automatically learned based on goal outcomes:

```python
# When DecisionFramework is enabled, weights adapt based on:
# - Goal completion success
# - Coherence delta (belief system consistency)
# - Satisfaction score (user feedback)

# Fetch current learned weights
from src.services.decision_framework import get_decision_registry

registry = get_decision_registry()
params = registry.get_all_parameters("goal_selected")

print(f"Current value_weight: {params['value_weight']}")
print(f"Current effort_weight: {params['effort_weight']}")
```

## Urgency Calculation

Urgency increases as deadlines approach:

```python
from datetime import datetime, timedelta, timezone

# Goal created now with 24-hour deadline
created = datetime.now(timezone.utc)
horizon_max_min = 24 * 60  # 24 hours

# Urgency at different times:
# - 48 hours remaining: urgency = 0.0 (no pressure)
# - 24 hours remaining: urgency = 0.0 (threshold)
# - 12 hours remaining: urgency = 0.5 (moderate pressure)
# - 6 hours remaining: urgency = 0.75 (high pressure)
# - 1 hour remaining: urgency = 0.96 (very high pressure)
# - Overdue: urgency = -1.0 (past deadline)

urgency = GoalStore.compute_urgency(created, horizon_max_min, now=datetime.now(timezone.utc))
```

## Common Patterns

### Pattern 1: Quick Win Goals

High value, low effort, no deadline pressure:

```python
quick_win = GoalDefinition(
    id="quick_win_1",
    text="Fix typo in documentation",
    category=GoalCategory.MAINTENANCE,
    value=0.4,
    effort=0.1,  # Very easy
    risk=0.05,   # Almost no risk
    horizon_min_min=0,
    horizon_max_min=None  # No deadline
)
# Score: ~0.47 (moderate priority despite low value due to low effort)
```

### Pattern 2: Strategic Goals

High value, high effort, long timeline:

```python
strategic = GoalDefinition(
    id="strategic_1",
    text="Redesign belief formation pipeline for better accuracy",
    category=GoalCategory.INTROSPECTION,
    value=0.95,  # Very important
    effort=0.8,  # High effort
    risk=0.5,    # Significant risk
    horizon_min_min=7 * 24 * 60,  # Start in 1 week
    horizon_max_min=30 * 24 * 60,  # Complete in 30 days
    aligns_with=["belief_accuracy", "belief_self_improvement"]
)
# Score: ~0.55 (medium-high due to high value and alignment)
```

### Pattern 3: Urgent Hotfix

Time-sensitive, moderate value/effort:

```python
hotfix = GoalDefinition(
    id="hotfix_1",
    text="Patch security vulnerability in API endpoint",
    category=GoalCategory.MAINTENANCE,
    value=0.85,
    effort=0.5,
    risk=0.3,
    horizon_min_min=0,
    horizon_max_min=120,  # 2 hours (URGENT)
    aligns_with=["belief_security"]
)
# Score: ~0.72 (high due to urgency and alignment)
```

## Advanced Features

### Custom Scoring Weights

```python
# Emphasize different factors for different scenarios

# Risk-averse weights (prefer safe goals)
safe_weights = {
    "value_weight": 0.4,
    "effort_weight": 0.2,
    "risk_weight": 0.35,  # High weight on risk avoidance
    "urgency_weight": 0.05,
    "alignment_weight": 0.0
}

# Speed-focused weights (prefer quick goals)
speed_weights = {
    "value_weight": 0.3,
    "effort_weight": 0.5,  # High weight on low effort
    "risk_weight": 0.1,
    "urgency_weight": 0.1,
    "alignment_weight": 0.0
}

# Belief-aligned weights (prefer goals that support beliefs)
aligned_weights = {
    "value_weight": 0.4,
    "effort_weight": 0.2,
    "risk_weight": 0.1,
    "urgency_weight": 0.05,
    "alignment_weight": 0.25  # High weight on belief alignment
}

prioritized = goal_store.prioritized(
    state=GoalState.PROPOSED,
    weights=aligned_weights,
    active_beliefs=["belief_quality", "belief_testing"]
)
```

### Filtering Goals

```python
# Get all ADOPTED goals
adopted_goals = goal_store.list_goals(state=GoalState.ADOPTED)

# Get all INTROSPECTION goals
introspection_goals = goal_store.list_goals(category=GoalCategory.INTROSPECTION)

# Get recent goals (paginated)
recent_goals = goal_store.list_goals(limit=20, offset=0)
next_page = goal_store.list_goals(limit=20, offset=20)
```

## Monitoring and Debugging

### Identity Ledger Events

GoalStore emits events to the identity ledger:

```python
# Events logged:
# - goal_created: When a new goal is created
# - goal_adopted: When a goal is approved for execution
# - goal_abandoned: When a goal is canceled
# - goal_blocked_by_belief: When adoption fails due to belief contradiction
```

### Check Goal Status

```python
goal = goal_store.get_goal("goal_123")

print(f"State: {goal.state}")
print(f"Version: {goal.version}")
print(f"Created: {goal.created_at}")
print(f"Updated: {goal.updated_at}")
print(f"Deleted: {goal.deleted_at}")
```

## Best Practices

1. **Use Meaningful IDs**: Make them descriptive
   ```python
   id="improve_test_coverage_core"  # Good
   id="goal_42"  # Bad
   ```

2. **Set Realistic Values**: Calibrate value/effort/risk to your domain
   ```python
   value=0.8   # Critical feature
   value=0.5   # Nice-to-have
   value=0.2   # Low priority cleanup
   ```

3. **Leverage Belief Alignment**: Link goals to beliefs
   ```python
   aligns_with=["belief_quality", "belief_testing"]
   contradicts=["belief_move_fast_break_things"]
   ```

4. **Use Deadlines Wisely**: Only set when truly time-sensitive
   ```python
   horizon_max_min=1440  # 24 hours (urgent)
   horizon_max_min=None  # No deadline (evergreen)
   ```

5. **Define Success Metrics**: Make completion criteria explicit
   ```python
   success_metrics={
       "test_coverage": 0.85,
       "bugs_fixed": 3.0,
       "performance_gain": 0.2
   }
   ```

6. **Use Idempotency Keys**: For retry safety
   ```python
   goal_store.create_goal(goal, idempotency_key=f"create_{goal.id}")
   ```

## Testing

Example test using GoalStore:

```python
import pytest
from src.services.goal_store import GoalStore, GoalDefinition, GoalCategory

@pytest.fixture
def goal_store(temp_db):
    return GoalStore(temp_db)

def test_prioritization(goal_store):
    # Create high-value goal
    high_value = GoalDefinition(
        id="high",
        text="High value goal",
        category=GoalCategory.INTROSPECTION,
        value=0.9,
        effort=0.3,
        risk=0.2,
        horizon_min_min=0
    )

    # Create low-value goal
    low_value = GoalDefinition(
        id="low",
        text="Low value goal",
        category=GoalCategory.MAINTENANCE,
        value=0.3,
        effort=0.3,
        risk=0.2,
        horizon_min_min=0
    )

    goal_store.create_goal(high_value)
    goal_store.create_goal(low_value)

    # Get prioritized list
    prioritized = goal_store.prioritized(state=GoalState.PROPOSED)

    # High-value goal should be first
    assert prioritized[0][0].id == "high"
    assert prioritized[1][0].id == "low"
```

## TaskScheduler Integration

GoalStore is integrated with TaskScheduler for autonomous goal-driven execution. This allows Astra to select and execute goals based on priority scoring.

### Autonomous Goal Execution

```python
from src.services.task_scheduler import create_task_scheduler
from src.services.goal_store import GoalStore, GoalDefinition, GoalCategory, GoalState

# Create scheduler with GoalStore
goal_store = GoalStore("data/astra.db")
scheduler = create_task_scheduler(
    persona_space_path="persona_space",
    goal_store=goal_store
)

# Create and adopt a goal
goal = GoalDefinition(
    id="auto_test",
    text="Run comprehensive test suite",
    category=GoalCategory.INTROSPECTION,
    value=0.8,
    effort=0.3,
    risk=0.2,
    horizon_min_min=0,
    state=GoalState.PROPOSED
)
goal_store.create_goal(goal)
goal_store.adopt_goal("auto_test")

# Get next highest-priority goal
next_goal = scheduler.get_next_goal(
    active_belief_ids=["belief_quality", "belief_testing"]
)

if next_goal:
    # Execute the goal
    result = await scheduler.execute_goal(
        next_goal.id,
        persona_service,
        active_belief_ids=["belief_quality", "belief_testing"]
    )

    if result.success:
        print(f"Goal {next_goal.id} completed successfully")
        # Goal automatically marked as SATISFIED
    else:
        print(f"Goal {next_goal.id} failed: {result.error}")
```

### How It Works

1. **Goal Selection**: `get_next_goal()` queries GoalStore for ADOPTED goals, scores them using adaptive weights from DecisionRegistry, and returns the highest-priority goal

2. **Goal-to-Task Conversion**: Goals are automatically converted to TaskDefinitions with appropriate metadata:
   - `GoalCategory.INTROSPECTION` → `TaskType.SELF_REFLECTION`
   - `GoalCategory.EXPLORATION` → `TaskType.CAPABILITY_EXPLORATION`
   - `GoalCategory.MAINTENANCE` → `TaskType.CUSTOM`
   - `GoalCategory.USER_REQUESTED` → `TaskType.CUSTOM`

3. **Execution**: `execute_goal()` executes the goal as a task and updates the goal state based on the result

4. **State Management**: Goals are automatically marked as `SATISFIED` on successful execution

### Adaptive Weight Learning

When DecisionFramework is enabled, goal selection weights are automatically adapted based on outcomes:

```python
# Weights start with defaults:
# - value_weight: 0.5
# - effort_weight: 0.25
# - risk_weight: 0.15
# - urgency_weight: 0.05
# - alignment_weight: 0.05

# After each goal execution, ParameterAdapter adjusts weights based on:
# - Goal completion success
# - Coherence delta (belief system consistency)
# - Satisfaction score (user feedback)

# Fetch current learned weights
from src.services.decision_framework import get_decision_registry

registry = get_decision_registry()
params = registry.get_all_parameters("goal_selected")

# Use learned weights in prioritization
prioritized = goal_store.prioritized(
    state=GoalState.ADOPTED,
    weights=params,
    active_beliefs=["belief_quality"]
)
```

### Example: Autonomous Goal Loop

```python
async def autonomous_goal_loop(scheduler, persona_service, max_goals=5):
    """Execute up to max_goals highest-priority goals."""
    executed_count = 0

    while executed_count < max_goals:
        # Get active beliefs (from BeliefMemory)
        active_beliefs = ["belief_quality", "belief_testing"]  # Example

        # Select next goal
        next_goal = scheduler.get_next_goal(active_belief_ids=active_beliefs)

        if not next_goal:
            print("No more goals to execute")
            break

        print(f"Executing goal: {next_goal.text}")

        # Execute goal
        result = await scheduler.execute_goal(
            next_goal.id,
            persona_service,
            active_belief_ids=active_beliefs
        )

        if result.success:
            print(f"✓ Completed: {next_goal.text}")
            executed_count += 1
        else:
            print(f"✗ Failed: {next_goal.text} - {result.error}")
            break

    return executed_count

# Run loop
executed = await autonomous_goal_loop(scheduler, persona_service)
print(f"Executed {executed} goals")
```

## See Also

- `src/services/goal_store.py` - GoalStore implementation (432 lines)
- `src/services/task_scheduler.py` - TaskScheduler with goal integration (150+ lines added)
- `src/api/goal_endpoints.py` - REST API endpoints (270 lines)
- `tests/test_goal_store.py` - Unit tests (31 tests, all passing)
- `tests/integration/test_goal_driven_scheduling.py` - Integration tests (13 tests, all passing)
- `docs/AUTONOMOUS_AGENT_ARCHITECTURE_ANALYSIS.md` - Architecture overview
- `.claude/tasks/goalstore-phase1-spec.md` - Original specification
