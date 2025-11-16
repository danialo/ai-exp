# Phase 0: Integration Wiring - Implementation Plan

**Branch**: `feature/phase0-integration-wiring`
**Timeline**: 1-2 weeks
**Goal**: Close the adaptive learning loop by wiring existing components together

---

## Work Groups (Safe to Program in Parallel)

### GROUP 1: Metrics Extension (Independent)
**Can be done in parallel with all other groups**

**Dependencies**: None - only extends existing OutcomeEvaluator
**Risk**: LOW - purely additive, no changes to existing logic

#### Tasks:
1. **Extend OutcomeEvaluator for task outcomes**
   - File: `src/services/outcome_evaluator.py`
   - Add `evaluate_task_outcome(task_id, task_execution_id, window_hours)`
   - Query RawStore for task execution experience
   - Compute coherence/dissonance/satisfaction deltas around task time
   - Return TaskOutcome object

2. **Add task execution metrics tracking**
   - File: `src/services/task_metrics.py` (new)
   - Track success rate by task type
   - Monitor task latency distribution
   - Alert on failure spike detection
   - Prometheus-compatible metrics

**Test files**:
- `tests/test_task_outcome_evaluator.py`
- `tests/test_task_metrics.py`

**Success criteria**:
- `evaluate_task_outcome()` returns TaskOutcome with valid deltas
- Metrics update on task completion
- No changes to existing outcome evaluation

---

### GROUP 2: Safety Integration (Independent)
**Can be done in parallel with Groups 1, 3, 4**

**Dependencies**: None - only reads from AbortConditionMonitor
**Risk**: LOW - adds checks but doesn't modify monitor logic

#### Tasks:
1. **Wire AbortConditionMonitor to TaskScheduler**
   - File: `src/services/task_scheduler.py`
   - Import AbortConditionMonitor
   - Check `monitor.should_abort()` before task execution
   - Block task if abort condition active
   - Log blocked task with abort reason

2. **Add safety check before task execution**
   - File: `src/services/task_scheduler.py`
   - Create `_check_safety_envelope(task)` method
   - Return (ok: bool, reason: Optional[str])
   - Log safety blocks to identity ledger

**Test files**:
- `tests/test_task_safety_integration.py`

**Success criteria**:
- Tasks blocked when `should_abort() == True`
- Blocked tasks logged with abort reason
- No tasks execute during abort state
- Monitor state doesn't affect non-task systems

---

### GROUP 3: Decision Framework Integration (Depends on Group 1)
**Can be done in parallel with Group 2**
**Should be done AFTER Group 1 completes**

**Dependencies**: Requires TaskOutcome from Group 1
**Risk**: MEDIUM - modifies task execution flow

#### Tasks:
1. **Wire DecisionFramework to TaskScheduler**
   - File: `src/services/task_scheduler.py`
   - Import DecisionRegistry
   - Record decision before task execution
   - Store decision_record_id with task execution

2. **Register task_selected decision point**
   - File: `src/services/task_scheduler.py`
   - Define TaskSelectionDecision
   - Parameters: task_type_preference, schedule_urgency
   - Success metrics: coherence, completion_rate
   - Register on scheduler initialization

3. **Link task outcomes to decision records**
   - File: `src/services/task_scheduler.py`
   - After task completion, evaluate outcome (Group 1)
   - Call `decision_registry.record_outcome(decision_record_id, outcome)`
   - Enable ParameterAdapter to learn from task decisions

**Test files**:
- `tests/test_decision_task_integration.py`

**Success criteria**:
- Every task execution creates decision record
- Decision records link to outcomes
- DecisionRegistry queryable by task type
- No impact on task execution if DecisionFramework disabled

---

### GROUP 4: Closed Loop Learning (Depends on Groups 1 and 3)
**Must be done LAST - requires all other groups**

**Dependencies**: Requires Groups 1 and 3 complete
**Risk**: LOW - only wires existing learning components

#### Tasks:
1. **Connect OutcomeEvaluator to ParameterAdapter for tasks**
   - File: `src/services/parameter_adapter.py`
   - Extend `adapt_all_decisions()` to include task decisions
   - Query DecisionRegistry for unevaluated task_selected decisions
   - Use TaskOutcome (Group 1) to compute success scores
   - Adapt task selection parameters

2. **Test end-to-end closed loop**
   - File: `tests/integration/test_closed_loop_learning.py`
   - Execute multiple tasks
   - Verify outcomes recorded
   - Verify parameters adapted
   - Verify adapted parameters affect next task selection

**Test files**:
- `tests/integration/test_closed_loop_learning.py`
- `tests/integration/test_task_learning_convergence.py`

**Success criteria**:
- Parameters adapt after task outcomes evaluated
- Adapted parameters improve task selection over time
- Learning loop completes within 24 hours
- No feedback loops or oscillation

---

## Dependency Graph

```
GROUP 1 (Metrics)          GROUP 2 (Safety)
    |                           |
    ↓                           ↓
    └─→ GROUP 3 (Decision) ←────┘
              |
              ↓
         GROUP 4 (Learning)
```

## Implementation Order

### Week 1:
**Monday-Tuesday**: GROUP 1 + GROUP 2 (parallel)
- Extend OutcomeEvaluator for tasks
- Add task metrics
- Wire AbortConditionMonitor to TaskScheduler
- Write tests for both groups

**Wednesday-Friday**: GROUP 3 (depends on Group 1)
- Wire DecisionFramework to TaskScheduler
- Register task_selected decision point
- Link outcomes to decisions
- Write integration tests

### Week 2:
**Monday-Tuesday**: GROUP 4 (depends on Groups 1 and 3)
- Connect OutcomeEvaluator to ParameterAdapter
- Test end-to-end closed loop
- Regression testing

**Wednesday-Thursday**: Integration testing and bug fixes
- Run full test suite
- Monitor staging deployment
- Fix any edge cases

**Friday**: Code review and merge preparation
- Documentation updates
- Performance benchmarking
- Merge to main

---

## Testing Strategy

### Unit Tests (per group)
Each group has isolated unit tests that don't depend on other groups.

### Integration Tests (after groups complete)
- `test_metrics_safety_integration.py` - Groups 1 + 2
- `test_decision_outcome_integration.py` - Groups 1 + 3
- `test_closed_loop_learning.py` - All groups

### Regression Tests
- Verify existing task execution still works
- Verify belief-related decisions still work
- Verify no performance degradation

---

## Rollback Plan

Each group is independently revertable:

**GROUP 1**: Remove TaskOutcome evaluation (no impact on existing systems)
**GROUP 2**: Remove safety checks (tasks execute as before)
**GROUP 3**: Remove decision recording (tasks execute as before)
**GROUP 4**: Disable task parameter adaptation (learning continues for beliefs)

---

## Monitoring During Rollout

### Metrics to watch:
- Task execution success rate (should not decrease)
- Task execution latency (should not increase significantly)
- Decision registry growth rate (should be ~1 decision per task)
- Parameter adaptation rate (should be 1-2 adaptations per day)
- Abort condition trigger rate (should be rare, <1% of tasks)

### Alerts:
- Task success rate drops below 80%
- Decision recording fails (indicates schema issue)
- Parameter adapter throws exceptions
- Abort monitor blocks >10% of tasks

---

## Files Modified

### Modified:
- `src/services/outcome_evaluator.py` - Add task outcome evaluation
- `src/services/task_scheduler.py` - Add decision recording and safety checks
- `src/services/parameter_adapter.py` - Extend to handle task decisions

### New:
- `src/services/task_metrics.py` - Task execution metrics
- `tests/test_task_outcome_evaluator.py`
- `tests/test_task_metrics.py`
- `tests/test_task_safety_integration.py`
- `tests/test_decision_task_integration.py`
- `tests/integration/test_closed_loop_learning.py`

### Docs:
- This file (PHASE0_IMPLEMENTATION_PLAN.md)
- Update AUTONOMOUS_AGENT_ARCHITECTURE_ANALYSIS.md with completion status

---

## Success Criteria for Phase 0 Completion

1. ✅ All tasks log decisions to DecisionRegistry
2. ✅ AbortConditionMonitor blocks tasks during abort state
3. ✅ TaskOutcomes computed for all task executions
4. ✅ ParameterAdapter learns from task outcomes
5. ✅ Parameters demonstrably improve task selection over 48 hours
6. ✅ All tests passing (unit + integration)
7. ✅ No regression in existing functionality
8. ✅ Documentation updated

---

## Next Phase

After Phase 0 completes, proceed to:
**Phase 1: GoalStore** - Add value/effort/risk prioritization to tasks

Phase 1 will build on the closed learning loop established in Phase 0.

---

*Implementation plan ready. Proceed with GROUP 1 and GROUP 2 in parallel.*
