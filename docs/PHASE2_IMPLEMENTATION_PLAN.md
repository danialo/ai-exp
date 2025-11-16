# Phase 2: TaskGraph + Executor - Implementation Plan

**Branch**: `claude/feature/phase2-taskgraph`
**Timeline**: 3-4 weeks
**Status**: Ready to start

---

## WORK GROUPS

Safe to implement in parallel where noted.

### GROUP 1: Core Data Structures ✓ (Independent)

**Week 1 - Estimated 8-12 hours**

- [ ] Create `src/services/task_graph.py`
  - [ ] TaskState enum
  - [ ] TaskNode dataclass
  - [ ] TaskGraph class with dependency tracking
  - [ ] State transition methods
  - [ ] get_ready_tasks() implementation
  - [ ] get_blocked_tasks() implementation

- [ ] Create `tests/test_task_graph.py`
  - [ ] Test adding tasks with dependencies
  - [ ] Test state transitions
  - [ ] Test ready task detection
  - [ ] Test blocked task detection
  - [ ] Test graph completion checking
  - [ ] Test statistics

**Success Criteria:**
- All unit tests pass
- TaskGraph correctly tracks dependencies
- State transitions work properly

---

### GROUP 2: TaskExecutor ✓ (Depends on GROUP 1)

**Week 2 - Estimated 12-16 hours**

- [ ] Create `src/services/task_executor.py`
  - [ ] TaskExecutor class structure
  - [ ] Safety envelope integration
  - [ ] Idempotency checking
  - [ ] Retry logic with exponential backoff
  - [ ] Decision recording integration
  - [ ] Error handling and classification

- [ ] Extract execution logic from TaskScheduler
  - [ ] Move _execute_task_logic() to TaskExecutor
  - [ ] Preserve decision recording
  - [ ] Preserve metadata capture

- [ ] Create `tests/test_task_executor.py`
  - [ ] Test successful execution
  - [ ] Test safety envelope blocking
  - [ ] Test idempotency
  - [ ] Test retry with backoff
  - [ ] Test error classification
  - [ ] Test decision recording

**Success Criteria:**
- All unit tests pass
- TaskExecutor executes tasks correctly
- Safety checks work
- Retry logic works
- No regression in execution behavior

---

### GROUP 3: TaskScheduler Integration ✓ (Depends on GROUP 1 + 2)

**Week 3 - Estimated 10-14 hours**

- [ ] Update `src/services/task_scheduler.py`
  - [ ] Add TaskGraph member
  - [ ] Add TaskExecutor member
  - [ ] Implement execute_graph() for parallel execution
  - [ ] Implement execute_task_with_retry()
  - [ ] Update factory function
  - [ ] Preserve parameter adaptation trigger
  - [ ] Add backward compatibility layer

- [ ] Update `app.py`
  - [ ] Wire TaskExecutor to TaskScheduler
  - [ ] Update factory calls

**Success Criteria:**
- Existing tests still pass (no regression)
- New parallel execution works
- Decision recording still works
- Parameter adaptation still triggers
- Backward compatibility maintained

---

### GROUP 4: Integration Testing ✓ (Depends on GROUP 1-3)

**Week 4 - Estimated 8-12 hours**

- [ ] Create `tests/integration/test_task_graph_execution.py`
  - [ ] Test parallel execution with dependencies
  - [ ] Test dependency constraint enforcement
  - [ ] Test retry behavior
  - [ ] Test safety envelope blocks execution
  - [ ] Test idempotency prevents duplicates
  - [ ] Test blocked task detection
  - [ ] Test full graph completion

- [ ] Run regression tests
  - [ ] All Phase 0 tests still pass
  - [ ] All existing task tests still pass
  - [ ] No performance regression

- [ ] Update documentation
  - [ ] Update AUTONOMOUS_AGENT_ARCHITECTURE_ANALYSIS.md
  - [ ] Add TaskGraph usage examples
  - [ ] Document new metrics

**Success Criteria:**
- All integration tests pass
- All regression tests pass
- Documentation updated
- Ready for PR

---

## IMPLEMENTATION ORDER

**Sequential (must be done in order):**

1. **GROUP 1** (Week 1) - Core data structures
   ↓
2. **GROUP 2** (Week 2) - TaskExecutor
   ↓
3. **GROUP 3** (Week 3) - Integration
   ↓
4. **GROUP 4** (Week 4) - Testing & docs

**Total estimated time:** 38-54 hours over 4 weeks

---

## FILES TO CREATE

```
src/services/task_graph.py          (new)
src/services/task_executor.py       (new)
tests/test_task_graph.py            (new)
tests/test_task_executor.py         (new)
tests/integration/test_task_graph_execution.py  (new)
```

## FILES TO MODIFY

```
src/services/task_scheduler.py      (refactor)
app.py                               (wire TaskExecutor)
docs/AUTONOMOUS_AGENT_ARCHITECTURE_ANALYSIS.md  (update)
```

---

## TESTING CHECKLIST

### Unit Tests (Groups 1-2)
- [ ] TaskState transitions
- [ ] TaskNode ready detection
- [ ] TaskGraph dependency tracking
- [ ] TaskGraph ready tasks selection
- [ ] TaskGraph blocked tasks detection
- [ ] TaskExecutor execution
- [ ] TaskExecutor safety checks
- [ ] TaskExecutor idempotency
- [ ] TaskExecutor retry logic

### Integration Tests (Group 4)
- [ ] Parallel execution (3 independent tasks)
- [ ] Sequential execution (A→B→C dependencies)
- [ ] Diamond dependencies (A→B,C→D)
- [ ] Retry after transient failure
- [ ] Safety envelope blocks task
- [ ] Idempotency prevents duplicate
- [ ] Blocked task stays blocked
- [ ] Full graph completes successfully

### Regression Tests
- [ ] All Phase 0 tests pass
- [ ] All task execution tests pass
- [ ] Decision recording still works
- [ ] Parameter adaptation still triggers
- [ ] No performance degradation

---

## ROLLOUT STRATEGY

### Phase 2a: Core Implementation
- Merge Groups 1-2 (TaskGraph + TaskExecutor)
- Feature flag OFF by default
- Manual testing in dev

### Phase 2b: Integration
- Merge Group 3 (TaskScheduler integration)
- Feature flag still OFF
- Integration testing

### Phase 2c: Production
- Enable feature flag gradually
- Monitor metrics
- Rollback if issues

---

## METRICS TO WATCH

During rollout, monitor:
- Task execution success rate (should not decrease)
- Task execution latency (should not increase significantly)
- Retry rate (track retry effectiveness)
- Safety blocks (should be rare)
- Parallel execution count (new metric)
- Blocked task count (should be zero or low)

---

## COMPLETION CRITERIA

Phase 2 is complete when:

1. ✅ All code implemented and merged
2. ✅ All unit tests passing
3. ✅ All integration tests passing
4. ✅ All regression tests passing
5. ✅ Documentation updated
6. ✅ Feature flag deployed
7. ✅ Metrics dashboard created
8. ✅ Team trained on new system

---

*Ready to begin implementation.*
