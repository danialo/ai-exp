# Current Work Assignments

**Date**: 2025-11-08
**Status**: Ready to begin parallel work

---

## Branch Status

### ✅ master
- **Status**: Just merged `feature/immutable-belief-logit-bias`
- **Contains**: Complete adaptive decision framework, belief system, task tracking
- **Next**: Base branch for all new work

### 🚧 feature/phase0-integration-wiring (Claude's work)
- **Assignee**: Claude
- **Purpose**: Integration wiring to close the adaptive learning loop
- **Spec**: `docs/PHASE0_IMPLEMENTATION_PLAN.md`
- **Timeline**: 1-2 weeks
- **Work Groups**:
  - GROUP 1 (Metrics): Extend OutcomeEvaluator for tasks - INDEPENDENT
  - GROUP 2 (Safety): Wire AbortMonitor to TaskScheduler - INDEPENDENT
  - GROUP 3 (Decision): Wire DecisionFramework to tasks - DEPENDS ON GROUP 1
  - GROUP 4 (Learning): Close the loop - DEPENDS ON GROUPS 1+3

### 🚧 feature/goal-store (Codex's work)
- **Assignee**: Codex
- **Purpose**: Add value/effort/risk goal prioritization
- **Spec**: `.claude/tasks/goalstore-phase1-spec.md`
- **Timeline**: 2-3 weeks
- **Work Tasks**:
  - Task 1: Extend data model (GoalDefinition)
  - Task 2: Implement GoalStore class
  - Task 3: Register goal_selected decision point
  - Task 4: Migrate existing tasks to goals
  - Task 5: Integration with TaskScheduler

---

## Parallel Work Strategy

These two branches are **completely independent** and can be worked on simultaneously:

**Claude (Phase 0)**: Wiring existing components together
- Extends OutcomeEvaluator, AbortConditionMonitor, DecisionFramework
- No new data structures
- Pure integration work

**Codex (GoalStore)**: Adding goal prioritization layer
- New GoalDefinition dataclass
- New GoalStore class
- Extends TaskScheduler
- No conflicts with Phase 0 work

**Merge order**: Either can merge first, no dependencies.

---

## Next Steps

1. **User**: Enhance `.claude/tasks/goalstore-phase1-spec.md` as needed
2. **Codex**: Create `feature/goal-store` branch and begin Task 1
3. **Claude**: Begin Phase 0 GROUP 1 (Metrics Extension)

---

## Communication

If either branch needs changes that affect the other:
- Post in shared doc
- Coordinate merge timing
- Rebase if needed

---

*Let's ship it.*
