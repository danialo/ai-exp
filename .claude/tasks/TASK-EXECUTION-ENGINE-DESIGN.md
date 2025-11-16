# Task Execution Engine - Design

## Overview

Complete the autonomous task execution pipeline: Goals → HTN Planning → TaskGraph → Execution → Completion

## Current State

**What works:**
- ✅ HTN Planner decomposes goals into task sequences
- ✅ TaskGraph tracks dependencies, states, concurrency
- ✅ TaskScheduler has scheduling infrastructure
- ✅ CodeAccessService can modify files
- ✅ TaskGraph Query API + UI

**What's missing:**
- ❌ Primitive task executors (the actual code that runs each task)
- ❌ Execution loop that picks ready tasks and runs them
- ❌ Wiring HTN planner output to execution
- ❌ State updates (running → succeeded/failed)

## Architecture

### 1. Primitive Task Executors

Each executor handles one task type:

```python
class TaskExecutor(ABC):
    @abstractmethod
    async def execute(self, task_node: TaskNode) -> ExecutionResult:
        """Execute task and return result."""
        pass

    @abstractmethod
    def can_handle(self, action_name: str) -> bool:
        """Check if this executor can handle the action."""
        pass
```

**Executor types:**

1. **CodeModificationExecutor**
   - Actions: `modify_code`, `create_file`, `delete_file`
   - Uses: CodeAccessService
   - Validation: Syntax check, git diff

2. **TestExecutor**
   - Actions: `run_tests`, `pytest`, `npm_test`
   - Uses: subprocess
   - Validation: Exit code, coverage output

3. **ShellCommandExecutor**
   - Actions: `shell_command`, `bash`
   - Uses: subprocess with timeout
   - Validation: Exit code

4. **ValidationExecutor**
   - Actions: `check_output`, `verify_state`
   - Uses: File reads, pattern matching
   - Validation: Assertion checks

5. **BuildExecutor**
   - Actions: `npm_build`, `go_build`, `docker_build`
   - Uses: subprocess
   - Validation: Build artifacts exist

### 2. Task Execution Engine

Main execution loop:

```python
class TaskExecutionEngine:
    def __init__(
        self,
        task_graph: TaskGraph,
        executors: List[TaskExecutor],
        max_concurrent: int = 4
    ):
        self.task_graph = task_graph
        self.executors = {ex.__class__.__name__: ex for ex in executors}
        self.running_tasks: Dict[str, asyncio.Task] = {}

    async def run(self):
        """Main execution loop."""
        while not self.task_graph.is_complete():
            # Get ready tasks
            ready = self.task_graph.get_ready_tasks()

            # Respect concurrency limits
            available_slots = self.max_concurrent - len(self.running_tasks)

            for task in ready[:available_slots]:
                # Find executor
                executor = self._find_executor(task.action_name)

                # Start async execution
                asyncio_task = asyncio.create_task(
                    self._execute_task(task, executor)
                )
                self.running_tasks[task.task_id] = asyncio_task

            # Wait for any task to complete
            if self.running_tasks:
                done, pending = await asyncio.wait(
                    self.running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                # Clean up completed tasks
                for task in done:
                    task_id = self._get_task_id_from_future(task)
                    del self.running_tasks[task_id]

            await asyncio.sleep(1)  # Polling interval

    async def _execute_task(self, task: TaskNode, executor: TaskExecutor):
        """Execute single task."""
        try:
            # Mark running
            self.task_graph.mark_running(task.task_id)

            # Execute
            result = await executor.execute(task)

            # Update state
            if result.success:
                self.task_graph.mark_completed(task.task_id)
            else:
                self.task_graph.mark_failed(
                    task.task_id,
                    error=result.error,
                    error_class=result.error_class
                )
        except Exception as e:
            self.task_graph.mark_failed(
                task.task_id,
                error=str(e),
                error_class=e.__class__.__name__
            )
```

### 3. Integration with HTN Planner

Wire the flow:

```python
# In goal execution
async def execute_goal(goal: Goal):
    # 1. HTN Planning
    plan = htn_planner.plan(
        goal_id=goal.id,
        goal_text=goal.description
    )

    if not plan:
        raise PlanningError("No valid plan found")

    # 2. Convert to TaskGraph
    task_graph = plan_to_task_graph(plan)

    # 3. Persist TaskGraph
    _persist_task_graph(task_graph)  # Already exists

    # 4. Create execution engine
    executors = [
        CodeModificationExecutor(code_access_service),
        TestExecutor(),
        ShellCommandExecutor(),
        ValidationExecutor(),
        BuildExecutor()
    ]

    engine = TaskExecutionEngine(
        task_graph=task_graph,
        executors=executors,
        max_concurrent=4
    )

    # 5. Run to completion
    await engine.run()

    # 6. Return results
    stats = task_graph.get_stats()
    return ExecutionSummary(
        goal_id=goal.id,
        plan_id=plan.plan_id,
        graph_id=task_graph.graph_id,
        total_tasks=stats['total_tasks'],
        succeeded=stats['states'].get('succeeded', 0),
        failed=stats['states'].get('failed', 0),
        duration_ms=task_graph.get_duration_ms()
    )
```

### 4. HTN Method Library

Define decomposition methods for common programming tasks:

```python
PROGRAMMING_METHODS = [
    Method(
        name="implement_feature_with_tests",
        task="implement_feature",
        preconditions=["has_codebase", "has_test_framework"],
        subtasks=[
            "analyze_requirements",
            "design_implementation",
            "write_code",
            "write_tests",
            "run_tests",
            "fix_issues"
        ],
        cost=0.8
    ),
    Method(
        name="fix_bug",
        task="fix_bug",
        preconditions=["has_error_logs"],
        subtasks=[
            "locate_bug_source",
            "write_fix",
            "run_tests",
            "verify_fix"
        ],
        cost=0.6
    ),
    Method(
        name="refactor_code",
        task="refactor_code",
        preconditions=["has_target_code"],
        subtasks=[
            "analyze_current_structure",
            "design_refactoring",
            "apply_refactoring",
            "run_tests",
            "verify_no_regression"
        ],
        cost=0.7
    )
]

PRIMITIVE_TASKS = {
    "write_code",
    "write_tests",
    "run_tests",
    "analyze_requirements",
    "locate_bug_source",
    "verify_fix",
    "apply_refactoring"
}
```

## Implementation Plan

### Phase 1: Core Executors (Day 1)
1. Create `src/services/task_executors/` directory
2. Implement base `TaskExecutor` abstract class
3. Implement `CodeModificationExecutor` (uses CodeAccessService)
4. Implement `TestExecutor` (runs pytest, npm test)
5. Implement `ShellCommandExecutor` (basic subprocess)
6. Add unit tests for each executor

### Phase 2: Execution Engine (Day 2)
1. Implement `TaskExecutionEngine` class
2. Add async task execution with concurrency control
3. Add state management (running → succeeded/failed)
4. Add retry logic integration
5. Add circuit breaker integration
6. Add execution telemetry/logging

### Phase 3: HTN Integration (Day 3)
1. Define programming method library
2. Wire HTN planner to execution engine
3. Add goal → execution flow in goal_store
4. Test end-to-end: create goal → execute → view in UI

### Phase 4: Advanced Executors (Day 4)
1. Implement `ValidationExecutor`
2. Implement `BuildExecutor`
3. Add file watcher for changed files
4. Add result validation strategies

### Phase 5: Testing & Polish (Day 5)
1. End-to-end integration tests
2. Load testing (multiple concurrent goals)
3. Failure recovery testing
4. UI updates (show running tasks live)
5. Documentation

## Success Criteria

- [ ] User creates goal: "Implement a function to calculate fibonacci"
- [ ] HTN planner decomposes into tasks
- [ ] TaskGraph shows in UI with pending tasks
- [ ] Execution engine picks up ready tasks
- [ ] Tasks execute automatically (code written, tests run)
- [ ] TaskGraph updates in real-time (pending → running → succeeded)
- [ ] User sees final result in UI
- [ ] Goal marked complete

## Files to Create

```
src/services/task_executors/
├── __init__.py
├── base.py                    # TaskExecutor ABC
├── code_modification.py       # CodeModificationExecutor
├── test_runner.py             # TestExecutor
├── shell_command.py           # ShellCommandExecutor
├── validation.py              # ValidationExecutor
└── build.py                   # BuildExecutor

src/services/task_execution_engine.py   # Main engine
src/services/htn_method_library.py      # Programming methods
tests/services/test_executors.py        # Executor tests
tests/services/test_execution_engine.py # Engine tests
```

## Risk Mitigation

**Risk:** Infinite execution loops
**Mitigation:** Task timeout limits, max retry budgets, circuit breakers

**Risk:** Code modification errors break codebase
**Mitigation:** Git integration, automatic rollback on test failures

**Risk:** Resource exhaustion
**Mitigation:** Concurrency limits, memory monitoring, kill switches

**Risk:** Security issues (arbitrary code execution)
**Mitigation:** Sandbox executors, whitelist allowed commands, audit logs

## Next Steps

1. Review this design with user
2. Start with Phase 1: Core Executors
3. Build incrementally with testing at each phase
