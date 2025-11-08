# Phase 2: TaskGraph + Executor - Design Specification

**Branch**: `claude/feature/phase2-taskgraph`
**Timeline**: 3-4 weeks
**Dependencies**: Phase 0 (merged in PR #2)
**Goal**: Add dependency tracking, proper state management, and robust task execution
**Status**: ✅ APPROVED by Quantum Tsar of Arrays with production-ready requirements

---

## EXECUTIVE SUMMARY

Phase 2 extracts task execution logic into a proper state machine with dependency tracking. This enables:
- **Parallel task execution** with dependency constraints
- **Robust error handling** with automatic retries
- **Safety-gated execution** integrated with AbortConditionMonitor
- **Idempotent execution** with proper deduplication

### What We're Building

```
TaskGraph: Dependency tracking + state management
    ↓
TaskExecutor: Isolated execution logic + retry/idempotency
    ↓
TaskScheduler: Orchestration + decision recording
```

### Success Criteria

1. Tasks respect dependency constraints (can't run until dependencies complete)
2. Safety envelope blocks task execution during abort conditions
3. Failed tasks retry with exponential backoff
4. Idempotency prevents duplicate execution
5. State transitions logged for observability
6. No regression in existing task execution

---

## ARCHITECTURE

### Current State (Phase 0)

```python
TaskScheduler:
  - execute_task() - monolithic execution
  - Decision recording ✓
  - Parameter adaptation ✓
  - No dependency tracking ✗
  - No retry logic ✗
  - Limited state management ✗
```

### Target State (Phase 2)

```python
TaskGraph:
  - Track task dependencies
  - Manage task states (PENDING/READY/RUNNING/SUCCEEDED/FAILED/ABORTED)
  - Return ready tasks (all dependencies met)
  - Handle task state transitions

TaskExecutor:
  - Execute single task
  - Retry with exponential backoff
  - Idempotency key checking
  - Safety envelope integration
  - Capture execution metadata

TaskScheduler:
  - Orchestrate task graph
  - Record decisions
  - Trigger parameter adaptation
  - Schedule task execution
```

---

## COMPONENT DESIGN

### 1. TaskState Enum

```python
class TaskState(str, Enum):
    PENDING = "pending"        # Created, waiting for dependencies
    READY = "ready"            # Dependencies met, ready to execute
    RUNNING = "running"        # Currently executing
    SUCCEEDED = "succeeded"    # Completed successfully
    FAILED = "failed"          # Failed after retries
    ABORTED = "aborted"        # Aborted by safety envelope
    BLOCKED = "blocked"        # Blocked by missing dependency
```

### 2. TaskNode

```python
@dataclass
class TaskNode:
    """Node in task dependency graph."""
    task: Task
    state: TaskState
    dependencies: List[str]  # task_ids this depends on
    dependents: List[str]    # task_ids that depend on this
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    idempotency_key: Optional[str] = None

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are completed."""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
```

### 3. TaskGraph

```python
class TaskGraph:
    """Manages task dependencies and states."""

    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()

    def add_task(
        self,
        task: Task,
        dependencies: List[str] = None
    ) -> None:
        """Add task to graph."""
        if task.id in self.nodes:
            raise ValueError(f"Task {task.id} already in graph")

        node = TaskNode(
            task=task,
            state=TaskState.PENDING,
            dependencies=dependencies or [],
            dependents=[]
        )

        # Update dependents
        for dep_id in node.dependencies:
            if dep_id in self.nodes:
                self.nodes[dep_id].dependents.append(task.id)

        self.nodes[task.id] = node

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks ready to execute (dependencies met, not running)."""
        ready = []
        for node in self.nodes.values():
            if node.state == TaskState.PENDING and node.is_ready(self.completed):
                node.state = TaskState.READY
                ready.append(node.task)
        return ready

    def mark_running(self, task_id: str) -> None:
        """Mark task as running."""
        if task_id not in self.nodes:
            raise KeyError(f"Task {task_id} not in graph")
        self.nodes[task_id].state = TaskState.RUNNING
        self.nodes[task_id].started_at = datetime.now(timezone.utc)

    def mark_completed(self, task_id: str, success: bool) -> None:
        """Mark task as completed (succeeded or failed)."""
        node = self.nodes[task_id]

        if success:
            node.state = TaskState.SUCCEEDED
            node.completed_at = datetime.now(timezone.utc)
            self.completed.add(task_id)
        else:
            node.state = TaskState.FAILED
            node.completed_at = datetime.now(timezone.utc)
            self.failed.add(task_id)

    def mark_aborted(self, task_id: str, reason: str) -> None:
        """Mark task as aborted by safety envelope."""
        node = self.nodes[task_id]
        node.state = TaskState.ABORTED
        node.last_error = f"Aborted: {reason}"
        self.failed.add(task_id)

    def get_blocked_tasks(self) -> List[Tuple[str, List[str]]]:
        """Get tasks blocked by failed dependencies."""
        blocked = []
        for task_id, node in self.nodes.items():
            if node.state == TaskState.PENDING:
                failed_deps = [dep for dep in node.dependencies if dep in self.failed]
                if failed_deps:
                    blocked.append((task_id, failed_deps))
        return blocked

    def is_complete(self) -> bool:
        """Check if all tasks are in terminal state."""
        terminal_states = {TaskState.SUCCEEDED, TaskState.FAILED, TaskState.ABORTED}
        return all(node.state in terminal_states for node in self.nodes.values())

    def get_stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        stats = defaultdict(int)
        for node in self.nodes.values():
            stats[node.state.value] += 1
        return dict(stats)
```

### 4. TaskExecutor

```python
class TaskExecutor:
    """Executes individual tasks with retry and safety checks."""

    def __init__(
        self,
        abort_monitor: Optional[AbortConditionMonitor] = None,
        decision_registry: Optional[DecisionRegistry] = None,
        raw_store: Optional[RawStore] = None
    ):
        self.abort_monitor = abort_monitor
        self.decision_registry = decision_registry
        self.raw_store = raw_store

    async def execute(
        self,
        task: Task,
        persona_service,
        idempotency_key: Optional[str] = None,
        retry_count: int = 0
    ) -> TaskExecutionResult:
        """
        Execute a single task with safety checks and idempotency.

        Returns:
            TaskExecutionResult with success status and metadata
        """
        # 1. Check for duplicate execution (idempotency)
        if idempotency_key and self._is_duplicate(idempotency_key):
            return self._get_cached_result(idempotency_key)

        # 2. Safety envelope check
        if self.abort_monitor and self.abort_monitor.should_abort():
            abort_reason = self.abort_monitor.get_abort_reason()
            return TaskExecutionResult(
                task_id=task.id,
                status=TaskStatus.ABORTED,
                error=f"Aborted by safety envelope: {abort_reason}",
                aborted=True,
                abort_reason=abort_reason
            )

        # 3. Record decision (if decision registry available)
        decision_record_id = None
        if self.decision_registry:
            decision_record_id = self.decision_registry.record_decision(
                decision_id="task_selected",
                context={"task_type": task.type, "retry_count": retry_count},
                parameters_used=self.decision_registry.get_all_parameters("task_selected") or {}
            )

        # 4. Execute task
        try:
            started_at = datetime.now(timezone.utc)

            # Actual execution logic
            result = await self._execute_task_logic(task, persona_service)

            completed_at = datetime.now(timezone.utc)

            # 5. Record execution to raw store
            if self.raw_store:
                self.raw_store.create_task_execution(
                    task_id=task.id,
                    task_type=task.type,
                    status="success" if result.success else "failed",
                    started_at=started_at,
                    completed_at=completed_at,
                    error_message=result.error,
                    trace_id=result.trace_id,
                    idempotency_key=idempotency_key
                )

            # 6. Cache result for idempotency
            if idempotency_key:
                self._cache_result(idempotency_key, result)

            return result

        except Exception as e:
            logger.error(f"Task execution failed: {task.id} - {e}")
            return TaskExecutionResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                retryable=self._is_retryable_error(e)
            )

    async def _execute_task_logic(
        self,
        task: Task,
        persona_service
    ) -> TaskExecutionResult:
        """Core task execution logic (extracted from TaskScheduler)."""
        # This will be the actual task execution code
        # moved from TaskScheduler.execute_task()
        pass

    def _is_duplicate(self, idempotency_key: str) -> bool:
        """Check if task already executed with this key."""
        if not self.raw_store:
            return False

        executions = self.raw_store.list_task_executions(
            idempotency_key=idempotency_key,
            limit=1
        )
        return len(executions) > 0

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if error is retryable."""
        # Network errors, timeouts are retryable
        # Validation errors, abort conditions are not
        retryable_types = (
            asyncio.TimeoutError,
            ConnectionError,
            # Add more retryable exception types
        )
        return isinstance(error, retryable_types)
```

### 5. Updated TaskScheduler

```python
class TaskScheduler:
    """Orchestrates task execution with graph management."""

    def __init__(
        self,
        persona_space_path: str,
        executor: TaskExecutor,
        decision_registry: Optional[DecisionRegistry] = None,
        parameter_adapter: Optional[ParameterAdapter] = None
    ):
        self.executor = executor
        self.decision_registry = decision_registry
        self.parameter_adapter = parameter_adapter
        self.graph = TaskGraph()

        # Keep existing task tracking
        self.tasks: Dict[str, Task] = {}
        self.executions_since_adaptation = 0
        self.adaptation_interval = 10

    async def execute_graph(
        self,
        persona_service,
        max_parallel: int = 3
    ) -> Dict[str, TaskExecutionResult]:
        """
        Execute entire task graph with dependencies.

        Args:
            persona_service: Service for task execution
            max_parallel: Maximum tasks to run in parallel

        Returns:
            Dictionary of task_id -> execution result
        """
        results = {}
        running_tasks = {}

        while not self.graph.is_complete():
            # Get ready tasks
            ready_tasks = self.graph.get_ready_tasks()

            # Start new tasks (up to max_parallel limit)
            while ready_tasks and len(running_tasks) < max_parallel:
                task = ready_tasks.pop(0)
                self.graph.mark_running(task.id)

                # Execute task asynchronously
                task_future = asyncio.create_task(
                    self.executor.execute(
                        task=task,
                        persona_service=persona_service
                    )
                )
                running_tasks[task.id] = task_future

            # Wait for any task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Process completed tasks
                for future in done:
                    # Find task_id for this future
                    task_id = next(
                        tid for tid, f in running_tasks.items() if f == future
                    )

                    result = await future
                    results[task_id] = result

                    # Update graph state
                    self.graph.mark_completed(
                        task_id,
                        success=result.status == TaskStatus.COMPLETED
                    )

                    # Remove from running
                    del running_tasks[task_id]

                    # Trigger parameter adaptation if needed
                    self.executions_since_adaptation += 1
                    self.trigger_parameter_adaptation()
            else:
                # No tasks running or ready - check for deadlock
                blocked = self.graph.get_blocked_tasks()
                if blocked:
                    logger.error(f"Deadlock detected: {len(blocked)} tasks blocked by failed dependencies")
                    break

                # No progress possible
                await asyncio.sleep(0.1)

        return results

    async def execute_task_with_retry(
        self,
        task: Task,
        persona_service,
        max_retries: int = 3
    ) -> TaskExecutionResult:
        """
        Execute single task with exponential backoff retry.

        Args:
            task: Task to execute
            persona_service: Service for execution
            max_retries: Maximum retry attempts

        Returns:
            Final execution result
        """
        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            result = await self.executor.execute(
                task=task,
                persona_service=persona_service,
                retry_count=retry_count
            )

            if result.status == TaskStatus.COMPLETED:
                return result

            if result.status == TaskStatus.ABORTED:
                # Don't retry aborted tasks
                return result

            if not result.retryable:
                # Don't retry non-retryable errors
                return result

            # Retry with exponential backoff
            retry_count += 1
            if retry_count <= max_retries:
                backoff_seconds = 2 ** retry_count  # 2, 4, 8 seconds
                logger.info(
                    f"Retrying task {task.id} in {backoff_seconds}s "
                    f"(attempt {retry_count}/{max_retries})"
                )
                await asyncio.sleep(backoff_seconds)
                last_error = result.error

        # All retries exhausted
        return TaskExecutionResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            error=f"Failed after {max_retries} retries. Last error: {last_error}"
        )
```

---

## IMPLEMENTATION PLAN

### GROUP 1: Core Data Structures (Week 1)

**Files to create:**
- `src/services/task_graph.py` - TaskGraph, TaskNode, TaskState
- `tests/test_task_graph.py` - Unit tests

**Tasks:**
1. Implement TaskState enum
2. Implement TaskNode dataclass
3. Implement TaskGraph class
4. Add dependency tracking logic
5. Add state transition methods
6. Write comprehensive unit tests

**Success criteria:**
- TaskGraph tracks dependencies correctly
- get_ready_tasks() returns only tasks with met dependencies
- State transitions work correctly
- Blocked task detection works
- All tests pass

---

### GROUP 2: TaskExecutor (Week 2)

**Files to create:**
- `src/services/task_executor.py` - TaskExecutor class
- `tests/test_task_executor.py` - Unit tests

**Files to modify:**
- `src/services/task_scheduler.py` - Extract execution logic

**Tasks:**
1. Create TaskExecutor class
2. Extract execution logic from TaskScheduler
3. Add safety envelope integration
4. Add idempotency checking
5. Add retry logic with exponential backoff
6. Write unit tests

**Success criteria:**
- TaskExecutor executes tasks correctly
- Safety envelope blocks tasks during abort
- Idempotency prevents duplicate execution
- Retry works with exponential backoff
- All tests pass

---

### GROUP 3: TaskScheduler Refactoring (Week 3)

**Files to modify:**
- `src/services/task_scheduler.py` - Use TaskGraph + TaskExecutor

**Tasks:**
1. Integrate TaskGraph into TaskScheduler
2. Implement execute_graph() for parallel execution
3. Implement execute_task_with_retry()
4. Preserve decision recording
5. Preserve parameter adaptation
6. Update factory functions

**Success criteria:**
- Tasks execute through new TaskExecutor
- Parallel execution works
- Decision recording still works
- Parameter adaptation still triggers
- No regression in existing tests

---

### GROUP 4: Integration Testing (Week 4)

**Files to create:**
- `tests/integration/test_task_graph_execution.py`

**Tasks:**
1. Test parallel task execution
2. Test dependency constraints
3. Test retry behavior
4. Test safety envelope integration
5. Test idempotency
6. Test blocked task detection
7. Regression testing

**Success criteria:**
- All integration tests pass
- All existing tests still pass
- No performance regression
- Documentation updated

---

## SAFETY & ROLLBACK

### Safety Checks

1. **Backward compatibility**: Keep old execute_task() API
2. **Feature flag**: Enable TaskGraph via config flag
3. **Graceful degradation**: Fall back to simple execution on error
4. **Logging**: Extensive logging at state transitions

### Rollback Plan

- Revert to TaskScheduler.execute_task() if issues
- Feature flag allows instant disable
- Database schema unchanged (no migrations needed)

---

## METRICS & OBSERVABILITY

### New Metrics

```python
task_graph_size - Gauge of tasks in graph
task_states - Counter per state (pending/running/succeeded/failed)
task_retry_count - Histogram of retry attempts
task_execution_duration - Histogram of execution time
task_blocked_count - Counter of blocked tasks
parallel_tasks - Gauge of currently running tasks
```

### Logging

```python
# State transitions
logger.info(f"Task {task_id}: {old_state} → {new_state}")

# Dependency blocking
logger.warning(f"Task {task_id} blocked by: {failed_dependencies}")

# Retries
logger.info(f"Retrying task {task_id} (attempt {retry_count}/{max_retries})")

# Safety blocks
logger.warning(f"Task {task_id} blocked by safety envelope: {reason}")
```

---

## TESTING STRATEGY

### Unit Tests

- TaskGraph dependency tracking
- TaskNode state transitions
- TaskExecutor retry logic
- Idempotency checking
- Safety envelope integration

### Integration Tests

- Parallel execution with dependencies
- Failed dependency blocking
- Retry with exponential backoff
- Abort condition during execution
- Full graph execution end-to-end

### Performance Tests

- 100 tasks in graph
- 10 parallel executions
- Measure overhead vs current implementation

---

## FUTURE ENHANCEMENTS (Phase 3)

- HTN Planner for automatic task decomposition
- Task priority scheduling
- Resource constraints (CPU/memory limits)
- Task cancellation/preemption
- Distributed execution across agents

---

## CONCLUSION

Phase 2 transforms task execution from simple sequential execution to a robust, parallel, dependency-aware system. This foundation enables:

- **Scalability**: Execute multiple tasks in parallel
- **Reliability**: Automatic retries and safety checks
- **Observability**: Full state tracking and metrics
- **Extensibility**: Ready for HTN planner integration

**Estimated effort**: 3-4 weeks
**Risk level**: MEDIUM (refactoring existing execution logic)
**Dependencies**: Phase 0 (merged)
**Blocker**: None

---

*Ready for implementation pending user approval.*
