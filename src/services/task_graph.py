"""
TaskGraph - Dependency tracking and state management for task execution.

Production-ready implementation with:
- Cycle detection
- Dependency failure policies
- Priority queue scheduling
- Persistence and recovery
- Safety envelope integration
- Circuit breakers

Approved by Quantum Tsar of Arrays 2025-11-08
"""

import logging
import hashlib
import json
import random
import heapq
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any, Literal
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    """Task execution states with full lifecycle support."""
    PENDING = "pending"          # Created, waiting for dependencies
    READY = "ready"              # Dependencies met, ready to execute
    RUNNING = "running"          # Currently executing
    SUCCEEDED = "succeeded"      # Completed successfully
    FAILED = "failed"            # Failed after retries
    ABORTED = "aborted"          # Aborted by safety envelope
    SKIPPED = "skipped"          # Skipped due to dependency policy
    CANCELLED = "cancelled"      # Cancelled by user/system


class DependencyPolicy(str, Enum):
    """How to handle dependency failures."""
    ABORT = "abort"                     # Abort if any dependency fails
    SKIP = "skip"                       # Skip if any dependency fails
    CONTINUE_IF_ANY = "continue_if_any" # Continue if at least one succeeds


@dataclass
class TaskNode:
    """
    Node in task dependency graph.

    Includes production features:
    - Dependency failure policies
    - Timeouts
    - Priority/deadline scheduling
    - Retry budget tracking
    - Circuit breaker state
    """
    task_id: str
    action_name: str
    normalized_args: Dict[str, Any]
    resource_ids: List[str]
    version: str

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    on_dep_fail: DependencyPolicy = DependencyPolicy.ABORT

    # State
    state: TaskState = TaskState.PENDING

    # Scheduling
    priority: float = 0.5  # 0.0-1.0
    deadline: Optional[datetime] = None
    cost: float = 1.0  # Resource cost estimate

    # Timeouts
    task_timeout_ms: int = 300000  # 5 minutes default

    # Retries
    retry_count: int = 0
    max_retries: int = 3
    retry_tokens_used: int = 0
    next_run_at: Optional[float] = None  # Monotonic timestamp for retry scheduling

    # Execution metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_error: Optional[str] = None
    error_class: Optional[str] = None

    # Idempotency
    idempotency_key: Optional[str] = None

    # Tracing
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    worker_id: Optional[str] = None

    # Attempt history
    attempts: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Generate deterministic idempotency key."""
        if not self.idempotency_key:
            self.idempotency_key = self._generate_idempotency_key()

    def _generate_idempotency_key(self) -> str:
        """
        Generate deterministic idempotency key.

        key = sha256(action_name, normalized_args, resource_ids, version)
        """
        key_data = {
            "action": self.action_name,
            "args": self.normalized_args,
            "resources": sorted(self.resource_ids),
            "version": self.version
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def is_ready(self, completed_tasks: Set[str], failed_tasks: Set[str]) -> bool:
        """
        Check if task is ready to execute.

        Applies dependency policy for failed dependencies.
        """
        if not self.dependencies:
            return True

        succeeded_deps = [
            dep for dep in self.dependencies
            if dep in completed_tasks
        ]
        failed_deps = [
            dep for dep in self.dependencies
            if dep in failed_tasks
        ]

        # All dependencies must be in terminal state
        resolved_deps = len(succeeded_deps) + len(failed_deps)
        if resolved_deps < len(self.dependencies):
            return False  # Still waiting

        # Apply dependency policy
        if failed_deps:
            if self.on_dep_fail == DependencyPolicy.ABORT:
                return False  # Will be marked ABORTED
            elif self.on_dep_fail == DependencyPolicy.SKIP:
                return False  # Will be marked SKIPPED
            elif self.on_dep_fail == DependencyPolicy.CONTINUE_IF_ANY:
                return len(succeeded_deps) > 0  # Continue if any succeeded

        # All dependencies succeeded
        return True

    def should_skip(self, failed_tasks: Set[str]) -> bool:
        """Check if task should be skipped due to dependency policy."""
        failed_deps = [dep for dep in self.dependencies if dep in failed_tasks]
        if failed_deps:
            return self.on_dep_fail == DependencyPolicy.SKIP
        return False

    def should_abort(self, failed_tasks: Set[str]) -> bool:
        """Check if task should be aborted due to dependency policy."""
        failed_deps = [dep for dep in self.dependencies if dep in failed_tasks]
        if failed_deps:
            return self.on_dep_fail == DependencyPolicy.ABORT
        return False

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries

    def record_attempt(
        self,
        error: Optional[str] = None,
        error_class: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        worker_id: Optional[str] = None
    ) -> None:
        """Record execution attempt with full metadata."""
        attempt = {
            "attempt_number": self.retry_count + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error,
            "error_class": error_class,
            "trace_id": trace_id,
            "span_id": span_id,
            "worker_id": worker_id
        }
        self.attempts.append(attempt)

        if error:
            self.last_error = error
            self.error_class = error_class

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        data = asdict(self)
        data["deadline"] = self.deadline.isoformat() if self.deadline else None
        data["started_at"] = self.started_at.isoformat() if self.started_at else None
        data["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return data


class CircuitBreaker:
    """
    Per-action circuit breaker with sliding window.

    Opens after threshold failures, closes after timeout.
    """
    def __init__(
        self,
        action_name: str,
        failure_threshold: int = 5,
        window_seconds: int = 60,
        recovery_timeout_seconds: int = 30
    ):
        self.action_name = action_name
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.recovery_timeout_seconds = recovery_timeout_seconds

        self.state = "closed"  # closed, open, half_open
        self.failures: deque = deque()  # (timestamp, error_class)
        self.opened_at: Optional[datetime] = None

    def is_open(self) -> bool:
        """Check if breaker is open (blocking executions)."""
        if self.state == "closed":
            return False

        if self.state == "open":
            # Check if recovery timeout passed
            if self.opened_at:
                elapsed = (datetime.now(timezone.utc) - self.opened_at).total_seconds()
                if elapsed > self.recovery_timeout_seconds:
                    self.state = "half_open"
                    logger.info(f"Circuit breaker {self.action_name}: open → half_open")
                    return False
            return True

        return False  # half_open allows one attempt

    def record_success(self) -> None:
        """Record successful execution."""
        if self.state == "half_open":
            self.state = "closed"
            self.failures.clear()
            logger.info(f"Circuit breaker {self.action_name}: half_open → closed")

    def record_failure(self, error_class: str) -> None:
        """Record failed execution and potentially open breaker."""
        now = datetime.now(timezone.utc)
        self.failures.append((now, error_class))

        # Remove old failures outside window
        cutoff = now - timedelta(seconds=self.window_seconds)
        while self.failures and self.failures[0][0] < cutoff:
            self.failures.popleft()

        # Check if should open
        if len(self.failures) >= self.failure_threshold:
            if self.state == "closed":
                self.state = "open"
                self.opened_at = now
                logger.warning(
                    f"Circuit breaker {self.action_name}: OPENED "
                    f"({len(self.failures)} failures in {self.window_seconds}s)"
                )
            elif self.state == "half_open":
                self.state = "open"
                self.opened_at = now
                logger.warning(f"Circuit breaker {self.action_name}: half_open → open")

    def get_state(self) -> str:
        """Get current breaker state."""
        return self.state


class TaskGraph:
    """
    Task dependency graph with production features.

    Features:
    - Cycle detection on add_task
    - Priority queue for ready tasks
    - Per-action concurrency caps
    - Retry token budget
    - Circuit breakers per action
    - Persistence and recovery
    """

    def __init__(
        self,
        graph_id: str,
        graph_timeout_ms: int = 3600000,  # 1 hour default
        max_retry_tokens: int = 100,
        max_parallel: int = 4
    ):
        self.graph_id = graph_id
        self.graph_timeout_ms = graph_timeout_ms
        self.max_retry_tokens = max_retry_tokens
        self.max_parallel = max_parallel

        # Task tracking
        self.nodes: Dict[str, TaskNode] = {}
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        self.skipped: Set[str] = set()
        self.aborted: Set[str] = set()

        # Ready queue (priority heap)
        # Items: (-priority, -deadline_ts, -cost, task_id)
        self.ready_queue: List[Tuple] = []

        # Resource tracking
        self.retry_tokens_used: int = 0
        self.running_tasks: Set[str] = set()

        # Per-action limits
        self.action_concurrency: Dict[str, int] = defaultdict(int)
        self.action_concurrency_caps: Dict[str, int] = {}

        # Circuit breakers
        self.breakers: Dict[str, CircuitBreaker] = {}

        # Graph metadata
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def add_task(
        self,
        task_id: str,
        action_name: str,
        normalized_args: Dict[str, Any],
        resource_ids: List[str],
        version: str,
        dependencies: List[str] = None,
        on_dep_fail: DependencyPolicy = DependencyPolicy.ABORT,
        priority: float = 0.5,
        deadline: Optional[datetime] = None,
        cost: float = 1.0,
        task_timeout_ms: int = 300000,
        max_retries: int = 3
    ) -> None:
        """
        Add task to graph with cycle detection.

        Raises:
            ValueError: If task already exists or would create cycle
        """
        if task_id in self.nodes:
            raise ValueError(f"Task {task_id} already in graph")

        # Validate dependencies exist
        deps = dependencies or []
        for dep_id in deps:
            if dep_id not in self.nodes:
                raise ValueError(f"Dependency {dep_id} not found in graph")

        # Check for cycles
        if self._would_create_cycle(task_id, deps):
            raise ValueError(f"Adding task {task_id} would create a cycle")

        # Create node
        node = TaskNode(
            task_id=task_id,
            action_name=action_name,
            normalized_args=normalized_args,
            resource_ids=resource_ids,
            version=version,
            dependencies=deps,
            on_dep_fail=on_dep_fail,
            priority=priority,
            deadline=deadline,
            cost=cost,
            task_timeout_ms=task_timeout_ms,
            max_retries=max_retries
        )

        # Update dependents
        for dep_id in deps:
            self.nodes[dep_id].dependents.append(task_id)

        self.nodes[task_id] = node
        logger.info(f"Added task {task_id} (action={action_name}, deps={len(deps)})")

    def _would_create_cycle(self, new_task_id: str, dependencies: List[str]) -> bool:
        """
        Check if adding task with dependencies would create a cycle.

        Uses DFS to detect cycles in execution-order graph.
        Execution-order graph has edges: dependency -> dependent (task that depends on it)
        """
        # Build execution-order adjacency list (dep -> tasks that depend on it)
        graph = defaultdict(list)

        # Existing edges: for each task, add edges from its dependencies to it
        for task_id, node in self.nodes.items():
            for dep in node.dependencies:
                graph[dep].append(task_id)

        # New task edges: from each dependency to new task
        for dep in dependencies:
            graph[dep].append(new_task_id)

        # Check if there's a cycle in the augmented graph
        # A cycle exists if from new_task_id we can reach back to new_task_id
        visited_in_path = set()
        visited_global = set()

        def has_cycle_from(node: str) -> bool:
            if node in visited_in_path:
                return True  # Found cycle
            if node in visited_global:
                return False  # Already explored this path

            visited_in_path.add(node)
            visited_global.add(node)

            for neighbor in graph.get(node, []):
                if has_cycle_from(neighbor):
                    return True

            visited_in_path.remove(node)
            return False

        # Check for cycle starting from the new task
        return has_cycle_from(new_task_id)

    def get_ready_tasks(
        self,
        per_action_caps: Optional[Dict[str, int]] = None
    ) -> List[str]:
        """
        Get tasks ready to execute using priority queue.

        Priority selection with:
        - Earliest deadline first (EDF) as tiebreaker
        - Respect per-action concurrency caps
        - Check circuit breakers
        - Apply global concurrency limit

        Returns:
            List of task_ids ready to execute
        """
        # Update ready queue
        self._update_ready_queue()

        ready_tasks = []
        caps = per_action_caps or self.action_concurrency_caps

        # Track how many of each action we've selected in this batch
        batch_action_count = defaultdict(int)

        # Extract from priority queue
        temp_queue = []

        while self.ready_queue and len(ready_tasks) < self.max_parallel:
            if len(self.running_tasks) >= self.max_parallel:
                break

            neg_priority, deadline_ts, neg_cost, task_id = heapq.heappop(self.ready_queue)

            # Check if still ready (state may have changed)
            node = self.nodes[task_id]
            if node.state != TaskState.READY:
                continue

            # Check per-action cap (running + selected in this batch)
            action_cap = caps.get(node.action_name, float('inf'))
            current_running = self.action_concurrency[node.action_name]
            current_batch = batch_action_count[node.action_name]

            if current_running + current_batch >= action_cap:
                temp_queue.append((neg_priority, deadline_ts, neg_cost, task_id))
                continue

            # Check circuit breaker
            breaker = self._get_breaker(node.action_name)
            if breaker.is_open():
                logger.warning(f"Circuit breaker open for {node.action_name}, skipping {task_id}")
                temp_queue.append((neg_priority, deadline_ts, neg_cost, task_id))
                continue

            ready_tasks.append(task_id)
            batch_action_count[node.action_name] += 1

        # Restore skipped tasks to queue
        for item in temp_queue:
            heapq.heappush(self.ready_queue, item)

        return ready_tasks

    def _update_ready_queue(self) -> None:
        """Update ready queue with newly ready tasks."""
        import time
        current_time = time.monotonic()

        for task_id, node in self.nodes.items():
            if node.state == TaskState.PENDING:
                # Check if this is a retry-scheduled task waiting for its retry time
                if node.next_run_at is not None:
                    if current_time < node.next_run_at:
                        # Retry time hasn't arrived yet, skip this task
                        continue
                    # Retry time has arrived, clear next_run_at and proceed
                    node.next_run_at = None
                    logger.info(f"Task {task_id} retry time arrived, moving to ready check")

                # Check if should skip/abort due to dep failure
                if node.should_skip(self.failed | self.aborted):
                    node.state = TaskState.SKIPPED
                    self.skipped.add(task_id)
                    logger.info(f"Task {task_id} SKIPPED (dependency failure)")
                    continue

                if node.should_abort(self.failed | self.aborted):
                    node.state = TaskState.ABORTED
                    self.aborted.add(task_id)
                    logger.info(f"Task {task_id} ABORTED (dependency failure)")
                    continue

                # Check if ready
                if node.is_ready(self.completed, self.failed | self.aborted):
                    node.state = TaskState.READY

                    # Add to priority queue
                    # Higher priority comes first (negate priority)
                    # Earlier deadline comes first (use deadline_ts directly)
                    # Lower cost comes first (negate cost)
                    priority = node.priority
                    deadline_ts = node.deadline.timestamp() if node.deadline else float('inf')
                    cost = node.cost

                    heapq.heappush(
                        self.ready_queue,
                        (-priority, deadline_ts, -cost, task_id)
                    )

    def mark_running(
        self,
        task_id: str,
        worker_id: Optional[str] = None,
        attempt_no: Optional[int] = None
    ) -> None:
        """Mark task as running.

        Args:
            task_id: Task ID
            worker_id: Optional worker ID executing the task
            attempt_no: Optional attempt number (for retry tracking)
        """
        node = self.nodes[task_id]
        node.state = TaskState.RUNNING
        node.started_at = datetime.now(timezone.utc)

        if worker_id:
            node.worker_id = worker_id

        self.running_tasks.add(task_id)
        self.action_concurrency[node.action_name] += 1

        logger.info(f"Task {task_id}: READY → RUNNING (attempt {attempt_no or 1})")

    def mark_completed(
        self,
        task_id: str,
        success: bool = True,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        error_class: Optional[str] = None
    ) -> None:
        """Mark task as completed (succeeded or failed).

        Args:
            task_id: Task ID
            success: True for success, False for failure (default: True)
            result: Optional result artifacts dictionary
            error: Optional error message
            error_class: Optional error class name
        """
        node = self.nodes[task_id]
        node.completed_at = datetime.now(timezone.utc)

        if success:
            node.state = TaskState.SUCCEEDED
            self.completed.add(task_id)

            # Store result artifacts if provided
            if result:
                node.normalized_args["_result"] = result

            # Update circuit breaker
            breaker = self._get_breaker(node.action_name)
            breaker.record_success()

            logger.info(f"Task {task_id}: RUNNING → SUCCEEDED")
        else:
            node.state = TaskState.FAILED
            node.last_error = error
            node.error_class = error_class
            self.failed.add(task_id)

            # Update circuit breaker
            breaker = self._get_breaker(node.action_name)
            breaker.record_failure(error_class or "unknown")

            logger.error(f"Task {task_id}: RUNNING → FAILED ({error_class})")

        # Cleanup
        self.running_tasks.discard(task_id)
        self.action_concurrency[node.action_name] -= 1

    def mark_failed(
        self,
        task_id: str,
        error: str,
        error_class: str,
        retry_scheduled_ms: Optional[int] = None
    ) -> None:
        """Mark task as failed.

        Args:
            task_id: Task ID
            error: Error message
            error_class: Error class name
            retry_scheduled_ms: If set, indicates retry is scheduled in N milliseconds
        """
        node = self.nodes[task_id]
        node.state = TaskState.FAILED
        node.last_error = error
        node.error_class = error_class
        node.completed_at = datetime.now(timezone.utc)
        node.retry_count += 1

        self.failed.add(task_id)

        # Update circuit breaker
        breaker = self._get_breaker(node.action_name)
        breaker.record_failure(error_class or "unknown")

        # Cleanup running state
        if task_id in self.running_tasks:
            self.running_tasks.discard(task_id)
            self.action_concurrency[node.action_name] -= 1

        if retry_scheduled_ms:
            logger.warning(
                f"Task {task_id}: RUNNING → FAILED ({error_class}), "
                f"retry in {retry_scheduled_ms}ms"
            )
        else:
            logger.error(f"Task {task_id}: RUNNING → FAILED ({error_class})")

    def schedule_retry(self, task_id: str, next_run_at: float) -> None:
        """Schedule task for retry.

        Args:
            task_id: Task ID
            next_run_at: Monotonic timestamp when task should be retried
        """
        node = self.nodes[task_id]

        # Store next run time (we'll check this in get_ready_tasks)
        node.next_run_at = next_run_at

        # Move from failed back to pending for retry
        self.failed.discard(task_id)
        node.state = TaskState.PENDING

        logger.info(f"Task {task_id} scheduled for retry at {next_run_at}")

    def can_retry(self, task: "TaskNode") -> bool:
        """Check if task can be retried based on graph-level budget and task limits.

        Args:
            task: Task to check

        Returns:
            True if retry is allowed
        """
        # Check task-level retry limit
        if task.retry_count >= task.max_retries:
            return False

        # Check graph-level retry budget
        if self.retry_tokens_used >= self.max_retry_tokens:
            return False

        return True

    def mark_aborted(self, task_id: str, reason: str) -> None:
        """Mark task as aborted by safety envelope."""
        node = self.nodes[task_id]
        node.state = TaskState.ABORTED
        node.last_error = f"Aborted: {reason}"
        node.completed_at = datetime.now(timezone.utc)

        self.aborted.add(task_id)

        if task_id in self.running_tasks:
            self.running_tasks.discard(task_id)
            self.action_concurrency[node.action_name] -= 1

        logger.warning(f"Task {task_id}: ABORTED ({reason})")

    def mark_cancelled(self, task_id: str) -> None:
        """Mark task as cancelled."""
        node = self.nodes[task_id]
        node.state = TaskState.CANCELLED
        node.completed_at = datetime.now(timezone.utc)

        if task_id in self.running_tasks:
            self.running_tasks.discard(task_id)
            self.action_concurrency[node.action_name] -= 1

        logger.info(f"Task {task_id}: CANCELLED")

    def _get_breaker(self, action_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for action."""
        if action_name not in self.breakers:
            self.breakers[action_name] = CircuitBreaker(action_name)
        return self.breakers[action_name]

    def use_retry_token(self, task_id: str) -> bool:
        """
        Try to use a retry token.

        Returns:
            True if token available, False if budget exhausted
        """
        if self.retry_tokens_used >= self.max_retry_tokens:
            logger.warning(f"Retry token budget exhausted for graph {self.graph_id}")
            return False

        self.retry_tokens_used += 1
        node = self.nodes[task_id]
        node.retry_tokens_used += 1

        return True

    def is_complete(self) -> bool:
        """Check if all tasks are in terminal state."""
        terminal_states = {
            TaskState.SUCCEEDED,
            TaskState.FAILED,
            TaskState.ABORTED,
            TaskState.SKIPPED,
            TaskState.CANCELLED
        }
        return all(node.state in terminal_states for node in self.nodes.values())

    def has_timed_out(self) -> bool:
        """Check if graph has exceeded timeout."""
        if not self.started_at:
            return False

        elapsed_ms = (datetime.now(timezone.utc) - self.started_at).total_seconds() * 1000
        return elapsed_ms > self.graph_timeout_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {
            "total_tasks": len(self.nodes),
            "states": defaultdict(int),
            "retry_tokens_used": self.retry_tokens_used,
            "running_tasks": len(self.running_tasks),
            "action_concurrency": dict(self.action_concurrency),
            "breaker_states": {
                action: breaker.get_state()
                for action, breaker in self.breakers.items()
            }
        }

        for node in self.nodes.values():
            stats["states"][node.state.value] += 1

        stats["states"] = dict(stats["states"])

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire graph for persistence."""
        return {
            "graph_id": self.graph_id,
            "graph_timeout_ms": self.graph_timeout_ms,
            "max_retry_tokens": self.max_retry_tokens,
            "max_parallel": self.max_parallel,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "nodes": {task_id: node.to_dict() for task_id, node in self.nodes.items()},
            "retry_tokens_used": self.retry_tokens_used,
            "stats": self.get_stats()
        }
