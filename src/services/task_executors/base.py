"""Base interfaces for task executors.

Defines unified task context, execution results, and executor protocol
with three safety hooks: admit, preflight, postcondition.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Callable


@dataclass
class RunContext:
    """Unified task execution context passed to all executors.

    Attributes:
        trace_id: Distributed trace ID for correlation
        span_id: Unique span ID for this task execution
        workdir: Working directory for task execution
        timeout_ms: Task execution timeout in milliseconds
        env: Environment variables for subprocess execution
        monotonic: Monotonic clock function (time.monotonic)
        ledger: IdentityLedger client for event emission
        breaker: Circuit breaker registry
        caps: Per-action and per-resource concurrency caps
    """
    trace_id: str
    span_id: str
    workdir: str
    timeout_ms: int
    env: Dict[str, str]
    monotonic: Callable[[], float]
    ledger: Any  # IdentityLedger client
    breaker: Any  # Circuit breaker registry
    caps: Dict[str, int]  # per-action, per-resource caps


@dataclass
class ExecutionResult:
    """Deterministic result shape for all executors.

    Attributes:
        success: True if task completed successfully
        stdout: Standard output captured during execution
        stderr: Standard error captured during execution
        artifacts: Dictionary of execution artifacts (diffs, files, metrics)
        error: Human-readable error message if failed
        error_class: Error class name for categorization
        retryable: True if this failure is retryable
        backoff_ms: Suggested backoff duration for retry (0 = use default)
    """
    success: bool
    stdout: str = ""
    stderr: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_class: Optional[str] = None
    retryable: bool = False
    backoff_ms: int = 0


class TaskExecutor(Protocol):
    """Protocol for task executors with three safety hooks.

    Every executor must implement:
    - admit(): Static checks before execution
    - preflight(): Runtime quotas, breaker state
    - execute(): Actual task execution
    - postcondition(): Result validation

    Attributes:
        actions: Set of action names this executor can handle
    """
    actions: set[str]

    def can_handle(self, action_name: str) -> bool:
        """Check if this executor can handle the given action.

        Args:
            action_name: Action name from task node

        Returns:
            True if this executor handles this action
        """
        return action_name in self.actions

    async def admit(self, task: "TaskNode", ctx: RunContext) -> tuple[bool, str]:
        """Static admission check before execution.

        Validates task parameters, required files exist, etc.
        Called before preflight and execute.

        Args:
            task: Task node to execute
            ctx: Execution context

        Returns:
            (admitted, reason): (True, "") if admitted, (False, reason) if rejected
        """
        ...

    async def preflight(self, task: "TaskNode", ctx: RunContext) -> tuple[bool, str]:
        """Runtime preflight check before execution.

        Checks breaker state, resource availability, quotas.
        Called after admit, before execute.

        Args:
            task: Task node to execute
            ctx: Execution context

        Returns:
            (ready, reason): (True, "") if ready, (False, reason) if blocked
        """
        ...

    async def execute(self, task: "TaskNode", ctx: RunContext) -> ExecutionResult:
        """Execute the task.

        Args:
            task: Task node to execute
            ctx: Execution context

        Returns:
            ExecutionResult with success status and artifacts
        """
        ...

    async def postcondition(
        self,
        task: "TaskNode",
        ctx: RunContext,
        res: ExecutionResult
    ) -> tuple[bool, str]:
        """Validate execution result.

        Checks output correctness, side effects, state consistency.
        Called after execute, can fail a successful execution.

        Args:
            task: Task node that was executed
            ctx: Execution context
            res: Execution result from execute()

        Returns:
            (valid, reason): (True, "") if valid, (False, reason) if invalid
        """
        ...
