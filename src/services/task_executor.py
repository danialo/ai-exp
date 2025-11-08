"""
Task Executor - Executes individual tasks with retry, idempotency, and safety checks.

This module provides robust task execution with:
- Exponential backoff with jitter for retries
- Idempotency key checking to prevent duplicate execution
- Safety envelope integration for abort conditions
- Circuit breaker integration
- Decision recording for parameter adaptation
- Timeout enforcement
- Error classification (transient vs permanent)
"""

import asyncio
import hashlib
import json
import logging
import random
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.services.task_graph import TaskNode, CircuitBreaker

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    ABORTED = "aborted"
    TIMEOUT = "timeout"
    DUPLICATE = "duplicate"


class ErrorClass(str, Enum):
    """Error classification for retry decisions."""
    TRANSIENT = "transient"  # Retry
    PERMANENT = "permanent"  # Don't retry
    TIMEOUT = "timeout"      # Retry with longer timeout
    SAFETY = "safety"        # Don't retry (abort condition)


@dataclass
class TaskExecutionResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    success: bool
    error: Optional[str] = None
    error_class: Optional[ErrorClass] = None
    retryable: bool = False
    aborted: bool = False
    abort_reason: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TaskExecutor:
    """
    Executes individual tasks with retry logic and safety checks.

    Features:
    - Exponential backoff with jitter
    - Idempotency checking
    - Safety envelope integration
    - Circuit breaker integration
    - Decision recording
    - Timeout enforcement
    - Error classification
    """

    def __init__(
        self,
        abort_monitor: Optional[Any] = None,
        decision_registry: Optional[Any] = None,
        raw_store: Optional[Any] = None,
        max_retries: int = 3,
        base_delay_ms: int = 1000,
        max_delay_ms: int = 20000,
        jitter_ms: int = 500,
        default_timeout_ms: int = 300000,  # 5 minutes
    ):
        """
        Initialize task executor.

        Args:
            abort_monitor: Safety envelope monitor
            decision_registry: Decision framework registry
            raw_store: Raw experience store
            max_retries: Maximum retry attempts per task
            base_delay_ms: Base delay for exponential backoff (default 1s)
            max_delay_ms: Maximum delay for backoff (default 20s)
            jitter_ms: Random jitter to add to delays (default 500ms)
            default_timeout_ms: Default task timeout (default 5min)
        """
        self.abort_monitor = abort_monitor
        self.decision_registry = decision_registry
        self.raw_store = raw_store

        # Retry configuration
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.jitter_ms = jitter_ms
        self.default_timeout_ms = default_timeout_ms

        # Idempotency cache (in-memory for now)
        self._execution_cache: Dict[str, TaskExecutionResult] = {}

    async def execute(
        self,
        node: TaskNode,
        task_callable: Any,
        circuit_breaker: Optional[CircuitBreaker] = None,
        idempotency_key: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> TaskExecutionResult:
        """
        Execute a task with full safety and retry logic.

        Args:
            node: TaskNode with task metadata
            task_callable: Async callable to execute
            circuit_breaker: Circuit breaker for this action type
            idempotency_key: Optional idempotency key (uses node's key if not provided)
            timeout_ms: Optional timeout override

        Returns:
            TaskExecutionResult with execution status and metadata
        """
        # Use node's idempotency key if not provided
        if idempotency_key is None:
            idempotency_key = node.idempotency_key

        # Check for duplicate execution
        if idempotency_key and self._is_duplicate(idempotency_key):
            logger.info(f"Task {node.task_id}: Skipping duplicate execution (key={idempotency_key[:16]}...)")
            return self._get_cached_result(idempotency_key)

        # Check circuit breaker
        if circuit_breaker and circuit_breaker.is_open():
            logger.warning(f"Task {node.task_id}: Circuit breaker open for {node.action_name}")
            return TaskExecutionResult(
                task_id=node.task_id,
                status=TaskStatus.ABORTED,
                success=False,
                error=f"Circuit breaker open for {node.action_name}",
                error_class=ErrorClass.SAFETY,
                aborted=True,
                abort_reason="circuit_breaker_open"
            )

        # Execute with retry logic
        result = await self._execute_with_retry(
            node=node,
            task_callable=task_callable,
            circuit_breaker=circuit_breaker,
            timeout_ms=timeout_ms or node.task_timeout_ms or self.default_timeout_ms
        )

        # Cache result for idempotency
        if idempotency_key:
            self._cache_result(idempotency_key, result)

        return result

    async def _execute_with_retry(
        self,
        node: TaskNode,
        task_callable: Any,
        circuit_breaker: Optional[CircuitBreaker],
        timeout_ms: int
    ) -> TaskExecutionResult:
        """Execute task with exponential backoff retry."""
        last_error = None
        last_error_class = None

        for attempt in range(node.max_retries + 1):
            # Check safety envelope before each attempt
            if self.abort_monitor and self.abort_monitor.should_abort():
                abort_reason = self.abort_monitor.get_abort_reason()
                logger.warning(f"Task {node.task_id}: Aborted by safety envelope - {abort_reason}")
                return TaskExecutionResult(
                    task_id=node.task_id,
                    status=TaskStatus.ABORTED,
                    success=False,
                    error=f"Aborted by safety envelope: {abort_reason}",
                    error_class=ErrorClass.SAFETY,
                    aborted=True,
                    abort_reason=abort_reason
                )

            # Record decision (if first attempt and decision registry available)
            decision_record_id = None
            if attempt == 0 and self.decision_registry:
                try:
                    decision_record_id = self.decision_registry.record_decision(
                        decision_id="task_execution",
                        context={
                            "task_id": node.task_id,
                            "action_name": node.action_name,
                            "attempt": attempt
                        },
                        parameters_used=self.decision_registry.get_all_parameters("task_execution") or {}
                    )
                except Exception as e:
                    logger.warning(f"Failed to record decision: {e}")

            # Execute attempt
            try:
                logger.info(f"Task {node.task_id}: Executing attempt {attempt + 1}/{node.max_retries + 1}")
                result = await self._execute_single_attempt(
                    node=node,
                    task_callable=task_callable,
                    timeout_ms=timeout_ms,
                    attempt=attempt
                )

                # Success!
                if result.success:
                    # Record success to circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_success()

                    # Record successful outcome to decision registry
                    if decision_record_id and self.decision_registry:
                        try:
                            self.decision_registry.record_outcome(
                                record_id=decision_record_id,
                                success_score=1.0,
                                outcome_details={
                                    "task_id": node.task_id,
                                    "execution_time_ms": result.execution_time_ms,
                                    "attempt": attempt
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Failed to record outcome: {e}")

                    return result

                # Failed - classify error
                last_error = result.error
                last_error_class = result.error_class

                # Record failure to circuit breaker
                if circuit_breaker and result.error_class:
                    circuit_breaker.record_failure(result.error_class.value)

                # Don't retry if error is permanent or safety-related
                if not result.retryable:
                    logger.warning(f"Task {node.task_id}: Non-retryable error ({result.error_class})")
                    return result

                # Calculate delay with jitter for next retry
                if attempt < node.max_retries:
                    delay_ms = self._calculate_backoff_delay(attempt)
                    logger.info(f"Task {node.task_id}: Retrying in {delay_ms}ms...")
                    await asyncio.sleep(delay_ms / 1000.0)

            except asyncio.TimeoutError:
                last_error = f"Task timeout after {timeout_ms}ms"
                last_error_class = ErrorClass.TIMEOUT
                logger.warning(f"Task {node.task_id}: Timeout on attempt {attempt + 1}")

                if circuit_breaker:
                    circuit_breaker.record_failure(ErrorClass.TIMEOUT.value)

                if attempt < node.max_retries:
                    delay_ms = self._calculate_backoff_delay(attempt)
                    await asyncio.sleep(delay_ms / 1000.0)

            except Exception as e:
                # Unexpected error
                last_error = f"Unexpected error: {str(e)}"
                last_error_class = ErrorClass.PERMANENT
                logger.error(f"Task {node.task_id}: Unexpected error - {e}\n{traceback.format_exc()}")

                if circuit_breaker:
                    circuit_breaker.record_failure(ErrorClass.PERMANENT.value)

                # Don't retry unexpected errors
                break

        # All retries exhausted
        logger.error(f"Task {node.task_id}: Failed after {node.max_retries + 1} attempts")
        return TaskExecutionResult(
            task_id=node.task_id,
            status=TaskStatus.FAILED,
            success=False,
            error=last_error or "Unknown error",
            error_class=last_error_class or ErrorClass.PERMANENT,
            retryable=False
        )

    async def _execute_single_attempt(
        self,
        node: TaskNode,
        task_callable: Any,
        timeout_ms: int,
        attempt: int
    ) -> TaskExecutionResult:
        """Execute a single task attempt with timeout."""
        started_at = datetime.now(timezone.utc)

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                task_callable(node.normalized_args),
                timeout=timeout_ms / 1000.0
            )

            completed_at = datetime.now(timezone.utc)
            execution_time_ms = (completed_at - started_at).total_seconds() * 1000

            # Interpret result
            if isinstance(result, dict):
                success = result.get("success", True)
                error = result.get("error")
            else:
                success = True
                error = None

            # Record to raw store if available
            if self.raw_store:
                try:
                    self.raw_store.create_task_execution(
                        task_id=node.task_id,
                        task_type=node.action_name,
                        status="success" if success else "failed",
                        started_at=started_at.isoformat(),
                        completed_at=completed_at.isoformat(),
                        error_message=error,
                        idempotency_key=node.idempotency_key,
                        metadata={
                            "attempt": attempt,
                            "execution_time_ms": execution_time_ms
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to record task execution: {e}")

            return TaskExecutionResult(
                task_id=node.task_id,
                status=TaskStatus.SUCCESS if success else TaskStatus.FAILED,
                success=success,
                error=error,
                error_class=self._classify_error(error) if error else None,
                retryable=self._is_retryable(error) if error else False,
                execution_time_ms=execution_time_ms,
                metadata={"attempt": attempt, "result": result}
            )

        except asyncio.TimeoutError:
            raise  # Let caller handle timeout

        except Exception as e:
            completed_at = datetime.now(timezone.utc)
            execution_time_ms = (completed_at - started_at).total_seconds() * 1000

            error_str = str(e)
            error_class = self._classify_error(e)

            return TaskExecutionResult(
                task_id=node.task_id,
                status=TaskStatus.FAILED,
                success=False,
                error=error_str,
                error_class=error_class,
                retryable=self._is_retryable(e),
                execution_time_ms=execution_time_ms,
                metadata={"attempt": attempt, "traceback": traceback.format_exc()}
            )

    def _calculate_backoff_delay(self, attempt: int) -> int:
        """
        Calculate exponential backoff delay with jitter.

        Formula: min(base_delay * 2^attempt, max_delay) + random_jitter
        """
        delay = min(self.base_delay_ms * (2 ** attempt), self.max_delay_ms)
        jitter = random.randint(0, self.jitter_ms)
        return delay + jitter

    def _classify_error(self, error: Any) -> ErrorClass:
        """Classify error type for retry decisions."""
        error_str = str(error).lower()

        # Safety/abort errors
        if any(kw in error_str for kw in ["abort", "safety", "envelope"]):
            return ErrorClass.SAFETY

        # Timeout errors
        if any(kw in error_str for kw in ["timeout", "timed out"]):
            return ErrorClass.TIMEOUT

        # Transient network errors
        if any(kw in error_str for kw in ["connection", "network", "temporary", "unavailable"]):
            return ErrorClass.TRANSIENT

        # Permanent errors
        if any(kw in error_str for kw in ["not found", "invalid", "forbidden", "unauthorized"]):
            return ErrorClass.PERMANENT

        # Default to transient (retry)
        return ErrorClass.TRANSIENT

    def _is_retryable(self, error: Any) -> bool:
        """Determine if error is retryable."""
        error_class = self._classify_error(error)
        return error_class in {ErrorClass.TRANSIENT, ErrorClass.TIMEOUT}

    def _is_duplicate(self, idempotency_key: str) -> bool:
        """Check if task already executed with this key."""
        # Check in-memory cache first
        if idempotency_key in self._execution_cache:
            return True

        # Check raw store if available
        if self.raw_store:
            try:
                executions = self.raw_store.list_task_executions(
                    idempotency_key=idempotency_key,
                    limit=1
                )
                return len(executions) > 0
            except Exception as e:
                logger.warning(f"Failed to check idempotency: {e}")
                return False

        return False

    def _get_cached_result(self, idempotency_key: str) -> TaskExecutionResult:
        """Get cached result for duplicate task."""
        if idempotency_key in self._execution_cache:
            cached = self._execution_cache[idempotency_key]
            # Create new result with DUPLICATE status
            return TaskExecutionResult(
                task_id=cached.task_id,
                status=TaskStatus.DUPLICATE,
                success=cached.success,
                error=cached.error,
                error_class=cached.error_class,
                execution_time_ms=0.0,
                metadata={"cached": True, "original_result": cached.metadata}
            )

        # Fallback: create generic duplicate result
        return TaskExecutionResult(
            task_id="unknown",
            status=TaskStatus.DUPLICATE,
            success=True,
            metadata={"cached": True, "from_raw_store": True}
        )

    def _cache_result(self, idempotency_key: str, result: TaskExecutionResult) -> None:
        """Cache execution result for idempotency."""
        self._execution_cache[idempotency_key] = result

        # Limit cache size (simple LRU)
        if len(self._execution_cache) > 1000:
            # Remove oldest 10%
            keys_to_remove = list(self._execution_cache.keys())[:100]
            for key in keys_to_remove:
                del self._execution_cache[key]


# Factory function
def create_task_executor(
    abort_monitor: Optional[Any] = None,
    decision_registry: Optional[Any] = None,
    raw_store: Optional[Any] = None,
    **kwargs
) -> TaskExecutor:
    """
    Factory function to create TaskExecutor.

    Args:
        abort_monitor: Safety envelope monitor
        decision_registry: Decision framework registry
        raw_store: Raw experience store
        **kwargs: Additional configuration options

    Returns:
        TaskExecutor instance
    """
    return TaskExecutor(
        abort_monitor=abort_monitor,
        decision_registry=decision_registry,
        raw_store=raw_store,
        **kwargs
    )
