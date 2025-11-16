"""Task execution engine with retry logic and circuit breakers.

Runs TaskGraph tasks with:
- Exponential backoff with jitter
- Circuit breaker integration
- Concurrency control
- IdentityLedger event emission
"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Callable, Optional, Any
from uuid import uuid4

from src.services.task_executors.base import RunContext, ExecutionResult, TaskExecutor

logger = logging.getLogger(__name__)

# Retry constants
JITTER_MAX = 500
MAX_BACKOFF_MS = 20000


def next_delay_ms(attempt: int) -> int:
    """Calculate next retry delay with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-indexed)

    Returns:
        Delay in milliseconds
    """
    base_delay = (2 ** attempt) * 2000
    capped_delay = min(base_delay, MAX_BACKOFF_MS)
    jitter = random.randint(0, JITTER_MAX)
    return capped_delay + jitter


class TaskExecutionEngine:
    """Async task execution engine with retry and breaker support.

    Features:
    - Concurrent task execution with semaphore control
    - Exponential backoff with jitter for retries
    - Circuit breaker integration
    - Identity ledger event emission
    - Deterministic state transitions
    """

    def __init__(
        self,
        graph: "TaskGraph",
        executors: List[TaskExecutor],
        max_concurrent: int = 4,
        ledger: Optional[Any] = None
    ):
        """Initialize execution engine.

        Args:
            graph: TaskGraph to execute
            executors: List of task executors
            max_concurrent: Maximum concurrent tasks
            ledger: Optional IdentityLedger for event emission
        """
        self.graph = graph
        self.execs = executors
        self.max_concurrent = max_concurrent
        self.ledger = ledger

        # Running tasks tracked by task_id
        self.running: Dict[str, asyncio.Task] = {}

        # Concurrency semaphore
        self.sem = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"TaskExecutionEngine initialized: max_concurrent={max_concurrent}, "
            f"executors={[e.__class__.__name__ for e in executors]}"
        )

    def _pick_executor(self, action: str) -> Optional[TaskExecutor]:
        """Find executor that can handle the given action.

        Args:
            action: Action name from task node

        Returns:
            Executor if found, None otherwise
        """
        for ex in self.execs:
            if ex.can_handle(action):
                return ex
        return None

    async def run(self, ctx_factory: Callable[["TaskNode"], RunContext]):
        """Main execution loop.

        Runs until graph is complete or all tasks blocked.

        Args:
            ctx_factory: Factory function to create RunContext for each task
        """
        logger.info(f"Starting execution of graph {self.graph.graph_id}")

        while not self.graph.is_complete():
            # Get ready task IDs respecting concurrency caps
            ready_ids = self.graph.get_ready_tasks()

            # Filter out already running tasks
            ready_ids = [tid for tid in ready_ids if tid not in self.running]

            if ready_ids:
                # Start available tasks
                for task_id in ready_ids:
                    # Check if we have capacity
                    if len(self.running) >= self.max_concurrent:
                        break

                    # Get task node
                    task_node = self.graph.nodes[task_id]

                    # Find executor
                    ex = self._pick_executor(task_node.action_name)

                    if not ex:
                        logger.error(
                            f"No executor for action '{task_node.action_name}' "
                            f"in task {task_node.task_id}"
                        )
                        self.graph.mark_failed(
                            task_node.task_id,
                            error="no_executor",
                            error_class="ConfigError"
                        )
                        continue

                    # Create context
                    ctx = ctx_factory(task_node)

                    # Start task execution
                    await self._start_one(task_node, ex, ctx)

            # Wait for any task to complete
            if self.running:
                done, _ = await asyncio.wait(
                    self.running.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Clean up completed tasks
                for fut in done:
                    tid = getattr(fut, "_task_id", None)
                    if tid and tid in self.running:
                        del self.running[tid]
            else:
                # No tasks running and no tasks ready - check if truly complete
                if not self.graph.is_complete():
                    # Check if any tasks are scheduled for retry
                    import time
                    current_time = time.monotonic()
                    retry_scheduled = [
                        (tid, node.next_run_at)
                        for tid, node in self.graph.nodes.items()
                        if node.state.value == "pending" and node.next_run_at is not None
                    ]

                    if retry_scheduled:
                        # Find earliest retry time
                        earliest_retry_tid, earliest_retry_time = min(retry_scheduled, key=lambda x: x[1])
                        wait_time = max(0, earliest_retry_time - current_time)

                        logger.info(
                            f"Waiting {wait_time:.3f}s for scheduled retry of task {earliest_retry_tid}"
                        )
                        await asyncio.sleep(wait_time + 0.01)  # Small buffer
                        continue

                    # No retries scheduled - truly stalled
                    pending = [
                        tid for tid, node in self.graph.nodes.items()
                        if node.state.value in ("pending", "ready")
                    ]
                    if pending:
                        logger.warning(
                            f"Graph execution stalled: {len(pending)} tasks blocked: {pending[:5]}"
                        )
                    break

                # Small sleep to avoid busy loop
                await asyncio.sleep(0.05)

        logger.info(
            f"Graph {self.graph.graph_id} execution complete: "
            f"stats={self.graph.get_stats()}"
        )

    async def _start_one(
        self,
        task: "TaskNode",
        ex: TaskExecutor,
        ctx: RunContext
    ):
        """Start async execution of one task.

        Args:
            task: Task node to execute
            ex: Executor to use
            ctx: Execution context
        """
        async def _runner():
            """Inner runner with full lifecycle."""
            fut = asyncio.current_task()
            setattr(fut, "_task_id", task.task_id)

            # Emit task_started event
            if self.ledger:
                try:
                    await self.ledger.emit_event(
                        event_type="task_started",
                        data={
                            "task_id": task.task_id,
                            "action": task.action_name,
                            "trace_id": ctx.trace_id,
                            "span_id": ctx.span_id,
                            "attempt": task.retry_count + 1
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit task_started: {e}")

            # Mark running
            self.graph.mark_running(
                task.task_id,
                worker_id="engine",
                attempt_no=task.retry_count + 1
            )

            # Admit check
            ok, why = await ex.admit(task, ctx)
            if not ok:
                logger.warning(f"Task {task.task_id} admission rejected: {why}")
                self.graph.mark_failed(
                    task.task_id,
                    error=why,
                    error_class="AdmitRejected"
                )
                await self._emit_failure_event(task, ctx, why, "AdmitRejected")
                return

            # Preflight check
            ok, why = await ex.preflight(task, ctx)
            if not ok:
                logger.warning(f"Task {task.task_id} preflight rejected: {why}")
                self.graph.mark_failed(
                    task.task_id,
                    error=why,
                    error_class="PreflightRejected"
                )
                await self._emit_failure_event(task, ctx, why, "PreflightRejected")
                return

            # Execute
            try:
                res = await ex.execute(task, ctx)
            except Exception as e:
                logger.error(f"Task {task.task_id} execution raised: {e}", exc_info=True)
                res = ExecutionResult(
                    success=False,
                    error=str(e),
                    error_class=e.__class__.__name__,
                    retryable=True
                )

            # Postcondition check
            ok, why = await ex.postcondition(task, ctx, res)
            if not ok:
                logger.warning(f"Task {task.task_id} postcondition failed: {why}")
                res.success = False
                res.error = res.error or why
                res.error_class = res.error_class or "PostconditionFailed"

            # Handle result
            if res.success:
                # Success path
                self.graph.mark_completed(task.task_id, result=res.artifacts)

                # Emit task_succeeded event
                if self.ledger:
                    try:
                        await self.ledger.emit_event(
                            event_type="task_succeeded",
                            data={
                                "task_id": task.task_id,
                                "action": task.action_name,
                                "trace_id": ctx.trace_id,
                                "span_id": ctx.span_id,
                                "artifacts": res.artifacts
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to emit task_succeeded: {e}")

                logger.info(f"Task {task.task_id} succeeded")
                return

            # Failure path
            logger.warning(
                f"Task {task.task_id} failed: error={res.error}, "
                f"error_class={res.error_class}, retryable={res.retryable}"
            )

            # Check if retryable and budget available
            retry_ok = self.graph.can_retry(task)

            if res.retryable and retry_ok:
                # Check breaker state
                breaker_open = False
                if ctx.breaker and hasattr(ctx.breaker, "is_open"):
                    breaker_open = ctx.breaker.is_open(task.action_name)

                if not breaker_open:
                    # Schedule retry
                    delay = res.backoff_ms or next_delay_ms(task.retry_count)
                    next_run_at = ctx.monotonic() + (delay / 1000.0)

                    # Store error info before scheduling retry
                    task.last_error = res.error
                    task.error_class = res.error_class
                    task.retry_count += 1

                    self.graph.schedule_retry(task.task_id, next_run_at)

                    logger.info(
                        f"Task {task.task_id} scheduled for retry in {delay}ms "
                        f"(attempt {task.retry_count + 1})"
                    )

                    await self._emit_failure_event(
                        task, ctx, res.error, res.error_class, retry_scheduled=True
                    )
                    return

            # No retry - permanent failure
            # Trip breaker if needed
            if ctx.breaker and hasattr(ctx.breaker, "record_failure"):
                try:
                    ctx.breaker.record_failure(task.action_name)
                except Exception as e:
                    logger.warning(f"Failed to record breaker failure: {e}")

            self.graph.mark_failed(
                task.task_id,
                error=res.error,
                error_class=res.error_class
            )

            await self._emit_failure_event(task, ctx, res.error, res.error_class)

        # Create and store task
        self.running[task.task_id] = asyncio.create_task(_runner())

    async def _emit_failure_event(
        self,
        task: "TaskNode",
        ctx: RunContext,
        error: str,
        error_class: str,
        retry_scheduled: bool = False
    ):
        """Emit task_failed event to ledger.

        Args:
            task: Failed task
            ctx: Execution context
            error: Error message
            error_class: Error class name
            retry_scheduled: Whether retry is scheduled
        """
        if self.ledger:
            try:
                await self.ledger.emit_event(
                    event_type="task_failed",
                    data={
                        "task_id": task.task_id,
                        "action": task.action_name,
                        "trace_id": ctx.trace_id,
                        "span_id": ctx.span_id,
                        "error": error,
                        "error_class": error_class,
                        "retry_scheduled": retry_scheduled,
                        "attempt": task.retry_count + 1
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to emit task_failed: {e}")
