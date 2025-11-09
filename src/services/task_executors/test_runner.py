"""Test execution executor for running test suites."""

import asyncio
import logging
from typing import TYPE_CHECKING

from .base import RunContext, ExecutionResult

if TYPE_CHECKING:
    from src.services.task_graph import TaskNode

logger = logging.getLogger(__name__)


class TestExecutor:
    """Executor for test running tasks.

    Handles: run_tests, pytest, npm_test
    Runs test commands with timeout and captures output.
    """

    actions = {"run_tests", "pytest", "npm_test"}

    def can_handle(self, action_name: str) -> bool:
        """Check if this executor handles the action."""
        return action_name in self.actions

    async def admit(self, task: "TaskNode", ctx: RunContext) -> tuple[bool, str]:
        """Static admission check.

        Validates:
        - Command field present
        - Command is not empty

        Args:
            task: Task to check
            ctx: Execution context

        Returns:
            (admitted, reason)
        """
        args = task.normalized_args

        if "cmd" not in args:
            return False, "missing_cmd"

        cmd = args["cmd"]
        if not cmd or (isinstance(cmd, str) and not cmd.strip()):
            return False, "empty_cmd"

        return True, ""

    async def preflight(self, task: "TaskNode", ctx: RunContext) -> tuple[bool, str]:
        """Runtime preflight check.

        For Phase 1, just check breaker state.

        Args:
            task: Task to check
            ctx: Execution context

        Returns:
            (ready, reason)
        """
        # Check breaker state
        if ctx.breaker and hasattr(ctx.breaker, "is_open"):
            if ctx.breaker.is_open(task.action_name):
                return False, "circuit_breaker_open"

        return True, ""

    async def execute(self, task: "TaskNode", ctx: RunContext) -> ExecutionResult:
        """Execute test command.

        Args:
            task: Task to execute
            ctx: Execution context

        Returns:
            ExecutionResult with test output
        """
        args = task.normalized_args
        cmd = args["cmd"]

        # Convert command to list if string
        if isinstance(cmd, str):
            cmd_list = cmd.split()
        else:
            cmd_list = cmd

        try:
            # Run with timeout
            timeout_seconds = ctx.timeout_ms / 1000.0

            process = await asyncio.create_subprocess_exec(
                *cmd_list,
                cwd=ctx.workdir,
                env=ctx.env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds
                )

                stdout = stdout_bytes.decode('utf-8', errors='replace')
                stderr = stderr_bytes.decode('utf-8', errors='replace')
                returncode = process.returncode

            except asyncio.TimeoutError:
                # Kill process on timeout
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass

                return ExecutionResult(
                    success=False,
                    error=f"test_timeout:{timeout_seconds}s",
                    error_class="TimeoutError",
                    retryable=False  # Tests shouldn't be retried
                )

            # Success if exit code is 0
            success = (returncode == 0)

            return ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                artifacts={
                    "exit_code": returncode,
                    "cmd": " ".join(cmd_list)
                },
                error=None if success else f"tests_failed:exit_code_{returncode}",
                error_class=None if success else "TestFailure",
                retryable=False  # Don't retry failed tests
            )

        except Exception as e:
            logger.error(f"Test execution failed: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=str(e),
                error_class=e.__class__.__name__,
                retryable=True,  # Infrastructure errors are retryable
                backoff_ms=3000
            )

    async def postcondition(
        self,
        task: "TaskNode",
        ctx: RunContext,
        res: ExecutionResult
    ) -> tuple[bool, str]:
        """Validate execution result.

        For Phase 1, no additional validation needed for tests.
        The exit code determines success/failure.

        Args:
            task: Executed task
            ctx: Execution context
            res: Execution result

        Returns:
            (valid, reason)
        """
        return True, ""
