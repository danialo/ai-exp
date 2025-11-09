"""Shell command executor for running bash commands."""

import asyncio
import logging
from typing import TYPE_CHECKING

from .base import RunContext, ExecutionResult

if TYPE_CHECKING:
    from src.services.task_graph import TaskNode

logger = logging.getLogger(__name__)


class ShellCommandExecutor:
    """Executor for shell command tasks.

    Handles: shell_command, bash
    Runs shell commands with timeout and captures output.
    """

    actions = {"shell_command", "bash"}

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
        """Execute shell command.

        Args:
            task: Task to execute
            ctx: Execution context

        Returns:
            ExecutionResult with command output
        """
        args = task.normalized_args
        cmd = args["cmd"]

        try:
            # Run with timeout
            timeout_seconds = ctx.timeout_ms / 1000.0

            # Run through shell for proper command parsing
            process = await asyncio.create_subprocess_shell(
                cmd,
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
                    error=f"command_timeout:{timeout_seconds}s",
                    error_class="TimeoutError",
                    retryable=True,
                    backoff_ms=5000
                )

            # Success if exit code is 0
            success = (returncode == 0)

            return ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                artifacts={
                    "exit_code": returncode,
                    "cmd": cmd
                },
                error=None if success else f"command_failed:exit_code_{returncode}",
                error_class=None if success else "CommandFailure",
                retryable=(returncode != 0),  # Non-zero exits are potentially retryable
                backoff_ms=3000
            )

        except Exception as e:
            logger.error(f"Shell command execution failed: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=str(e),
                error_class=e.__class__.__name__,
                retryable=True,
                backoff_ms=4000
            )

    async def postcondition(
        self,
        task: "TaskNode",
        ctx: RunContext,
        res: ExecutionResult
    ) -> tuple[bool, str]:
        """Validate execution result.

        For Phase 1, no additional validation needed for shell commands.
        The exit code determines success/failure.

        Args:
            task: Executed task
            ctx: Execution context
            res: Execution result

        Returns:
            (valid, reason)
        """
        return True, ""
