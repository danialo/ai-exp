"""Code modification executor for file operations."""

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .base import RunContext, ExecutionResult

if TYPE_CHECKING:
    from src.services.task_graph import TaskNode
    from src.services.code_access import CodeAccessService

logger = logging.getLogger(__name__)


class CodeModificationExecutor:
    """Executor for code modification tasks.

    Handles: modify_code, create_file, delete_file
    Uses CodeAccessService for safe file operations.
    """

    actions = {"modify_code", "create_file", "delete_file"}

    def __init__(self, code_access: "CodeAccessService"):
        """Initialize executor.

        Args:
            code_access: CodeAccessService for file operations
        """
        self.code_access = code_access

    def can_handle(self, action_name: str) -> bool:
        """Check if this executor handles the action."""
        return action_name in self.actions

    async def admit(self, task: "TaskNode", ctx: RunContext) -> tuple[bool, str]:
        """Static admission check.

        Validates:
        - Required fields present (file_path, content/reason)
        - File path is accessible

        Args:
            task: Task to check
            ctx: Execution context

        Returns:
            (admitted, reason)
        """
        args = task.normalized_args

        # Check required fields
        if "file_path" not in args:
            return False, "missing_file_path"

        if task.action_name in ("modify_code", "create_file"):
            if "content" not in args:
                return False, "missing_content"
            if "reason" not in args:
                return False, "missing_reason"

        # Check file access
        file_path = args["file_path"]
        can_access, reason = self.code_access.can_access(file_path)

        if not can_access:
            return False, f"access_denied:{reason}"

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
        """Execute code modification.

        Args:
            task: Task to execute
            ctx: Execution context

        Returns:
            ExecutionResult with diff in artifacts
        """
        args = task.normalized_args
        file_path = args["file_path"]
        action = task.action_name

        try:
            if action in ("modify_code", "create_file"):
                content = args["content"]
                reason = args["reason"]
                goal_id = args.get("goal_id", "unknown")

                modification, error = await self.code_access.modify_file(
                    file_path=file_path,
                    new_content=content,
                    reason=reason,
                    goal_id=goal_id
                )

                if error:
                    return ExecutionResult(
                        success=False,
                        error=error,
                        error_class="ModificationError",
                        retryable=False
                    )

                return ExecutionResult(
                    success=True,
                    stdout=f"Modified {file_path}",
                    artifacts={
                        "diff": modification.diff if modification else "",
                        "modification_id": modification.id if modification else None,
                        "file_path": file_path
                    }
                )

            elif action == "delete_file":
                # For Phase 1, simple delete
                abs_path = Path(ctx.workdir) / file_path

                if not abs_path.exists():
                    return ExecutionResult(
                        success=False,
                        error=f"file_not_found:{file_path}",
                        error_class="FileNotFoundError",
                        retryable=False
                    )

                abs_path.unlink()

                return ExecutionResult(
                    success=True,
                    stdout=f"Deleted {file_path}",
                    artifacts={"file_path": file_path}
                )

            else:
                return ExecutionResult(
                    success=False,
                    error=f"unknown_action:{action}",
                    error_class="ConfigError",
                    retryable=False
                )

        except Exception as e:
            logger.error(f"Code modification failed: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=str(e),
                error_class=e.__class__.__name__,
                retryable=True,
                backoff_ms=2000
            )

    async def postcondition(
        self,
        task: "TaskNode",
        ctx: RunContext,
        res: ExecutionResult
    ) -> tuple[bool, str]:
        """Validate execution result.

        For Phase 1, check Python syntax if it's a .py file.

        Args:
            task: Executed task
            ctx: Execution context
            res: Execution result

        Returns:
            (valid, reason)
        """
        if not res.success:
            return True, ""  # Already failed, no need to validate

        file_path = task.normalized_args.get("file_path", "")

        # Only check Python files for syntax
        if file_path.endswith(".py"):
            abs_path = Path(ctx.workdir) / file_path

            if abs_path.exists():
                try:
                    result = subprocess.run(
                        ["python3", "-m", "py_compile", str(abs_path)],
                        cwd=ctx.workdir,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

                    if result.returncode != 0:
                        return False, f"syntax_error:{result.stderr[:200]}"

                except subprocess.TimeoutExpired:
                    return False, "syntax_check_timeout"
                except Exception as e:
                    logger.warning(f"Syntax check failed: {e}")
                    # Don't fail on syntax check errors
                    pass

        return True, ""
