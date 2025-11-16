"""Tests for GoalExecutionService - End-to-end goal execution."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from dataclasses import dataclass

from src.services.goal_execution_service import (
    GoalExecutionService,
    GoalExecutionResult,
    DEFAULT_CODING_METHODS,
    DEFAULT_PRIMITIVE_TASKS
)
from src.services.task_executors.base import RunContext, ExecutionResult, TaskExecutor


# === Mock Executors for Testing ===

class MockCodeExecutor(TaskExecutor):
    """Mock executor for code operations."""
    actions = {"create_file", "modify_code", "delete_file"}

    def __init__(self):
        self.executed_tasks = []

    async def admit(self, task, ctx):
        return True, ""

    async def preflight(self, task, ctx):
        return True, ""

    async def execute(self, task, ctx):
        self.executed_tasks.append(task.task_id)
        await asyncio.sleep(0.001)  # Tiny delay to simulate work
        return ExecutionResult(
            success=True,
            stdout=f"Executed {task.action_name}",
            artifacts={"file_path": task.normalized_args.get("file_path", "unknown")}
        )

    async def postcondition(self, task, ctx, res):
        return True, ""


class MockTestExecutor(TaskExecutor):
    """Mock executor for test operations."""
    actions = {"run_tests", "pytest"}

    def __init__(self):
        self.tests_run = []

    async def admit(self, task, ctx):
        return True, ""

    async def preflight(self, task, ctx):
        return True, ""

    async def execute(self, task, ctx):
        self.tests_run.append(task.task_id)
        await asyncio.sleep(0.001)
        return ExecutionResult(
            success=True,
            stdout="All tests passed",
            artifacts={"tests_passed": 10, "tests_failed": 0}
        )

    async def postcondition(self, task, ctx, res):
        return True, ""


class MockShellExecutor(TaskExecutor):
    """Mock executor for shell commands."""
    actions = {"shell_command", "bash"}

    async def admit(self, task, ctx):
        return True, ""

    async def preflight(self, task, ctx):
        return True, ""

    async def execute(self, task, ctx):
        await asyncio.sleep(0.001)
        return ExecutionResult(success=True, stdout="command executed")

    async def postcondition(self, task, ctx, res):
        return True, ""


# === Mock CodeAccessService ===

@dataclass
class MockModification:
    """Mock file modification result."""
    id: str
    file_path: str
    diff: str
    success: bool = True


class MockCodeAccessService:
    """Mock CodeAccessService for testing."""

    def __init__(self):
        self.modifications = []
        self.files_created = []

    def can_access(self, file_path: str):
        """Check if file can be accessed."""
        # Allow all files for testing
        return True, ""

    async def modify_file(
        self,
        file_path: str,
        new_content: str,
        reason: str,
        goal_id: str = "test"
    ):
        """Mock file modification."""
        modification = MockModification(
            id=f"mod-{len(self.modifications)}",
            file_path=file_path,
            diff=f"+ {new_content[:50]}..."
        )

        self.modifications.append(modification)

        if not any(f == file_path for f in self.files_created):
            self.files_created.append(file_path)

        return modification, None


# === Tests ===

@pytest.mark.asyncio
async def test_execute_goal_implement_feature():
    """Test executing 'implement_feature' goal end-to-end.

    Flow:
    1. Service receives goal "implement_feature"
    2. HTN planner decomposes to: [create_file, create_file, run_tests]
    3. TaskGraph creates sequential dependency chain
    4. Execution engine runs tasks
    5. Result shows success with all tasks completed
    """
    code_access = MockCodeAccessService()
    executors = [MockCodeExecutor(), MockTestExecutor(), MockShellExecutor()]

    service = GoalExecutionService(
        code_access=code_access,
        identity_ledger=None,
        workdir="/tmp/test",
        max_concurrent=2,
        executors=executors
    )

    # Execute goal
    result = await service.execute_goal(
        goal_text="implement_feature",
        context={},
        timeout_ms=30000
    )

    # Verify result
    assert result is not None, "Result should not be None"
    assert result.goal_text == "implement_feature"

    # Check HTN planning
    assert result.plan_id is not None, "Plan should be created"
    assert "implement_feature_full" in result.methods_used, "Should use implement_feature_full method"
    assert result.total_tasks == 3, "Should decompose to 3 tasks"

    # Check execution
    assert result.success is True, f"Goal should succeed. Errors: {result.errors}"
    assert len(result.completed_tasks) == 3, "All 3 tasks should complete"
    assert len(result.failed_tasks) == 0, "No tasks should fail"

    # Check tasks
    task_names = [t.task_name for t in result.completed_tasks]
    assert "create_file" in task_names, "Should have create_file task"
    assert "run_tests" in task_names, "Should have run_tests task"

    # Check timing
    assert result.execution_time_ms > 0, "Execution time should be recorded"

    print(f"✅ Goal executed successfully in {result.execution_time_ms:.1f}ms")
    print(f"   Plan: {result.methods_used}")
    print(f"   Tasks: {len(result.completed_tasks)} completed, {len(result.failed_tasks)} failed")


@pytest.mark.asyncio
async def test_execute_goal_fix_bug():
    """Test executing 'fix_bug' goal."""
    code_access = MockCodeAccessService()
    executors = [MockCodeExecutor(), MockTestExecutor(), MockShellExecutor()]

    service = GoalExecutionService(
        code_access=code_access,
        workdir="/tmp/test",
        executors=executors
    )

    result = await service.execute_goal(
        goal_text="fix_bug",
        timeout_ms=30000
    )

    assert result.success is True, f"Goal should succeed. Errors: {result.errors}"
    assert "fix_bug_simple" in result.methods_used
    assert result.total_tasks == 2, "fix_bug decomposes to 2 tasks"
    assert len(result.completed_tasks) == 2

    # Verify modify_code was called
    task_names = [t.task_name for t in result.completed_tasks]
    assert "modify_code" in task_names

    print(f"✅ Bug fix goal executed in {result.execution_time_ms:.1f}ms")


@pytest.mark.asyncio
async def test_execute_goal_no_plan_found():
    """Test goal that cannot be planned (no matching method)."""
    code_access = MockCodeAccessService()
    service = GoalExecutionService(
        code_access=code_access,
        workdir="/tmp/test"
    )

    result = await service.execute_goal(
        goal_text="unknown_task_that_has_no_method",
        timeout_ms=30000
    )

    assert result.success is False, "Goal should fail if planning fails"
    assert len(result.errors) > 0, "Should have error messages"
    assert "planning failed" in result.errors[0].lower()
    assert result.plan_id is None, "No plan should be created"

    print(f"✅ Correctly handled unpla nnable goal")


@pytest.mark.asyncio
async def test_execute_goal_with_artifacts():
    """Test that task artifacts are collected."""
    code_access = MockCodeAccessService()
    executors = [MockCodeExecutor(), MockTestExecutor(), MockShellExecutor()]

    service = GoalExecutionService(
        code_access=code_access,
        workdir="/tmp/test",
        executors=executors
    )

    result = await service.execute_goal(
        goal_text="implement_feature",
        timeout_ms=30000
    )

    assert result.success is True
    assert len(result.artifacts) > 0, "Should collect task artifacts"

    # Each task should contribute artifacts
    assert len(result.artifacts) == result.total_tasks

    print(f"✅ Artifacts collected: {len(result.artifacts)} task artifacts")


@pytest.mark.asyncio
async def test_execute_goal_tracks_retries():
    """Test that retry counts are tracked (even if no retries occur)."""
    code_access = MockCodeAccessService()
    executors = [MockCodeExecutor(), MockTestExecutor(), MockShellExecutor()]

    service = GoalExecutionService(
        code_access=code_access,
        workdir="/tmp/test",
        executors=executors
    )

    result = await service.execute_goal(
        goal_text="add_tests",
        timeout_ms=30000
    )

    assert result.success is True
    assert result.retry_count == 0, "No retries should occur in successful execution"

    # Verify task results have retry info
    for task in result.completed_tasks:
        assert task.retry_count >= 0, "Retry count should be tracked"

    print(f"✅ Retry tracking working: {result.retry_count} total retries")


@pytest.mark.asyncio
async def test_goal_execution_service_initialization():
    """Test GoalExecutionService initializes correctly."""
    code_access = MockCodeAccessService()
    service = GoalExecutionService(
        code_access=code_access,
        workdir="/tmp/test",
        max_concurrent=5
    )

    assert service.planner is not None, "HTN planner should be initialized"
    assert len(service.planner.methods) >= 4, "Should have default coding methods"
    assert len(service.executors) == 3, "Should have 3 default executors"
    assert service.max_concurrent == 5

    print(f"✅ Service initialized with {len(service.planner.methods)} methods")
