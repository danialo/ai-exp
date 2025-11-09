"""Phase 2 Integration Tests: GoalExecutionService (HTN Planning)

Tests HTN planning, TaskGraph creation, and end-to-end goal execution.
"""
import pytest
import asyncio
from pathlib import Path
from uuid import uuid4

from src.services.goal_execution_service import GoalExecutionService, GoalExecutionResult
from src.services.htn_planner import HTNPlanner, Method
from src.services.code_access import create_code_access_service


@pytest.fixture
def project_root():
    """Project root for testing."""
    return Path("/home/d/git/ai-exp")


@pytest.fixture
def code_access(project_root):
    """CodeAccessService for testing."""
    return create_code_access_service(
        project_root=project_root,
        max_file_size_kb=100,
        auto_branch=False
    )


@pytest.fixture
def goal_service(code_access, project_root):
    """GoalExecutionService for testing."""
    return GoalExecutionService(
        code_access=code_access,
        identity_ledger=None,
        workdir=str(project_root),
        max_concurrent=2
    )


class TestPhase2HTNPlanning:
    """Phase 2A: HTN planning tests."""

    def test_htn_planner_initialization(self):
        """Test HTN planner initializes with default methods."""
        planner = HTNPlanner(
            belief_kernel=None,
            method_library=None,  # Use defaults
            primitive_tasks=None   # Use defaults
        )

        # Should have default methods loaded
        assert len(planner.methods) > 0
        assert len(planner.primitive_tasks) > 0

        # Check for expected methods
        method_names = {m.name for m in planner.methods}
        assert "implement_feature_full" in method_names
        assert "fix_bug_simple" in method_names

    def test_htn_plan_implement_feature(self):
        """Test HTN planning for implement_feature goal."""
        planner = HTNPlanner(
            belief_kernel=None,
            method_library=None,
            primitive_tasks=None
        )

        plan = planner.plan(
            goal_id="test-goal-1",
            goal_text="implement_feature",
            world_state={},
            constraints=[]
        )

        # Should produce a valid plan
        assert plan is not None
        assert plan.plan_id is not None
        assert len(plan.tasks) > 0

        # Should use implement_feature_full method
        assert "implement_feature_full" in plan.methods_used

        # Should have primitive tasks
        task_names = {t.task_name for t in plan.tasks}
        assert "create_file" in task_names

    def test_htn_plan_fix_bug(self):
        """Test HTN planning for fix_bug goal."""
        planner = HTNPlanner(
            belief_kernel=None,
            method_library=None,
            primitive_tasks=None
        )

        plan = planner.plan(
            goal_id="test-goal-2",
            goal_text="fix_bug",
            world_state={},
            constraints=[]
        )

        assert plan is not None
        assert "fix_bug_simple" in plan.methods_used

        # Should have modify_code and run_tests
        task_names = {t.task_name for t in plan.tasks}
        assert "modify_code" in task_names
        assert "run_tests" in task_names


class TestPhase2GoalExecution:
    """Phase 2B: Goal execution tests."""

    @pytest.mark.asyncio
    async def test_execute_implement_feature(self, goal_service):
        """Test executing implement_feature goal."""
        result = await goal_service.execute_goal(
            goal_text="implement_feature",
            context={},
            timeout_ms=60000
        )

        # Verify result structure
        assert isinstance(result, GoalExecutionResult)
        assert result.goal_id is not None
        assert result.goal_text == "implement_feature"

        # Should have plan info
        assert result.plan_id is not None
        assert len(result.methods_used) > 0
        assert result.total_tasks >= 3

        # Should have completed some tasks (file creation)
        assert len(result.completed_tasks) >= 2

        # Verify files were created
        generated_files = list(Path("tests/generated").glob("feature_*.py"))
        assert len(generated_files) > 0

    @pytest.mark.asyncio
    async def test_execute_fix_bug(self, goal_service):
        """Test executing fix_bug goal."""
        result = await goal_service.execute_goal(
            goal_text="fix_bug",
            context={},
            timeout_ms=60000
        )

        assert result.goal_text == "fix_bug"
        assert result.total_tasks >= 2

        # Should use fix_bug_simple method
        assert "fix_bug_simple" in result.methods_used

        # Should have modify_code task
        task_names = {t.task_name for t in result.completed_tasks + result.failed_tasks}
        assert "modify_code" in task_names

    @pytest.mark.asyncio
    async def test_execute_refactor_code(self, goal_service):
        """Test executing refactor_code goal."""
        result = await goal_service.execute_goal(
            goal_text="refactor_code",
            context={},
            timeout_ms=60000
        )

        assert result.goal_text == "refactor_code"
        assert "refactor_code_safe" in result.methods_used

    @pytest.mark.asyncio
    async def test_execute_add_tests(self, goal_service):
        """Test executing add_tests goal."""
        result = await goal_service.execute_goal(
            goal_text="add_tests",
            context={},
            timeout_ms=60000
        )

        assert result.goal_text == "add_tests"
        assert "add_tests_only" in result.methods_used

        # Should create test file
        assert len(result.completed_tasks) >= 1


class TestPhase2ParameterEnrichment:
    """Phase 2C: Parameter enrichment tests."""

    @pytest.mark.asyncio
    async def test_file_paths_in_allowed_directory(self, goal_service):
        """Test that generated file paths are in allowed directory."""
        result = await goal_service.execute_goal(
            goal_text="implement_feature",
            context={},
            timeout_ms=60000
        )

        # Check all file artifacts
        for task in result.completed_tasks:
            if "file_path" in task.artifacts:
                file_path = task.artifacts["file_path"]
                # Should be in tests/generated/
                assert file_path.startswith("tests/generated/")

    @pytest.mark.asyncio
    async def test_unique_file_naming(self, goal_service):
        """Test that multiple executions create unique files."""
        # Execute twice
        result1 = await goal_service.execute_goal(
            goal_text="implement_feature",
            context={},
            timeout_ms=60000
        )

        result2 = await goal_service.execute_goal(
            goal_text="implement_feature",
            context={},
            timeout_ms=60000
        )

        # Extract file paths
        files1 = {t.artifacts.get("file_path") for t in result1.completed_tasks if "file_path" in t.artifacts}
        files2 = {t.artifacts.get("file_path") for t in result2.completed_tasks if "file_path" in t.artifacts}

        # Should be different (based on unique goal_id)
        assert files1 != files2

    @pytest.mark.asyncio
    async def test_pytest_command_format(self, goal_service):
        """Test that pytest commands use correct format."""
        result = await goal_service.execute_goal(
            goal_text="implement_feature",
            context={},
            timeout_ms=60000
        )

        # Find run_tests task
        test_tasks = [t for t in result.completed_tasks + result.failed_tasks if t.task_name == "run_tests"]

        if test_tasks:
            task = test_tasks[0]
            cmd = task.artifacts.get("cmd", [])

            # Should use "python3 -m pytest" format
            assert cmd[0] == "python3"
            assert cmd[1] == "-m"
            assert cmd[2] == "pytest"


class TestPhase2ErrorHandling:
    """Phase 2D: Error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_goal_text(self, goal_service):
        """Test execution with invalid goal text."""
        result = await goal_service.execute_goal(
            goal_text="nonexistent_goal_type",
            context={},
            timeout_ms=60000
        )

        # Should fail to plan
        assert result.success is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self, goal_service):
        """Test that timeout is respected."""
        import time

        start = time.monotonic()

        result = await goal_service.execute_goal(
            goal_text="implement_feature",
            context={},
            timeout_ms=5000  # Very short timeout
        )

        elapsed = time.monotonic() - start

        # Should complete within reasonable time
        # (tasks are fast, so this shouldn't timeout, but engine should respect limit)
        assert elapsed < 10.0

    @pytest.mark.asyncio
    async def test_execution_metrics(self, goal_service):
        """Test that execution metrics are captured."""
        result = await goal_service.execute_goal(
            goal_text="implement_feature",
            context={},
            timeout_ms=60000
        )

        # Should have timing info
        assert result.execution_time_ms > 0
        assert result.started_at is not None

        # Should have HTN planning cost
        assert result.planning_cost >= 0

        # Should have retry count (even if 0)
        assert result.retry_count >= 0

    @pytest.mark.asyncio
    async def test_artifact_collection(self, goal_service):
        """Test that artifacts are collected from tasks."""
        result = await goal_service.execute_goal(
            goal_text="implement_feature",
            context={},
            timeout_ms=60000
        )

        # Should have artifacts dict
        assert isinstance(result.artifacts, dict)

        # Should have task IDs as keys
        assert len(result.artifacts) > 0

        # Each artifact should have task parameters
        for task_id, artifacts in result.artifacts.items():
            assert isinstance(artifacts, dict)


class TestPhase2Integration:
    """Phase 2E: Integration tests."""

    @pytest.mark.asyncio
    async def test_plan_to_taskgraph_conversion(self, goal_service):
        """Test that HTN plan converts correctly to TaskGraph."""
        # Execute goal
        result = await goal_service.execute_goal(
            goal_text="implement_feature",
            context={},
            timeout_ms=60000
        )

        # Verify plan was created
        assert result.plan_id is not None

        # Verify tasks match plan
        assert result.total_tasks == len(result.completed_tasks) + len(result.failed_tasks)

        # Verify methods were used
        assert len(result.methods_used) > 0

    @pytest.mark.asyncio
    async def test_sequential_execution_order(self, goal_service):
        """Test that sequential tasks execute in order."""
        result = await goal_service.execute_goal(
            goal_text="implement_feature",
            context={},
            timeout_ms=60000
        )

        # Get task completion times
        tasks_by_time = sorted(
            result.completed_tasks,
            key=lambda t: t.completed_at if t.completed_at else t.started_at
        )

        # First tasks should be file creation
        # Last task should be run_tests
        if len(tasks_by_time) >= 3:
            # First two should be create_file
            assert tasks_by_time[0].task_name == "create_file"
            assert tasks_by_time[1].task_name == "create_file"

    @pytest.mark.asyncio
    async def test_concurrent_goal_execution(self, goal_service):
        """Test executing multiple goals concurrently."""
        # Execute 3 goals in parallel
        tasks = [
            goal_service.execute_goal("implement_feature", {}, 60000),
            goal_service.execute_goal("fix_bug", {}, 60000),
            goal_service.execute_goal("add_tests", {}, 60000)
        ]

        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 3

        # Each should have unique goal_id
        goal_ids = {r.goal_id for r in results}
        assert len(goal_ids) == 3


def test_suite_info():
    """Print test suite information."""
    print("\n" + "="*70)
    print("PHASE 2 TEST SUITE: GoalExecutionService (HTN Planning)")
    print("="*70)
    print("\nTest Categories:")
    print("  A. HTN Planning")
    print("     - Planner initialization")
    print("     - Plan generation for each goal type")
    print("  B. Goal Execution")
    print("     - Execute implement_feature")
    print("     - Execute fix_bug")
    print("     - Execute refactor_code")
    print("     - Execute add_tests")
    print("  C. Parameter Enrichment")
    print("     - File paths in allowed directories")
    print("     - Unique file naming")
    print("     - Correct pytest command format")
    print("  D. Error Handling")
    print("     - Invalid goal text")
    print("     - Timeout handling")
    print("     - Execution metrics")
    print("     - Artifact collection")
    print("  E. Integration")
    print("     - Plan → TaskGraph conversion")
    print("     - Sequential execution order")
    print("     - Concurrent goal execution")
    print("\nRun with: pytest tests/integration/test_phase2_goal_execution_service.py -v")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_suite_info()
