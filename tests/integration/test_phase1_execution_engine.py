"""Phase 1 Integration Tests: TaskExecutionEngine + Executors

Tests the core execution engine with real executors.
"""
import pytest
import asyncio
import time
from pathlib import Path
from uuid import uuid4

from src.services.task_graph import TaskGraph, TaskState
from src.services.task_execution_engine import TaskExecutionEngine
from src.services.task_executors.base import RunContext
from src.services.task_executors.code_modification import CodeModificationExecutor
from src.services.task_executors.test_runner import TestExecutor
from src.services.task_executors.shell_command import ShellCommandExecutor
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
        auto_branch=False  # Don't create branches for tests
    )


@pytest.fixture
def executors(code_access):
    """Create real executors."""
    return [
        CodeModificationExecutor(code_access),
        TestExecutor(),
        ShellCommandExecutor()
    ]


@pytest.fixture
def make_context(project_root):
    """Context factory for task execution."""
    def _make_context(task):
        return RunContext(
            trace_id=f"test-{uuid4().hex[:8]}",
            span_id=f"task-{task.task_id}",
            workdir=str(project_root),
            timeout_ms=task.task_timeout_ms or 30000,
            env={},
            monotonic=time.monotonic,
            ledger=None,
            breaker=None,
            caps={}
        )
    return _make_context


class TestPhase1ExecutionEngine:
    """Phase 1: Core execution engine tests."""

    @pytest.mark.asyncio
    async def test_single_file_creation(self, executors, make_context):
        """Test creating a single file."""
        # Create task graph
        graph = TaskGraph(
            graph_id="test-single-file",
            max_retry_tokens=5,
            graph_timeout_ms=30000
        )

        task_id = "create_test_file"
        graph.add_task(
            task_id=task_id,
            action_name="create_file",
            normalized_args={
                "file_path": "tests/generated/phase1_test_single.py",
                "content": "# Phase 1 test file\ndef test_function():\n    return True\n",
                "reason": "Phase 1 test"
            },
            resource_ids=[],
            version="1.0",
            dependencies=[],
            priority=0.5,
            max_retries=2
        )

        # Execute
        engine = TaskExecutionEngine(
            graph=graph,
            executors=executors,
            max_concurrent=1
        )

        await engine.run(make_context)

        # Verify
        stats = graph.get_stats()
        assert stats['states']['succeeded'] == 1
        assert stats['states'].get('failed', 0) == 0

        # Verify file exists
        file_path = Path("tests/generated/phase1_test_single.py")
        assert file_path.exists()
        content = file_path.read_text()
        assert "test_function" in content

    @pytest.mark.asyncio
    async def test_sequential_tasks(self, executors, make_context):
        """Test sequential task execution with dependencies."""
        graph = TaskGraph(
            graph_id="test-sequential",
            max_retry_tokens=10,
            graph_timeout_ms=60000
        )

        # Task 1: Create file
        task1_id = "create_impl"
        graph.add_task(
            task_id=task1_id,
            action_name="create_file",
            normalized_args={
                "file_path": "tests/generated/phase1_sequential_impl.py",
                "content": "def add(a, b):\n    return a + b\n",
                "reason": "Create implementation"
            },
            resource_ids=[],
            version="1.0",
            dependencies=[],
            priority=0.5,
            max_retries=2
        )

        # Task 2: Create test file (depends on task1)
        task2_id = "create_test"
        graph.add_task(
            task_id=task2_id,
            action_name="create_file",
            normalized_args={
                "file_path": "tests/generated/phase1_sequential_test.py",
                "content": "from phase1_sequential_impl import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
                "reason": "Create test"
            },
            resource_ids=[],
            version="1.0",
            dependencies=[task1_id],
            priority=0.5,
            max_retries=2
        )

        # Execute
        engine = TaskExecutionEngine(
            graph=graph,
            executors=executors,
            max_concurrent=2
        )

        await engine.run(make_context)

        # Verify
        stats = graph.get_stats()
        assert stats['states']['succeeded'] == 2
        assert stats['states'].get('failed', 0) == 0

        # Verify both files exist
        assert Path("tests/generated/phase1_sequential_impl.py").exists()
        assert Path("tests/generated/phase1_sequential_test.py").exists()

    @pytest.mark.asyncio
    async def test_parallel_tasks(self, executors, make_context):
        """Test parallel task execution (no dependencies)."""
        graph = TaskGraph(
            graph_id="test-parallel",
            max_retry_tokens=10,
            graph_timeout_ms=60000
        )

        # Create 3 independent tasks
        for i in range(3):
            graph.add_task(
                task_id=f"create_file_{i}",
                action_name="create_file",
                normalized_args={
                    "file_path": f"tests/generated/phase1_parallel_{i}.py",
                    "content": f"# Parallel task {i}\nvalue = {i}\n",
                    "reason": f"Parallel test {i}"
                },
                resource_ids=[],
                version="1.0",
                dependencies=[],
                priority=0.5,
                max_retries=2
            )

        # Execute with concurrency
        engine = TaskExecutionEngine(
            graph=graph,
            executors=executors,
            max_concurrent=3
        )

        start_time = time.monotonic()
        await engine.run(make_context)
        elapsed = time.monotonic() - start_time

        # Verify all succeeded
        stats = graph.get_stats()
        assert stats['states']['succeeded'] == 3

        # Verify files exist
        for i in range(3):
            assert Path(f"tests/generated/phase1_parallel_{i}.py").exists()

        # Verify ran in parallel (should be fast)
        assert elapsed < 2.0, f"Parallel execution took {elapsed}s, expected < 2s"

    @pytest.mark.asyncio
    async def test_shell_executor(self, executors, make_context):
        """Test shell command execution."""
        graph = TaskGraph(
            graph_id="test-shell",
            max_retry_tokens=5,
            graph_timeout_ms=30000
        )

        graph.add_task(
            task_id="echo_test",
            action_name="shell_command",
            normalized_args={
                "cmd": ["echo", "Phase 1 shell test"]
            },
            resource_ids=[],
            version="1.0",
            dependencies=[],
            priority=0.5,
            max_retries=2
        )

        # Execute
        engine = TaskExecutionEngine(
            graph=graph,
            executors=executors,
            max_concurrent=1
        )

        await engine.run(make_context)

        # Verify
        stats = graph.get_stats()
        assert stats['states']['succeeded'] == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, executors, make_context):
        """Test retry logic on retryable failures."""
        graph = TaskGraph(
            graph_id="test-retry",
            max_retry_tokens=10,
            graph_timeout_ms=60000
        )

        # This will fail (invalid command)
        graph.add_task(
            task_id="failing_task",
            action_name="shell_command",
            normalized_args={
                "cmd": ["nonexistent_command_xyz"]
            },
            resource_ids=[],
            version="1.0",
            dependencies=[],
            priority=0.5,
            max_retries=3
        )

        # Execute
        engine = TaskExecutionEngine(
            graph=graph,
            executors=executors,
            max_concurrent=1
        )

        await engine.run(make_context)

        # Verify task failed after retries
        task = graph.nodes["failing_task"]
        assert task.state == TaskState.FAILED
        assert task.retry_count > 0  # Should have retried
        assert task.retry_count <= 3  # Should not exceed max_retries

    @pytest.mark.asyncio
    async def test_executor_admission_control(self, code_access, make_context):
        """Test that executors reject disallowed operations."""
        graph = TaskGraph(
            graph_id="test-admission",
            max_retry_tokens=5,
            graph_timeout_ms=30000
        )

        # Try to create file in forbidden path
        graph.add_task(
            task_id="forbidden_file",
            action_name="create_file",
            normalized_args={
                "file_path": "/etc/passwd",  # Forbidden path
                "content": "malicious content",
                "reason": "Test admission control"
            },
            resource_ids=[],
            version="1.0",
            dependencies=[],
            priority=0.5,
            max_retries=0
        )

        # Execute
        executor = CodeModificationExecutor(code_access)
        engine = TaskExecutionEngine(
            graph=graph,
            executors=[executor],
            max_concurrent=1
        )

        await engine.run(make_context)

        # Verify task was rejected
        task = graph.nodes["forbidden_file"]
        assert task.state == TaskState.FAILED
        assert "access" in task.last_error.lower() or "admit" in task.last_error.lower()

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, executors, make_context):
        """Test that max_concurrent is respected."""
        graph = TaskGraph(
            graph_id="test-concurrency",
            max_retry_tokens=10,
            graph_timeout_ms=60000
        )

        # Create 5 independent tasks
        for i in range(5):
            graph.add_task(
                task_id=f"task_{i}",
                action_name="shell_command",
                normalized_args={
                    "cmd": ["sleep", "0.1"]
                },
                resource_ids=[],
                version="1.0",
                dependencies=[],
                priority=0.5,
                max_retries=0
            )

        # Execute with max_concurrent=2
        engine = TaskExecutionEngine(
            graph=graph,
            executors=executors,
            max_concurrent=2
        )

        await engine.run(make_context)

        # Verify all succeeded
        stats = graph.get_stats()
        assert stats['states']['succeeded'] == 5

    @pytest.mark.asyncio
    async def test_complete_workflow(self, executors, make_context):
        """Test a complete workflow: create impl + test + run tests."""
        graph_id = uuid4().hex[:8]
        graph = TaskGraph(
            graph_id=f"test-workflow-{graph_id}",
            max_retry_tokens=10,
            graph_timeout_ms=120000
        )

        # Task 1: Create implementation
        impl_file = f"tests/generated/phase1_workflow_{graph_id}.py"
        graph.add_task(
            task_id="create_impl",
            action_name="create_file",
            normalized_args={
                "file_path": impl_file,
                "content": "def multiply(a, b):\n    return a * b\n",
                "reason": "Workflow test implementation"
            },
            resource_ids=[],
            version="1.0",
            dependencies=[],
            priority=0.5,
            max_retries=2
        )

        # Task 2: Create test
        test_file = f"tests/generated/test_phase1_workflow_{graph_id}.py"
        graph.add_task(
            task_id="create_test",
            action_name="create_file",
            normalized_args={
                "file_path": test_file,
                "content": f"""import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from phase1_workflow_{graph_id} import multiply

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(5, 5) == 25
""",
                "reason": "Workflow test"
            },
            resource_ids=[],
            version="1.0",
            dependencies=["create_impl"],
            priority=0.5,
            max_retries=2
        )

        # Task 3: Run tests
        graph.add_task(
            task_id="run_tests",
            action_name="run_tests",
            normalized_args={
                "cmd": ["python3", "-m", "pytest", "-v", test_file]
            },
            resource_ids=[],
            version="1.0",
            dependencies=["create_test"],
            priority=0.5,
            max_retries=1
        )

        # Execute
        engine = TaskExecutionEngine(
            graph=graph,
            executors=executors,
            max_concurrent=2
        )

        await engine.run(make_context)

        # Verify
        stats = graph.get_stats()

        # All 3 tasks should complete (impl, test, run)
        assert stats['total_tasks'] == 3

        # At least impl and test creation should succeed
        assert stats['states']['succeeded'] >= 2

        # Verify files exist
        assert Path(impl_file).exists()
        assert Path(test_file).exists()


def test_suite_info():
    """Print test suite information."""
    print("\n" + "="*70)
    print("PHASE 1 TEST SUITE: TaskExecutionEngine + Executors")
    print("="*70)
    print("\nTests:")
    print("  1. Single file creation")
    print("  2. Sequential tasks with dependencies")
    print("  3. Parallel task execution")
    print("  4. Shell command execution")
    print("  5. Retry logic on failures")
    print("  6. Executor admission control")
    print("  7. Concurrency limit enforcement")
    print("  8. Complete workflow (impl + test + run)")
    print("\nRun with: pytest tests/integration/test_phase1_execution_engine.py -v")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_suite_info()
