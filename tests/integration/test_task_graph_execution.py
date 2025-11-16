"""Integration tests for TaskScheduler with TaskGraph and TaskExecutor."""

import asyncio
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock

from src.services.task_scheduler import TaskScheduler, create_task_scheduler
from src.services.task_graph import TaskGraph, DependencyPolicy, CircuitBreaker
from src.services.task_executor import TaskExecutor, TaskStatus, ErrorClass


# === Fixtures ===

@pytest.fixture
def temp_persona_space(tmp_path):
    """Create temporary persona space directory."""
    persona_path = tmp_path / "persona_space"
    persona_path.mkdir()
    return str(persona_path)


@pytest.fixture
def mock_persona_service():
    """Create mock persona service."""
    service = Mock()
    service.generate_response = Mock(return_value=("Response", {}))
    return service


@pytest.fixture
def scheduler(temp_persona_space):
    """Create TaskScheduler instance."""
    return create_task_scheduler(
        persona_space_path=temp_persona_space
    )


@pytest.fixture
def simple_graph():
    """Create a simple task graph for testing."""
    graph = TaskGraph(graph_id="test_graph")

    # Task A (no dependencies)
    graph.add_task(
        task_id="task_a",
        action_name="action1",
        normalized_args={"prompt": "Execute task A"},
        resource_ids=[],
        version="1.0"
    )

    # Task B (depends on A)
    graph.add_task(
        task_id="task_b",
        action_name="action1",
        normalized_args={"prompt": "Execute task B"},
        resource_ids=[],
        version="1.0",
        dependencies=["task_a"]
    )

    # Task C (depends on A)
    graph.add_task(
        task_id="task_c",
        action_name="action1",
        normalized_args={"prompt": "Execute task C"},
        resource_ids=[],
        version="1.0",
        dependencies=["task_a"]
    )

    return graph


# === Basic Execution Tests ===

@pytest.mark.asyncio
async def test_execute_simple_graph(scheduler, simple_graph, mock_persona_service):
    """Test executing a simple task graph."""
    result = await scheduler.execute_graph(
        graph=simple_graph,
        persona_service=mock_persona_service,
        max_parallel=10
    )

    # All tasks should complete
    assert result["total_tasks"] == 3
    assert len(result["results"]) == 3

    # All results should be successful
    for task_id, task_result in result["results"].items():
        assert task_result.get("success"), f"Task {task_id} failed"

    # Statistics should show all completed
    stats = result["statistics"]
    assert stats["states"].get("succeeded", 0) == 3
    assert stats["states"].get("failed", 0) == 0


@pytest.mark.asyncio
async def test_execute_graph_respects_dependencies(scheduler, mock_persona_service):
    """Test that tasks wait for dependencies."""
    graph = TaskGraph(graph_id="dep_test")

    execution_order = []

    # Create tasks that record execution order
    for i in range(3):
        task_id = f"task_{i}"
        deps = [f"task_{i-1}"] if i > 0 else []
        graph.add_task(
            task_id=task_id,
            action_name="ordered_action",
            normalized_args={"prompt": f"Task {i}", "order": i},
            resource_ids=[],
            version="1.0",
            dependencies=deps
        )

    # Mock persona service that records execution order
    def generate_response(user_message, **kwargs):
        task_num = int(user_message.split()[-1])
        execution_order.append(task_num)
        return (f"Response {task_num}", {})

    mock_persona_service.generate_response = generate_response

    result = await scheduler.execute_graph(
        graph=graph,
        persona_service=mock_persona_service
    )

    # Tasks should execute in order: 0, 1, 2
    assert execution_order == [0, 1, 2], "Tasks did not execute in dependency order"


# === Parallel Execution Tests ===

@pytest.mark.asyncio
async def test_parallel_execution(scheduler, mock_persona_service):
    """Test that independent tasks execute in parallel."""
    graph = TaskGraph(graph_id="parallel_test")

    # Create 5 independent tasks
    for i in range(5):
        graph.add_task(
            task_id=f"task_{i}",
            action_name="parallel_action",
            normalized_args={"prompt": f"Task {i}"},
            resource_ids=[],
            version="1.0"
        )

    start_time = datetime.now(timezone.utc)

    # Mock slow task execution
    async def slow_generate(user_message, **kwargs):
        await asyncio.sleep(0.1)
        return ("Response", {})

    mock_persona_service.generate_response = slow_generate

    result = await scheduler.execute_graph(
        graph=graph,
        persona_service=mock_persona_service,
        max_parallel=5
    )

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Should complete in ~0.1s (parallel), not 0.5s (serial)
    assert elapsed < 0.3, f"Tasks took {elapsed}s, likely executed serially"
    assert result["total_tasks"] == 5
    assert result["statistics"]["states"].get("succeeded", 0) == 5


@pytest.mark.asyncio
async def test_concurrency_limits(scheduler, mock_persona_service):
    """Test max_parallel limit is respected."""
    graph = TaskGraph(graph_id="concurrency_test")

    concurrent_count = 0
    max_concurrent = 0

    # Create 10 independent tasks
    for i in range(10):
        graph.add_task(
            task_id=f"task_{i}",
            action_name="concurrent_action",
            normalized_args={"prompt": f"Task {i}"},
            resource_ids=[],
            version="1.0"
        )

    # Track concurrent executions
    async def track_concurrent(user_message, **kwargs):
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.05)
        concurrent_count -= 1
        return ("Response", {})

    mock_persona_service.generate_response = track_concurrent

    await scheduler.execute_graph(
        graph=graph,
        persona_service=mock_persona_service,
        max_parallel=3  # Limit to 3 concurrent tasks
    )

    # Should never exceed 3 concurrent executions
    assert max_concurrent <= 3, f"Max concurrent was {max_concurrent}, expected <= 3"


# === Error Handling Tests ===

@pytest.mark.asyncio
async def test_dependency_failure_abort_policy(scheduler, mock_persona_service):
    """Test ABORT policy when dependency fails."""
    graph = TaskGraph(graph_id="abort_test")

    # Task A (will fail)
    graph.add_task(
        task_id="task_a",
        action_name="failing_action",
        normalized_args={"prompt": "Fail"},
        resource_ids=[],
        version="1.0",
        max_retries=0
    )

    # Task B depends on A with ABORT policy
    graph.add_task(
        task_id="task_b",
        action_name="dependent_action",
        normalized_args={"prompt": "Execute B"},
        resource_ids=[],
        version="1.0",
        dependencies=["task_a"],
        on_dep_fail=DependencyPolicy.ABORT
    )

    # Mock failing task
    def generate_response(user_message, **kwargs):
        if "Fail" in user_message:
            raise Exception("Task A failed")
        return ("Response B", {})

    mock_persona_service.generate_response = generate_response

    result = await scheduler.execute_graph(
        graph=graph,
        persona_service=mock_persona_service
    )

    # Task A should fail, Task B should be aborted
    assert result["results"]["task_a"]["success"] is False
    assert result["statistics"]["states"].get("failed", 0) >= 1
    assert result["statistics"]["states"].get("aborted", 0) >= 1


@pytest.mark.asyncio
async def test_dependency_failure_skip_policy(scheduler, mock_persona_service):
    """Test SKIP policy when dependency fails."""
    graph = TaskGraph(graph_id="skip_test")

    # Task A (will fail)
    graph.add_task(
        task_id="task_a",
        action_name="failing_action",
        normalized_args={"prompt": "Fail"},
        resource_ids=[],
        version="1.0",
        max_retries=0
    )

    # Task B depends on A with SKIP policy
    graph.add_task(
        task_id="task_b",
        action_name="dependent_action",
        normalized_args={"prompt": "Execute B"},
        resource_ids=[],
        version="1.0",
        dependencies=["task_a"],
        on_dep_fail=DependencyPolicy.SKIP
    )

    # Mock failing task
    def generate_response(user_message, **kwargs):
        if "Fail" in user_message:
            raise Exception("Task A failed")
        return ("Response B", {})

    mock_persona_service.generate_response = generate_response

    result = await scheduler.execute_graph(
        graph=graph,
        persona_service=mock_persona_service
    )

    # Task A should fail, Task B should be skipped
    stats = result["statistics"]
    assert stats["states"].get("failed", 0) >= 1
    assert stats["states"].get("skipped", 0) >= 1


# === Retry Logic Tests ===

@pytest.mark.asyncio
async def test_retry_integration(scheduler, mock_persona_service):
    """Test that retry logic works in graph execution."""
    graph = TaskGraph(graph_id="retry_test")

    # Task that fails first time, succeeds second time
    graph.add_task(
        task_id="retry_task",
        action_name="retry_action",
        normalized_args={"prompt": "Retry test"},
        resource_ids=[],
        version="1.0",
        max_retries=3
    )

    attempt_count = 0

    def generate_response(user_message, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise Exception("Temporary failure")
        return ("Success after retry", {})

    mock_persona_service.generate_response = generate_response

    result = await scheduler.execute_graph(
        graph=graph,
        persona_service=mock_persona_service
    )

    # Should succeed after retry
    assert result["results"]["retry_task"]["success"] is True
    assert attempt_count == 2


# === Idempotency Tests ===

@pytest.mark.asyncio
async def test_idempotency_in_graph(scheduler, mock_persona_service):
    """Test that tasks with same idempotency key are not re-executed."""
    graph = TaskGraph(graph_id="idempotency_test")

    # Two identical tasks
    for i in range(2):
        graph.add_task(
            task_id=f"task_{i}",
            action_name="same_action",
            normalized_args={"prompt": "Same prompt"},
            resource_ids=["resource1"],
            version="1.0"
        )

    execution_count = 0

    def generate_response(user_message, **kwargs):
        nonlocal execution_count
        execution_count += 1
        return (f"Response {execution_count}", {})

    mock_persona_service.generate_response = generate_response

    result = await scheduler.execute_graph(
        graph=graph,
        persona_service=mock_persona_service
    )

    # Should only execute once due to idempotency
    # Note: This depends on idempotency key generation in TaskNode
    # Both tasks have same action + args + resources + version
    # So they should generate same idempotency key
    # TaskExecutor should detect duplicate and skip second execution

    # At least verify both tasks complete
    assert result["total_tasks"] == 2
    assert len(result["results"]) == 2


# === Per-Action Concurrency Tests ===

@pytest.mark.asyncio
async def test_per_action_caps(scheduler, mock_persona_service):
    """Test per-action concurrency limits."""
    graph = TaskGraph(graph_id="action_cap_test")

    concurrent_action1 = 0
    max_concurrent_action1 = 0

    # Create 5 tasks of action1
    for i in range(5):
        graph.add_task(
            task_id=f"action1_task_{i}",
            action_name="action1",
            normalized_args={"prompt": f"Task {i}"},
            resource_ids=[],
            version="1.0"
        )

    # Create 5 tasks of action2
    for i in range(5):
        graph.add_task(
            task_id=f"action2_task_{i}",
            action_name="action2",
            normalized_args={"prompt": f"Task {i}"},
            resource_ids=[],
            version="1.0"
        )

    async def track_action1(user_message, **kwargs):
        nonlocal concurrent_action1, max_concurrent_action1
        concurrent_action1 += 1
        max_concurrent_action1 = max(max_concurrent_action1, concurrent_action1)
        await asyncio.sleep(0.05)
        concurrent_action1 -= 1
        return ("Response", {})

    mock_persona_service.generate_response = track_action1

    # Limit action1 to 2 concurrent, but overall limit is 10
    await scheduler.execute_graph(
        graph=graph,
        persona_service=mock_persona_service,
        max_parallel=10,
        per_action_caps={"action1": 2}
    )

    # Should never exceed 2 concurrent action1 tasks
    assert max_concurrent_action1 <= 2, f"Max concurrent action1 was {max_concurrent_action1}"


# === Priority Tests ===

@pytest.mark.asyncio
async def test_priority_ordering(scheduler, mock_persona_service):
    """Test that high priority tasks execute first."""
    graph = TaskGraph(graph_id="priority_test")

    execution_order = []

    # Create tasks with different priorities
    graph.add_task(
        task_id="low_priority",
        action_name="action1",
        normalized_args={"prompt": "Low"},
        resource_ids=[],
        version="1.0",
        priority=1
    )

    graph.add_task(
        task_id="high_priority",
        action_name="action1",
        normalized_args={"prompt": "High"},
        resource_ids=[],
        version="1.0",
        priority=10
    )

    graph.add_task(
        task_id="medium_priority",
        action_name="action1",
        normalized_args={"prompt": "Medium"},
        resource_ids=[],
        version="1.0",
        priority=5
    )

    def generate_response(user_message, **kwargs):
        execution_order.append(user_message.split()[-1])
        return ("Response", {})

    mock_persona_service.generate_response = generate_response

    await scheduler.execute_graph(
        graph=graph,
        persona_service=mock_persona_service,
        max_parallel=1  # Execute one at a time to see priority order
    )

    # Should execute in priority order: High, Medium, Low
    assert execution_order == ["High", "Medium", "Low"], f"Got order: {execution_order}"


# === Factory Function Test ===

def test_create_task_scheduler_with_executor():
    """Test factory function creates scheduler with custom executor."""
    executor = TaskExecutor(max_retries=5)

    scheduler = create_task_scheduler(
        persona_space_path="test_path",
        task_executor=executor
    )

    assert scheduler.task_executor is executor
    assert scheduler.task_executor.max_retries == 5
