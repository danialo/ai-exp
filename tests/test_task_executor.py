"""Tests for TaskExecutor."""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, MagicMock

from src.services.task_executor import (
    TaskExecutor,
    TaskStatus,
    ErrorClass,
    TaskExecutionResult,
    create_task_executor
)
from src.services.task_graph import TaskNode, CircuitBreaker


# === Fixtures ===

@pytest.fixture
def task_node():
    """Create a basic task node."""
    return TaskNode(
        task_id="test_task",
        action_name="test_action",
        normalized_args={"arg1": "value1"},
        resource_ids=["resource1"],
        version="1.0",
        max_retries=3
    )


@pytest.fixture
def executor():
    """Create a basic task executor."""
    return TaskExecutor(
        max_retries=3,
        base_delay_ms=100,  # Fast for testing
        max_delay_ms=1000,
        jitter_ms=50
    )


@pytest.fixture
def mock_abort_monitor():
    """Create mock abort monitor."""
    monitor = Mock()
    monitor.should_abort.return_value = False
    monitor.get_abort_reason.return_value = None
    return monitor


@pytest.fixture
def mock_decision_registry():
    """Create mock decision registry."""
    registry = Mock()
    registry.record_decision.return_value = "decision_123"
    registry.get_all_parameters.return_value = {}
    registry.record_outcome.return_value = None
    return registry


@pytest.fixture
def mock_raw_store():
    """Create mock raw store."""
    store = Mock()
    store.create_task_execution.return_value = None
    store.list_task_executions.return_value = []
    return store


# === Successful Execution Tests ===

@pytest.mark.asyncio
async def test_successful_execution(executor, task_node):
    """Test successful task execution."""
    async def task_callable(args):
        return {"success": True, "result": "done"}

    result = await executor.execute(task_node, task_callable)

    assert result.success
    assert result.status == TaskStatus.SUCCESS
    assert result.error is None
    assert result.execution_time_ms > 0


@pytest.mark.asyncio
async def test_successful_execution_with_decision_recording(task_node, mock_decision_registry):
    """Test execution records decision."""
    executor = TaskExecutor(decision_registry=mock_decision_registry)

    async def task_callable(args):
        return {"success": True}

    await executor.execute(task_node, task_callable)

    # Should record decision
    mock_decision_registry.record_decision.assert_called_once()
    assert mock_decision_registry.record_decision.call_args[1]["decision_id"] == "task_execution"

    # Should record outcome
    mock_decision_registry.record_outcome.assert_called_once()


# === Retry Logic Tests ===

@pytest.mark.asyncio
async def test_retry_on_transient_failure(executor, task_node):
    """Test retry on transient failure."""
    attempt_count = 0

    async def failing_task(args):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception("Connection error - temporary")
        return {"success": True}

    result = await executor.execute(task_node, failing_task)

    assert result.success
    assert attempt_count == 3  # Failed twice, succeeded on third


@pytest.mark.asyncio
async def test_retry_exhaustion(executor, task_node):
    """Test all retries exhausted."""
    async def always_fails(args):
        raise Exception("Connection error")

    result = await executor.execute(task_node, always_fails)

    assert not result.success
    assert result.status == TaskStatus.FAILED
    assert result.error_class == ErrorClass.TRANSIENT
    assert not result.retryable  # Retries exhausted


@pytest.mark.asyncio
async def test_exponential_backoff_delay(executor, task_node):
    """Test exponential backoff calculates correctly."""
    # Test delay calculation
    delay_0 = executor._calculate_backoff_delay(0)
    delay_1 = executor._calculate_backoff_delay(1)
    delay_2 = executor._calculate_backoff_delay(2)

    # Should follow exponential pattern: base_delay * 2^attempt
    # With jitter, delays should be in expected ranges
    assert 100 <= delay_0 <= 150  # 100 * 2^0 + jitter
    assert 200 <= delay_1 <= 250  # 100 * 2^1 + jitter
    assert 400 <= delay_2 <= 450  # 100 * 2^2 + jitter


@pytest.mark.asyncio
async def test_max_delay_cap(executor):
    """Test delay caps at maximum."""
    # Very high attempt should cap at max_delay_ms
    delay = executor._calculate_backoff_delay(10)
    assert delay <= executor.max_delay_ms + executor.jitter_ms


@pytest.mark.asyncio
async def test_no_retry_on_permanent_error(executor, task_node):
    """Test permanent errors don't retry."""
    attempt_count = 0

    async def task_with_permanent_error(args):
        nonlocal attempt_count
        attempt_count += 1
        raise Exception("Invalid argument - not found")

    result = await executor.execute(task_node, task_with_permanent_error)

    assert not result.success
    assert attempt_count == 1  # No retries
    assert result.error_class == ErrorClass.PERMANENT
    assert not result.retryable


# === Idempotency Tests ===

@pytest.mark.asyncio
async def test_idempotency_prevents_duplicate(executor, task_node):
    """Test idempotency prevents duplicate execution."""
    call_count = 0

    async def task_callable(args):
        nonlocal call_count
        call_count += 1
        return {"success": True}

    # First execution
    result1 = await executor.execute(task_node, task_callable, idempotency_key="key123")
    assert result1.success
    assert call_count == 1

    # Second execution with same key - should skip
    result2 = await executor.execute(task_node, task_callable, idempotency_key="key123")
    assert result2.status == TaskStatus.DUPLICATE
    assert call_count == 1  # Not called again


@pytest.mark.asyncio
async def test_idempotency_with_raw_store(task_node, mock_raw_store):
    """Test idempotency checks raw store."""
    # Mock raw store has execution
    mock_raw_store.list_task_executions.return_value = [{"id": "exec1"}]

    executor = TaskExecutor(raw_store=mock_raw_store)

    call_count = 0

    async def task_callable(args):
        nonlocal call_count
        call_count += 1
        return {"success": True}

    result = await executor.execute(task_node, task_callable, idempotency_key="key456")

    assert result.status == TaskStatus.DUPLICATE
    assert call_count == 0  # Never executed
    mock_raw_store.list_task_executions.assert_called_once()


# === Safety Envelope Tests ===

@pytest.mark.asyncio
async def test_safety_envelope_blocks_execution(task_node, mock_abort_monitor):
    """Test safety envelope blocks task execution."""
    mock_abort_monitor.should_abort.return_value = True
    mock_abort_monitor.get_abort_reason.return_value = "System overload"

    executor = TaskExecutor(abort_monitor=mock_abort_monitor)

    call_count = 0

    async def task_callable(args):
        nonlocal call_count
        call_count += 1
        return {"success": True}

    result = await executor.execute(task_node, task_callable)

    assert not result.success
    assert result.status == TaskStatus.ABORTED
    assert result.aborted
    assert "System overload" in result.error
    assert call_count == 0  # Never executed


@pytest.mark.asyncio
async def test_safety_envelope_checked_before_each_retry(task_node, mock_abort_monitor):
    """Test safety envelope checked before each retry attempt."""
    attempt_count = 0

    # Abort on second attempt
    def should_abort_side_effect():
        return attempt_count > 0

    mock_abort_monitor.should_abort.side_effect = should_abort_side_effect
    mock_abort_monitor.get_abort_reason.return_value = "Abort on retry"

    executor = TaskExecutor(abort_monitor=mock_abort_monitor)

    async def failing_task(args):
        nonlocal attempt_count
        attempt_count += 1
        raise Exception("Connection error")

    result = await executor.execute(task_node, failing_task)

    assert result.status == TaskStatus.ABORTED
    assert attempt_count == 1  # Aborted before second attempt


# === Circuit Breaker Tests ===

@pytest.mark.asyncio
async def test_circuit_breaker_blocks_when_open(executor, task_node):
    """Test circuit breaker blocks execution when open."""
    breaker = CircuitBreaker(action_name="test_action", failure_threshold=1)
    breaker.record_failure("error")  # Open the breaker

    assert breaker.is_open()

    call_count = 0

    async def task_callable(args):
        nonlocal call_count
        call_count += 1
        return {"success": True}

    result = await executor.execute(task_node, task_callable, circuit_breaker=breaker)

    assert not result.success
    assert result.status == TaskStatus.ABORTED
    assert result.aborted
    assert "Circuit breaker open" in result.error
    assert call_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_records_success(executor, task_node):
    """Test circuit breaker records successful execution."""
    breaker = CircuitBreaker(action_name="test_action")

    async def task_callable(args):
        return {"success": True}

    await executor.execute(task_node, task_callable, circuit_breaker=breaker)

    # Breaker should remain closed and have no failures
    assert not breaker.is_open()
    assert len(breaker.failures) == 0


@pytest.mark.asyncio
async def test_circuit_breaker_records_failure(executor, task_node):
    """Test circuit breaker records failures."""
    breaker = CircuitBreaker(action_name="test_action", failure_threshold=5)

    async def task_callable(args):
        raise Exception("Connection error")

    await executor.execute(task_node, task_callable, circuit_breaker=breaker)

    # Breaker should have recorded failures
    assert len(breaker.failures) > 0


# === Timeout Tests ===

@pytest.mark.asyncio
async def test_task_timeout(executor, task_node):
    """Test task execution times out."""
    async def slow_task(args):
        await asyncio.sleep(10)  # Very slow
        return {"success": True}

    result = await executor.execute(task_node, slow_task, timeout_ms=100)

    assert not result.success
    assert result.status == TaskStatus.FAILED
    assert "timeout" in result.error.lower()


@pytest.mark.asyncio
async def test_timeout_retries(executor, task_node):
    """Test timeout error is retried."""
    attempt_count = 0

    async def sometimes_slow_task(args):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            await asyncio.sleep(1)  # Timeout
        return {"success": True}

    result = await executor.execute(task_node, sometimes_slow_task, timeout_ms=100)

    assert result.success
    assert attempt_count == 3


# === Error Classification Tests ===

def test_classify_safety_error(executor):
    """Test safety errors classified correctly."""
    assert executor._classify_error("Aborted by safety envelope") == ErrorClass.SAFETY
    assert executor._classify_error(Exception("Safety check failed")) == ErrorClass.SAFETY


def test_classify_timeout_error(executor):
    """Test timeout errors classified correctly."""
    assert executor._classify_error("Request timed out") == ErrorClass.TIMEOUT
    assert executor._classify_error(Exception("timeout exceeded")) == ErrorClass.TIMEOUT


def test_classify_transient_error(executor):
    """Test transient errors classified correctly."""
    assert executor._classify_error("Connection refused") == ErrorClass.TRANSIENT
    assert executor._classify_error(Exception("network unavailable")) == ErrorClass.TRANSIENT
    assert executor._classify_error("Temporary failure") == ErrorClass.TRANSIENT


def test_classify_permanent_error(executor):
    """Test permanent errors classified correctly."""
    assert executor._classify_error("Resource not found") == ErrorClass.PERMANENT
    assert executor._classify_error(Exception("Invalid input")) == ErrorClass.PERMANENT
    assert executor._classify_error("Unauthorized access") == ErrorClass.PERMANENT


def test_default_classification_is_transient(executor):
    """Test unknown errors default to transient (retry)."""
    assert executor._classify_error("Unknown weird error") == ErrorClass.TRANSIENT


def test_is_retryable(executor):
    """Test retryable error detection."""
    # Transient and timeout are retryable
    assert executor._is_retryable(Exception("Connection error"))
    assert executor._is_retryable(Exception("Timeout"))

    # Safety and permanent are not retryable
    assert not executor._is_retryable(Exception("Aborted"))
    assert not executor._is_retryable(Exception("Invalid"))


# === Raw Store Integration Tests ===

@pytest.mark.asyncio
async def test_records_execution_to_raw_store(task_node, mock_raw_store):
    """Test execution recorded to raw store."""
    executor = TaskExecutor(raw_store=mock_raw_store)

    async def task_callable(args):
        return {"success": True}

    await executor.execute(task_node, task_callable)

    # Should record to raw store
    mock_raw_store.create_task_execution.assert_called_once()
    call_args = mock_raw_store.create_task_execution.call_args[1]
    assert call_args["task_id"] == task_node.task_id
    assert call_args["status"] == "success"


# === Cache Management Tests ===

@pytest.mark.asyncio
async def test_cache_cleanup_limits_size(executor, task_node):
    """Test cache cleans up when too large."""
    # Fill cache beyond limit
    for i in range(1100):
        key = f"key{i}"
        result = TaskExecutionResult(
            task_id=task_node.task_id,
            status=TaskStatus.SUCCESS,
            success=True
        )
        executor._cache_result(key, result)

    # Cache should be limited to ~1000 entries
    assert len(executor._execution_cache) <= 1000


# === Factory Function Test ===

def test_create_task_executor():
    """Test factory function creates executor."""
    executor = create_task_executor(
        max_retries=5,
        base_delay_ms=2000
    )

    assert isinstance(executor, TaskExecutor)
    assert executor.max_retries == 5
    assert executor.base_delay_ms == 2000


# === Integration Test ===

@pytest.mark.asyncio
async def test_full_integration(task_node, mock_abort_monitor, mock_decision_registry, mock_raw_store):
    """Test full integration with all components."""
    executor = TaskExecutor(
        abort_monitor=mock_abort_monitor,
        decision_registry=mock_decision_registry,
        raw_store=mock_raw_store
    )

    async def task_callable(args):
        return {"success": True, "data": "result"}

    result = await executor.execute(task_node, task_callable)

    # Success
    assert result.success

    # Decision recorded
    mock_decision_registry.record_decision.assert_called_once()
    mock_decision_registry.record_outcome.assert_called_once()

    # Execution recorded to raw store
    mock_raw_store.create_task_execution.assert_called_once()

    # Safety envelope checked
    mock_abort_monitor.should_abort.assert_called()
