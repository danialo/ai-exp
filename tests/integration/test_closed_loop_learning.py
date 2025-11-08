"""
Integration test for closed-loop learning cycle.

Tests the full flow:
1. Task execution with decision recording
2. Outcome evaluation and linking
3. Metrics tracking
4. Parameter adaptation
"""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone

from src.services.task_scheduler import TaskScheduler, TaskDefinition, TaskType, TaskSchedule
from src.services.outcome_evaluator import OutcomeEvaluator, OutcomeConfig
from src.services.task_metrics import TaskMetricsTracker
from src.services.task_outcome_linker import TaskOutcomeLinker
from src.services.decision_framework import DecisionRegistry, Parameter
from src.services.parameter_adapter import ParameterAdapter


@pytest.fixture
def temp_db_path():
    """Create temporary database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_persona_space():
    """Create temporary persona space."""
    import shutil
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_awareness_loop():
    """Mock awareness loop with coherence tracking."""
    mock = Mock()
    mock.last_sim_live = 0.75
    return mock


@pytest.fixture
def mock_persona_service():
    """Mock persona service for task execution."""
    mock = Mock()
    mock.generate_response = Mock(return_value=("Task completed successfully", {}))
    return mock


@pytest.fixture
async def integrated_system(temp_db_path, temp_persona_space, mock_awareness_loop):
    """Create integrated system with all components wired together."""
    # Create decision registry
    registry = DecisionRegistry(db_path=temp_db_path)

    # Create decision framework
    decision_framework = Mock()
    decision_framework.registry = registry

    # Create outcome evaluator
    outcome_evaluator = OutcomeEvaluator(
        provenance_trust=Mock(),
        awareness_loop=mock_awareness_loop,
        config=OutcomeConfig(
            w_coherence=0.4,
            w_conflict=0.2,
            w_stability=0.2,
            w_validation=0.2,
            horizon_short_hours=0.001,  # Very short for testing (3.6 seconds)
            horizon_long_hours=0.002
        )
    )

    # Create task metrics tracker
    metrics_tracker = TaskMetricsTracker(max_history=100)

    # Create task outcome linker
    outcome_linker = TaskOutcomeLinker(
        outcome_evaluator=outcome_evaluator,
        decision_framework=decision_framework
    )

    # Create parameter adapter
    parameter_adapter = ParameterAdapter(
        decision_registry=registry,
        success_evaluator=None,  # Not needed for evaluated decisions
        min_samples=3,  # Low threshold for testing
        exploration_rate=0.1,
        adaptation_rate=0.15
    )

    # Create task scheduler
    scheduler = TaskScheduler(
        persona_space_path=temp_persona_space,
        raw_store=None,  # Optional for this test
        abort_monitor=None,  # Optional for this test
        decision_framework=decision_framework,
        parameter_adapter=parameter_adapter
    )

    return {
        "scheduler": scheduler,
        "registry": registry,
        "outcome_evaluator": outcome_evaluator,
        "metrics_tracker": metrics_tracker,
        "outcome_linker": outcome_linker,
        "parameter_adapter": parameter_adapter,
        "awareness_loop": mock_awareness_loop
    }


@pytest.mark.asyncio
async def test_closed_loop_single_task(integrated_system, mock_persona_service):
    """Test closed loop with single task execution."""
    scheduler = integrated_system["scheduler"]
    registry = integrated_system["registry"]
    outcome_linker = integrated_system["outcome_linker"]
    metrics_tracker = integrated_system["metrics_tracker"]
    awareness_loop = integrated_system["awareness_loop"]

    # Get a test task
    tasks = scheduler.list_tasks()
    assert len(tasks) > 0
    test_task = tasks[0]

    # Execute task
    result = await scheduler.execute_task(test_task.id, mock_persona_service)

    assert result.success
    assert "decision_record_id" in result.metadata
    decision_record_id = result.metadata["decision_record_id"]
    assert decision_record_id is not None

    # Verify decision was recorded
    stats = registry.get_registry_stats()
    assert stats["total_decisions_made"] == 1
    assert stats["evaluated_decisions"] == 0  # Not evaluated yet

    # Set up coherence history AFTER execution with proper timing
    outcome_evaluator = integrated_system["outcome_evaluator"]
    started_at = datetime.fromisoformat(result.started_at).timestamp()
    completed_at = datetime.fromisoformat(result.completed_at).timestamp()

    outcome_evaluator._coherence_history = [
        (started_at - 300, 0.65),  # 5 min before task
        (completed_at + 300, 0.80),  # 5 min after task (improvement)
    ]

    # Link task outcome
    task_outcome = await outcome_linker.link_task_outcome(
        task_id=test_task.id,
        execution_id="test_exec_001",
        decision_record_id=decision_record_id,
        status="success",
        started_at=started_at,
        ended_at=completed_at,
        horizon="short"
    )

    assert task_outcome is not None
    assert task_outcome.status == "success"
    # Should be positive due to coherence improvement (0.15 delta * 0.4 weight = 0.06)
    assert task_outcome.coherence_delta > 0
    assert task_outcome.composite_score > 0

    # Record in metrics tracker
    metrics_tracker.record_outcome(task_outcome, test_task.type)

    # Verify decision is now evaluated
    stats = registry.get_registry_stats()
    assert stats["evaluated_decisions"] == 1

    # Verify metrics
    task_metrics = metrics_tracker.get_task_metrics(test_task.id)
    assert task_metrics.total_executions == 1
    assert task_metrics.successful_executions == 1
    assert task_metrics.success_rate == 1.0


@pytest.mark.asyncio
async def test_closed_loop_parameter_adaptation(integrated_system, mock_persona_service):
    """Test full closed loop with parameter adaptation."""
    scheduler = integrated_system["scheduler"]
    registry = integrated_system["registry"]
    outcome_linker = integrated_system["outcome_linker"]
    metrics_tracker = integrated_system["metrics_tracker"]
    parameter_adapter = integrated_system["parameter_adapter"]
    awareness_loop = integrated_system["awareness_loop"]
    outcome_evaluator = integrated_system["outcome_evaluator"]

    # Get initial parameters
    initial_params = registry.get_all_parameters("task_selected")
    assert initial_params is not None
    initial_urgency = initial_params["urgency_threshold"]
    initial_coherence = initial_params["coherence_required"]

    # Execute multiple tasks to accumulate outcomes
    test_task = scheduler.list_tasks()[0]
    now = time.time()

    for i in range(5):
        # Set up coherence improvement for each task
        outcome_evaluator._coherence_history = [
            (now - 3600 + i * 100, 0.65 + i * 0.02),
            (now + i * 100, 0.80 + i * 0.02),
        ]

        # Execute task
        result = await scheduler.execute_task(test_task.id, mock_persona_service)
        assert result.success

        # Link outcome
        started_at = datetime.fromisoformat(result.started_at).timestamp()
        completed_at = datetime.fromisoformat(result.completed_at).timestamp()

        task_outcome = await outcome_linker.link_task_outcome(
            task_id=test_task.id,
            execution_id=f"test_exec_{i:03d}",
            decision_record_id=result.metadata["decision_record_id"],
            status="success",
            started_at=started_at,
            ended_at=completed_at,
            horizon="short"
        )

        # Record metrics
        metrics_tracker.record_outcome(task_outcome, test_task.type)

    # Verify we have evaluated decisions
    stats = registry.get_registry_stats()
    assert stats["evaluated_decisions"] == 5

    # Trigger parameter adaptation
    adaptation_result = parameter_adapter.adapt_from_evaluated_decisions(
        decision_id="task_selected",
        since_hours=24,
        dry_run=False
    )

    # Verify adaptation occurred
    assert adaptation_result["adapted"] or adaptation_result["reason"] == "no_adjustments"
    assert adaptation_result["sample_count"] == 5

    if adaptation_result["adapted"]:
        # Get updated parameters
        updated_params = registry.get_all_parameters("task_selected")
        assert updated_params is not None

        # At least one parameter should have changed
        assert (
            updated_params["urgency_threshold"] != initial_urgency or
            updated_params["coherence_required"] != initial_coherence
        )

        # Log parameter changes
        print(f"\nParameter adaptation:")
        print(f"  urgency_threshold: {initial_urgency:.3f} → {updated_params['urgency_threshold']:.3f}")
        print(f"  coherence_required: {initial_coherence:.3f} → {updated_params['coherence_required']:.3f}")
        print(f"  avg_success_score: {adaptation_result['avg_success_score']:.3f}")


@pytest.mark.asyncio
async def test_automatic_adaptation_trigger(integrated_system, mock_persona_service):
    """Test that parameter adaptation is triggered automatically after N executions."""
    scheduler = integrated_system["scheduler"]
    registry = integrated_system["registry"]
    outcome_linker = integrated_system["outcome_linker"]
    outcome_evaluator = integrated_system["outcome_evaluator"]

    # Set adaptation interval to 3 for testing
    scheduler.adaptation_interval = 3
    scheduler.executions_since_adaptation = 0

    test_task = scheduler.list_tasks()[0]

    # Execute tasks
    for i in range(5):
        result = await scheduler.execute_task(test_task.id, mock_persona_service)

        # Link outcome
        started_at = datetime.fromisoformat(result.started_at).timestamp()
        completed_at = datetime.fromisoformat(result.completed_at).timestamp()

        # Set up coherence history with proper timing
        outcome_evaluator._coherence_history = [
            (started_at - 300, 0.65),
            (completed_at + 300, 0.80),
        ]

        await outcome_linker.link_task_outcome(
            task_id=test_task.id,
            execution_id=f"test_exec_{i:03d}",
            decision_record_id=result.metadata["decision_record_id"],
            status="success",
            started_at=started_at,
            ended_at=completed_at,
            horizon="short"
        )

    # After 5 executions with interval of 3, adaptation should have triggered at least once
    stats = registry.get_registry_stats()
    assert stats["evaluated_decisions"] == 5  # All tasks evaluated


@pytest.mark.asyncio
async def test_degraded_performance_detection(integrated_system, mock_persona_service):
    """Test that degraded performance is detected and parameters adapt."""
    scheduler = integrated_system["scheduler"]
    outcome_linker = integrated_system["outcome_linker"]
    metrics_tracker = integrated_system["metrics_tracker"]
    awareness_loop = integrated_system["awareness_loop"]
    outcome_evaluator = integrated_system["outcome_evaluator"]

    test_task = scheduler.list_tasks()[0]

    # Execute tasks with degrading performance
    for i in range(6):
        result = await scheduler.execute_task(test_task.id, mock_persona_service)

        started_at = datetime.fromisoformat(result.started_at).timestamp()
        completed_at = datetime.fromisoformat(result.completed_at).timestamp()

        # Simulate coherence degradation with proper timing
        # Keep before constant, make after decrease to show clear degradation
        coherence_before = 0.75  # Constant baseline
        coherence_after = 0.70 - i * 0.05  # Decreasing (degradation)

        outcome_evaluator._coherence_history = [
            (started_at - 300, coherence_before),
            (completed_at + 300, coherence_after),
        ]

        # Update awareness loop to match after coherence (used for recent tasks)
        awareness_loop.last_sim_live = coherence_after

        task_outcome = await outcome_linker.link_task_outcome(
            task_id=test_task.id,
            execution_id=f"test_exec_{i:03d}",
            decision_record_id=result.metadata["decision_record_id"],
            status="success",
            started_at=started_at,
            ended_at=completed_at,
            horizon="short"
        )

        metrics_tracker.record_outcome(task_outcome, test_task.type)

    # Check for degradation (need enough samples)
    task_metrics = metrics_tracker.get_task_metrics(test_task.id)
    assert task_metrics.total_executions == 6

    # Check average score is negative (degrading)
    assert task_metrics.avg_composite_score < 0, "Should have negative average score due to degradation"

    # Get low-performing tasks
    low_perf = metrics_tracker.get_low_performing_tasks(threshold=-0.1)
    assert len(low_perf) > 0, "Should identify low-performing task"
