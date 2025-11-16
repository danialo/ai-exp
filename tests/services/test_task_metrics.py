"""
Tests for TaskMetricsTracker.
"""

import pytest
import time
from src.services.task_metrics import TaskMetricsTracker, TaskMetrics
from src.services.outcome_evaluator import TaskOutcome


@pytest.fixture
def tracker():
    """Create a fresh metrics tracker."""
    return TaskMetricsTracker(max_history=100)


@pytest.fixture
def sample_outcome():
    """Create a sample task outcome."""
    return TaskOutcome(
        task_id="test_task",
        execution_id="exec_001",
        status="success",
        coherence_delta=0.3,
        dissonance_delta=0.2,
        satisfaction_score=0.4,
        duration_ms=1500,
        composite_score=0.35,
        horizon="short",
        evaluated_at=time.time()
    )


def test_tracker_initialization(tracker):
    """Test tracker initializes correctly."""
    assert tracker.max_history == 100
    assert len(tracker.metrics) == 0
    assert len(tracker.type_metrics) == 0
    assert len(tracker.all_outcomes) == 0


def test_record_single_outcome(tracker, sample_outcome):
    """Test recording a single outcome."""
    tracker.record_outcome(sample_outcome, "self_reflection")

    # Check task metrics created
    assert "test_task" in tracker.metrics
    metrics = tracker.metrics["test_task"]

    assert metrics.total_executions == 1
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 0
    assert metrics.avg_duration_ms == 1500
    assert metrics.avg_composite_score == 0.35
    assert metrics.last_status == "success"


def test_record_multiple_outcomes(tracker):
    """Test recording multiple outcomes for same task."""
    outcomes = [
        TaskOutcome(
            task_id="test_task",
            execution_id=f"exec_{i:03d}",
            status="success" if i % 2 == 0 else "failed",
            coherence_delta=0.3,
            dissonance_delta=0.2,
            satisfaction_score=0.4,
            duration_ms=1000 + i * 100,
            composite_score=0.3 if i % 2 == 0 else -0.2,
            horizon="short",
            evaluated_at=time.time()
        )
        for i in range(10)
    ]

    for outcome in outcomes:
        tracker.record_outcome(outcome, "self_reflection")

    metrics = tracker.metrics["test_task"]
    assert metrics.total_executions == 10
    assert metrics.successful_executions == 5  # Even indices
    assert metrics.failed_executions == 5  # Odd indices
    assert metrics.success_rate == 0.5
    assert metrics.failure_rate == 0.5


def test_duration_statistics(tracker):
    """Test duration statistics are computed correctly."""
    outcomes = [
        TaskOutcome(
            task_id="test_task",
            execution_id=f"exec_{i:03d}",
            status="success",
            coherence_delta=0.0,
            dissonance_delta=0.0,
            satisfaction_score=0.0,
            duration_ms=duration,
            composite_score=0.0,
            horizon="short",
            evaluated_at=time.time()
        )
        for i, duration in enumerate([1000, 2000, 3000, 4000, 5000])
    ]

    for outcome in outcomes:
        tracker.record_outcome(outcome, "self_reflection")

    metrics = tracker.metrics["test_task"]
    assert metrics.min_duration_ms == 1000
    assert metrics.max_duration_ms == 5000
    assert metrics.avg_duration_ms == 3000  # (1000+2000+3000+4000+5000)/5


def test_type_level_aggregation(tracker):
    """Test type-level metrics aggregation."""
    # Create outcomes for different tasks of same type
    for task_id in ["task_1", "task_2", "task_3"]:
        outcome = TaskOutcome(
            task_id=task_id,
            execution_id=f"{task_id}_exec",
            status="success",
            coherence_delta=0.3,
            dissonance_delta=0.2,
            satisfaction_score=0.4,
            duration_ms=1500,
            composite_score=0.35,
            horizon="short",
            evaluated_at=time.time()
        )
        tracker.record_outcome(outcome, "self_reflection")

    # Check type metrics
    assert "self_reflection" in tracker.type_metrics
    type_metrics = tracker.type_metrics["self_reflection"]
    assert type_metrics.total_executions == 3
    assert type_metrics.successful_executions == 3


def test_recent_outcomes_limit(tracker):
    """Test recent outcomes are limited to 10."""
    outcomes = [
        TaskOutcome(
            task_id="test_task",
            execution_id=f"exec_{i:03d}",
            status="success",
            coherence_delta=0.0,
            dissonance_delta=0.0,
            satisfaction_score=0.0,
            duration_ms=1000,
            composite_score=0.0,
            horizon="short",
            evaluated_at=time.time()
        )
        for i in range(20)
    ]

    for outcome in outcomes:
        tracker.record_outcome(outcome, "self_reflection")

    metrics = tracker.metrics["test_task"]
    assert len(metrics.recent_outcomes) == 10  # Limited to 10


def test_get_failing_tasks(tracker):
    """Test identifying failing tasks."""
    # Create high-failure task
    for i in range(10):
        outcome = TaskOutcome(
            task_id="failing_task",
            execution_id=f"exec_{i:03d}",
            status="failed" if i < 7 else "success",  # 70% failure
            coherence_delta=0.0,
            dissonance_delta=0.0,
            satisfaction_score=0.0,
            duration_ms=1000,
            composite_score=-0.5 if i < 7 else 0.3,
            horizon="short",
            evaluated_at=time.time()
        )
        tracker.record_outcome(outcome, "self_reflection")

    # Create successful task
    for i in range(10):
        outcome = TaskOutcome(
            task_id="successful_task",
            execution_id=f"exec_{i:03d}",
            status="success",
            coherence_delta=0.3,
            dissonance_delta=0.2,
            satisfaction_score=0.4,
            duration_ms=1000,
            composite_score=0.35,
            horizon="short",
            evaluated_at=time.time()
        )
        tracker.record_outcome(outcome, "self_reflection")

    failing = tracker.get_failing_tasks(threshold=0.3)
    assert len(failing) == 1
    assert failing[0].task_id == "failing_task"


def test_get_low_performing_tasks(tracker):
    """Test identifying low-performing tasks."""
    # Create low-performing task
    for i in range(5):
        outcome = TaskOutcome(
            task_id="low_perf_task",
            execution_id=f"exec_{i:03d}",
            status="success",
            coherence_delta=-0.3,
            dissonance_delta=-0.2,
            satisfaction_score=-0.4,
            duration_ms=1000,
            composite_score=-0.35,
            horizon="short",
            evaluated_at=time.time()
        )
        tracker.record_outcome(outcome, "self_reflection")

    # Create high-performing task
    for i in range(5):
        outcome = TaskOutcome(
            task_id="high_perf_task",
            execution_id=f"exec_{i:03d}",
            status="success",
            coherence_delta=0.4,
            dissonance_delta=0.3,
            satisfaction_score=0.5,
            duration_ms=1000,
            composite_score=0.45,
            horizon="short",
            evaluated_at=time.time()
        )
        tracker.record_outcome(outcome, "self_reflection")

    low_performing = tracker.get_low_performing_tasks(threshold=-0.2)
    assert len(low_performing) == 1
    assert low_performing[0].task_id == "low_perf_task"


def test_detect_degradation(tracker):
    """Test degradation detection."""
    # Start with good performance
    for i in range(10):
        outcome = TaskOutcome(
            task_id="degrading_task",
            execution_id=f"exec_{i:03d}",
            status="success",
            coherence_delta=0.3,
            dissonance_delta=0.2,
            satisfaction_score=0.4,
            duration_ms=1000,
            composite_score=0.35,
            horizon="short",
            evaluated_at=time.time()
        )
        tracker.record_outcome(outcome, "self_reflection")

    # Then add recent poor performance
    for i in range(10, 15):
        outcome = TaskOutcome(
            task_id="degrading_task",
            execution_id=f"exec_{i:03d}",
            status="failed",
            coherence_delta=-0.3,
            dissonance_delta=-0.2,
            satisfaction_score=-0.4,
            duration_ms=1000,
            composite_score=-0.4,
            horizon="short",
            evaluated_at=time.time()
        )
        tracker.record_outcome(outcome, "self_reflection")

    # Should detect degradation
    assert tracker.detect_degradation("degrading_task", window=5)


def test_no_degradation_stable_task(tracker):
    """Test no degradation detected for stable task."""
    for i in range(15):
        outcome = TaskOutcome(
            task_id="stable_task",
            execution_id=f"exec_{i:03d}",
            status="success",
            coherence_delta=0.3,
            dissonance_delta=0.2,
            satisfaction_score=0.4,
            duration_ms=1000,
            composite_score=0.35,
            horizon="short",
            evaluated_at=time.time()
        )
        tracker.record_outcome(outcome, "self_reflection")

    # Should not detect degradation
    assert not tracker.detect_degradation("stable_task", window=5)


def test_correlation_analysis(tracker):
    """Test correlation analysis computation."""
    # Create outcomes with positive correlation between duration and success
    for i in range(20):
        duration = 1000 + i * 100
        score = 0.0 + i * 0.05  # Positive correlation
        outcome = TaskOutcome(
            task_id=f"task_{i}",
            execution_id=f"exec_{i:03d}",
            status="success",
            coherence_delta=score,
            dissonance_delta=0.0,
            satisfaction_score=score,
            duration_ms=duration,
            composite_score=score,
            horizon="short",
            evaluated_at=time.time()
        )
        tracker.record_outcome(outcome, "self_reflection")

    correlations = tracker.get_correlation_analysis()
    assert "duration_vs_success" in correlations
    assert "coherence_vs_satisfaction" in correlations

    # Should be strong positive correlation
    assert correlations["duration_vs_success"] > 0.9
    assert correlations["coherence_vs_satisfaction"] > 0.9


def test_telemetry(tracker, sample_outcome):
    """Test telemetry output."""
    tracker.record_outcome(sample_outcome, "self_reflection")

    telemetry = tracker.get_telemetry()
    assert telemetry["total_tasks_tracked"] == 1
    assert telemetry["total_task_types"] == 1
    assert telemetry["total_outcomes_recorded"] == 1
    assert "correlations" in telemetry


def test_history_limit(tracker):
    """Test global history is limited."""
    tracker_small = TaskMetricsTracker(max_history=10)

    for i in range(20):
        outcome = TaskOutcome(
            task_id=f"task_{i}",
            execution_id=f"exec_{i:03d}",
            status="success",
            coherence_delta=0.0,
            dissonance_delta=0.0,
            satisfaction_score=0.0,
            duration_ms=1000,
            composite_score=0.0,
            horizon="short",
            evaluated_at=time.time()
        )
        tracker_small.record_outcome(outcome, "self_reflection")

    # Should only keep last 10
    assert len(tracker_small.all_outcomes) == 10
