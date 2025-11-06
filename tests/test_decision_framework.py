"""
Tests for Adaptive Decision Framework.

Covers:
- Decision point registration
- Decision recording
- Parameter updates
- Success signal evaluation
- Abort condition monitoring
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone

from src.services.decision_framework import (
    DecisionRegistry,
    Parameter,
    DecisionOutcome
)
from src.services.success_signal_evaluator import SuccessSignalEvaluator
from src.services.abort_condition_monitor import AbortConditionMonitor, AbortThresholds


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def registry(temp_db):
    """Create decision registry for testing."""
    return DecisionRegistry(db_path=temp_db)


def test_decision_registration(registry):
    """Test registering a decision point."""
    registry.register_decision(
        decision_id="test_decision",
        subsystem="test",
        description="Test decision point",
        parameters={
            "threshold": Parameter(
                name="threshold",
                current_value=0.5,
                min_value=0.0,
                max_value=1.0
            )
        },
        success_metrics=["coherence_delta"]
    )

    # Verify registration
    param_value = registry.get_parameter("test_decision", "threshold")
    assert param_value == 0.5

    params = registry.get_all_parameters("test_decision")
    assert params == {"threshold": 0.5}


def test_decision_recording(registry):
    """Test recording a decision."""
    # Register decision first
    registry.register_decision(
        decision_id="test_decision",
        subsystem="test",
        description="Test",
        parameters={
            "threshold": Parameter("threshold", 0.5, 0.0, 1.0)
        },
        success_metrics=["test_metric"]
    )

    # Record decision
    record_id = registry.record_decision(
        decision_id="test_decision",
        context={"test": "context"},
        parameters_used={"threshold": 0.5},
        outcome_snapshot={"coherence": 0.7}
    )

    assert record_id.startswith("dec_test_decision_")

    # Verify recorded
    decisions = registry.get_recent_decisions("test_decision", limit=1)
    assert len(decisions) == 1
    assert decisions[0]["record_id"] == record_id
    assert decisions[0]["evaluated"] is False


def test_parameter_update(registry):
    """Test updating a parameter value."""
    # Register decision
    registry.register_decision(
        decision_id="test_decision",
        subsystem="test",
        description="Test",
        parameters={
            "threshold": Parameter("threshold", 0.5, 0.0, 1.0)
        },
        success_metrics=["test_metric"]
    )

    # Update parameter
    success = registry.update_parameter(
        decision_id="test_decision",
        param_name="threshold",
        new_value=0.7,
        reason="test_update"
    )

    assert success is True

    # Verify update
    param_value = registry.get_parameter("test_decision", "threshold")
    assert param_value == 0.7


def test_parameter_bounds(registry):
    """Test parameter bounds are enforced."""
    registry.register_decision(
        decision_id="test_decision",
        subsystem="test",
        description="Test",
        parameters={
            "threshold": Parameter("threshold", 0.5, 0.0, 1.0)
        },
        success_metrics=["test_metric"]
    )

    # Try to set above max - should clamp
    registry.update_parameter(
        decision_id="test_decision",
        param_name="threshold",
        new_value=1.5
    )

    param_value = registry.get_parameter("test_decision", "threshold")
    assert param_value == 1.0  # Clamped to max


def test_decision_outcome_update(registry):
    """Test updating decision with outcome."""
    registry.register_decision(
        decision_id="test_decision",
        subsystem="test",
        description="Test",
        parameters={"threshold": Parameter("threshold", 0.5, 0.0, 1.0)},
        success_metrics=["test_metric"]
    )

    record_id = registry.record_decision(
        decision_id="test_decision",
        context={},
        parameters_used={"threshold": 0.5}
    )

    # Update with outcome
    outcome = DecisionOutcome(
        decision_record_id=record_id,
        success_score=0.8,
        coherence_delta=0.1,
        dissonance_delta=-0.05,
        satisfaction_delta=0.2,
        aborted=False
    )

    registry.update_decision_outcome(record_id, outcome)

    # Verify outcome recorded
    decisions = registry.get_recent_decisions("test_decision", evaluated_only=True)
    assert len(decisions) == 1
    assert decisions[0]["evaluated"] is True
    assert decisions[0]["success_score"] == 0.8


def test_unevaluated_decisions(registry):
    """Test querying unevaluated decisions."""
    registry.register_decision(
        decision_id="test_decision",
        subsystem="test",
        description="Test",
        parameters={"threshold": Parameter("threshold", 0.5, 0.0, 1.0)},
        success_metrics=["test_metric"]
    )

    # Record some decisions
    for i in range(3):
        registry.record_decision(
            decision_id="test_decision",
            context={"iteration": i},
            parameters_used={"threshold": 0.5}
        )

    # Get unevaluated (need to use older_than_hours=0 for immediate query)
    unevaluated = registry.get_unevaluated_decisions(
        decision_id="test_decision",
        older_than_hours=0
    )

    assert len(unevaluated) == 3


def test_success_score_computation():
    """Test success score computation."""
    evaluator = SuccessSignalEvaluator()

    # Set baselines and targets
    evaluator.set_baselines(coherence=0.7, dissonance=0.2, satisfaction=0.6)
    evaluator.set_targets(coherence=0.85, dissonance=0.1, satisfaction=0.8)

    # Test positive improvement
    score = evaluator.compute_success_score(
        coherence_delta=0.05,  # Improved
        dissonance_delta=-0.03,  # Reduced (good)
        satisfaction_delta=0.1  # Improved
    )

    assert score > 0  # Should be positive

    # Test negative outcome
    score = evaluator.compute_success_score(
        coherence_delta=-0.1,  # Degraded
        dissonance_delta=0.05,  # Increased (bad)
        satisfaction_delta=-0.15  # Degraded
    )

    assert score < 0  # Should be negative


def test_abort_condition_monitoring():
    """Test abort condition checks."""
    monitor = AbortConditionMonitor(
        thresholds=AbortThresholds(
            belief_rate_limit=5
        )
    )

    # Record some belief formations
    for i in range(6):
        monitor.record_belief_formation()

    # Check if runaway detected
    aborted, reason = monitor.check_belief_runaway()

    assert aborted is True
    assert "runaway" in reason.lower()


def test_abort_recovery():
    """Test abort condition recovery."""
    monitor = AbortConditionMonitor()

    # Manually trigger abort
    monitor._trigger_abort("test_abort")

    assert monitor.aborted is True
    assert monitor.abort_reason == "test_abort"

    # Manual reset
    monitor.reset()

    assert monitor.aborted is False
    assert monitor.abort_reason is None


def test_registry_stats(registry):
    """Test registry statistics."""
    # Register multiple decision types
    for i in range(3):
        registry.register_decision(
            decision_id=f"decision_{i}",
            subsystem="test",
            description=f"Test decision {i}",
            parameters={"param": Parameter("param", 0.5, 0.0, 1.0)},
            success_metrics=["metric"]
        )

    # Record some decisions
    for i in range(3):
        for j in range(i + 1):  # Different counts per type
            registry.record_decision(
                decision_id=f"decision_{i}",
                context={},
                parameters_used={"param": 0.5}
            )

    stats = registry.get_registry_stats()

    assert stats["total_decision_types"] == 3
    assert stats["total_decisions_made"] == 6  # 1 + 2 + 3
    assert stats["evaluated_decisions"] == 0
    assert len(stats["decisions_by_type"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
