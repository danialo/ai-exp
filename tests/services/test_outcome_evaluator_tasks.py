"""
Tests for OutcomeEvaluator task outcome evaluation.
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock
from src.services.outcome_evaluator import OutcomeEvaluator, OutcomeConfig, TaskOutcome


@pytest.fixture
def mock_awareness_loop():
    """Create mock awareness loop."""
    mock = Mock()
    mock.last_sim_live = 0.75
    return mock


@pytest.fixture
def mock_raw_store():
    """Create mock raw store."""
    mock = Mock()
    return mock


@pytest.fixture
def mock_provenance_trust():
    """Create mock provenance trust."""
    mock = Mock()
    mock.update_trust = Mock()
    return mock


@pytest.fixture
def evaluator(mock_provenance_trust, mock_awareness_loop, mock_raw_store):
    """Create outcome evaluator with mocks."""
    config = OutcomeConfig(
        enabled=True,
        w_coherence=0.4,
        w_conflict=0.2,
        w_stability=0.2,
        w_validation=0.2,
        horizon_short_hours=2.0,
        horizon_long_hours=24.0
    )
    return OutcomeEvaluator(
        provenance_trust=mock_provenance_trust,
        awareness_loop=mock_awareness_loop,
        raw_store=mock_raw_store,
        config=config
    )


@pytest.mark.asyncio
async def test_evaluate_task_outcome_success(evaluator, mock_awareness_loop):
    """Test evaluating successful task outcome."""
    now = time.time()
    started_at = now - 3600  # 1 hour ago
    ended_at = now - 300  # 5 minutes ago

    # Set up coherence history
    evaluator._coherence_history = [
        (started_at - 300, 0.65),  # Before task
        (ended_at + 300, 0.75),  # After task
    ]

    outcome = await evaluator.evaluate_task_outcome(
        task_id="test_task",
        execution_id="exec_001",
        status="success",
        started_at=started_at,
        ended_at=ended_at,
        horizon="short"
    )

    assert isinstance(outcome, TaskOutcome)
    assert outcome.task_id == "test_task"
    assert outcome.execution_id == "exec_001"
    assert outcome.status == "success"
    assert outcome.horizon == "short"
    assert outcome.duration_ms == int((ended_at - started_at) * 1000)
    assert -1.0 <= outcome.composite_score <= 1.0
    assert -1.0 <= outcome.coherence_delta <= 1.0
    assert -1.0 <= outcome.dissonance_delta <= 1.0
    assert -1.0 <= outcome.satisfaction_score <= 1.0


@pytest.mark.asyncio
async def test_evaluate_task_outcome_failed(evaluator, mock_awareness_loop):
    """Test evaluating failed task outcome."""
    now = time.time()
    started_at = now - 3600
    ended_at = now - 300

    # Set up coherence history (degraded)
    evaluator._coherence_history = [
        (started_at - 300, 0.75),  # Before task
        (ended_at + 300, 0.65),  # After task (worse)
    ]

    outcome = await evaluator.evaluate_task_outcome(
        task_id="test_task",
        execution_id="exec_001",
        status="failed",
        started_at=started_at,
        ended_at=ended_at,
        horizon="short"
    )

    assert outcome.status == "failed"
    # Failed tasks get penalty
    assert outcome.composite_score < 0.0


@pytest.mark.asyncio
async def test_evaluate_task_outcome_partial(evaluator, mock_awareness_loop):
    """Test evaluating partial task outcome."""
    now = time.time()
    started_at = now - 3600
    ended_at = now - 300

    evaluator._coherence_history = [
        (started_at - 300, 0.70),
        (ended_at + 300, 0.70),
    ]

    outcome = await evaluator.evaluate_task_outcome(
        task_id="test_task",
        execution_id="exec_001",
        status="partial",
        started_at=started_at,
        ended_at=ended_at,
        horizon="short"
    )

    assert outcome.status == "partial"
    # Partial tasks get moderate penalty (with floating point tolerance)
    assert outcome.composite_score < 0.01  # Close to or below 0


@pytest.mark.asyncio
async def test_coherence_delta_computation(evaluator, mock_awareness_loop):
    """Test coherence delta computation for tasks."""
    now = time.time()
    started_at = now - 3600
    ended_at = now - 300

    # Set up coherence improvement
    evaluator._coherence_history = [
        (started_at - 300, 0.60),  # Before task
        (ended_at + 300, 0.80),  # After task (+0.20)
    ]

    delta = await evaluator._compute_coherence_delta_for_task(
        started_at, ended_at, now
    )

    # Should detect improvement
    assert delta > 0.0


@pytest.mark.asyncio
async def test_coherence_delta_with_baseline(evaluator, mock_awareness_loop):
    """Test coherence delta with baseline correction."""
    now = time.time()
    started_at = now - 3600
    ended_at = now - 300

    # Set baseline
    evaluator.baseline_coherence = 0.70

    # Set up coherence data
    evaluator._coherence_history = [
        (started_at - 300, 0.65),
        (ended_at + 300, 0.75),
    ]

    delta = await evaluator._compute_coherence_delta_for_task(
        started_at, ended_at, now
    )

    # Baseline correction should apply
    assert isinstance(delta, float)
    assert -1.0 <= delta <= 1.0


@pytest.mark.asyncio
async def test_coherence_delta_no_awareness_loop(mock_provenance_trust):
    """Test coherence delta when no awareness loop available."""
    evaluator = OutcomeEvaluator(
        provenance_trust=mock_provenance_trust,
        awareness_loop=None,  # No awareness loop
    )

    now = time.time()
    delta = await evaluator._compute_coherence_delta_for_task(
        now - 3600, now - 300, now
    )

    # Should return neutral when no awareness loop
    assert delta == 0.0


@pytest.mark.asyncio
async def test_coherence_delta_no_history(evaluator):
    """Test coherence delta with no history."""
    now = time.time()

    # No coherence history
    evaluator._coherence_history = []

    delta = await evaluator._compute_coherence_delta_for_task(
        now - 3600, now - 300, now
    )

    # Should return neutral when no history
    assert delta == 0.0


@pytest.mark.asyncio
async def test_dissonance_delta_computation(evaluator):
    """Test dissonance delta computation."""
    now = time.time()

    delta = await evaluator._compute_dissonance_delta_for_task(
        now - 3600, now - 300, now
    )

    # Currently returns neutral (TODO integration)
    assert delta == 0.0


@pytest.mark.asyncio
async def test_satisfaction_computation(evaluator, mock_raw_store):
    """Test satisfaction score computation."""
    now = time.time()

    score = await evaluator._compute_satisfaction_for_task(
        "exec_001", now - 3600, now
    )

    # Currently returns neutral (TODO integration)
    assert score == 0.0


@pytest.mark.asyncio
async def test_satisfaction_no_raw_store(mock_provenance_trust):
    """Test satisfaction when no raw store available."""
    evaluator = OutcomeEvaluator(
        provenance_trust=mock_provenance_trust,
        raw_store=None,  # No raw store
    )

    now = time.time()
    score = await evaluator._compute_satisfaction_for_task(
        "exec_001", now - 3600, now
    )

    # Should return neutral when no raw store
    assert score == 0.0


@pytest.mark.asyncio
async def test_composite_score_weighting(evaluator, mock_awareness_loop):
    """Test composite score uses correct weights."""
    now = time.time()
    started_at = now - 3600
    ended_at = now - 300

    # Set up clear coherence improvement
    evaluator._coherence_history = [
        (started_at - 300, 0.50),
        (ended_at + 300, 0.90),  # Large improvement
    ]

    outcome = await evaluator.evaluate_task_outcome(
        task_id="test_task",
        execution_id="exec_001",
        status="success",
        started_at=started_at,
        ended_at=ended_at,
        horizon="short"
    )

    # Coherence has 0.4 weight, should dominate
    assert outcome.composite_score > 0.0


@pytest.mark.asyncio
async def test_outcome_clamping(evaluator, mock_awareness_loop):
    """Test outcome scores are clamped to [-1, 1]."""
    now = time.time()
    started_at = now - 3600
    ended_at = now - 300

    # Set up extreme coherence values
    evaluator._coherence_history = [
        (started_at - 300, 0.0),
        (ended_at + 300, 1.0),  # Maximum improvement
    ]

    outcome = await evaluator.evaluate_task_outcome(
        task_id="test_task",
        execution_id="exec_001",
        status="success",
        started_at=started_at,
        ended_at=ended_at,
        horizon="short"
    )

    # Composite score should be clamped
    assert -1.0 <= outcome.composite_score <= 1.0
    assert -1.0 <= outcome.coherence_delta <= 1.0


@pytest.mark.asyncio
async def test_duration_tracking(evaluator):
    """Test task duration is tracked correctly."""
    now = time.time()
    started_at = now - 7200  # 2 hours ago
    ended_at = now - 3600  # 1 hour ago

    evaluator._coherence_history = [(started_at, 0.7), (ended_at, 0.7)]

    outcome = await evaluator.evaluate_task_outcome(
        task_id="test_task",
        execution_id="exec_001",
        status="success",
        started_at=started_at,
        ended_at=ended_at,
        horizon="long"
    )

    # Duration should be 1 hour = 3600000 ms
    assert outcome.duration_ms == 3600000
    assert outcome.horizon == "long"


@pytest.mark.asyncio
async def test_multiple_task_evaluations(evaluator, mock_awareness_loop):
    """Test evaluating multiple task outcomes."""
    now = time.time()

    evaluator._coherence_history = [
        (now - 7200, 0.65),
        (now - 3600, 0.70),
        (now - 1800, 0.75),
        (now - 300, 0.80),
    ]

    outcomes = []
    for i in range(3):
        outcome = await evaluator.evaluate_task_outcome(
            task_id=f"task_{i}",
            execution_id=f"exec_{i:03d}",
            status="success",
            started_at=now - 3600 - (i * 1800),
            ended_at=now - 1800 - (i * 1800),
            horizon="short"
        )
        outcomes.append(outcome)

    assert len(outcomes) == 3
    for outcome in outcomes:
        assert isinstance(outcome, TaskOutcome)
        assert outcome.status == "success"
