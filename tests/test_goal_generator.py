"""Tests for GoalGenerator service."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from src.services.goal_generator import GoalGenerator, GoalProposal, PatternDetector
from src.services.goal_store import GoalStore, GoalCategory, GoalState, GoalSource
from src.services.detectors import TaskFailureDetector


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    try:
        Path(db_path).unlink()
    except:
        pass


@pytest.fixture
def goal_store(temp_db):
    return GoalStore(temp_db)


def test_goal_proposal_creation():
    """Test creating a goal proposal."""
    proposal = GoalProposal(
        text="Fix task failure",
        category=GoalCategory.MAINTENANCE,
        pattern_detected="task_failure",
        evidence={"count": 5},
        confidence=0.8,
        estimated_value=0.7,
        estimated_effort=0.5,
        estimated_risk=0.3,
        detector_name="test_detector"
    )

    assert proposal.proposal_id.startswith("prop_")
    assert proposal.text == "Fix task failure"
    assert proposal.confidence == 0.8
    assert proposal.expires_at is not None


def test_goal_generator_initialization(goal_store):
    """Test GoalGenerator initialization."""
    generator = GoalGenerator(
        goal_store=goal_store,
        min_confidence=0.7
    )

    assert generator.goal_store == goal_store
    assert generator.min_confidence == 0.7
    assert len(generator.detectors) == 0


def test_add_detector(goal_store):
    """Test adding a detector."""
    generator = GoalGenerator(goal_store=goal_store)
    detector = TaskFailureDetector()

    generator.add_detector(detector)

    assert len(generator.detectors) == 1
    assert generator.detectors[0].name() == "task_failure_detector"


def test_create_system_goal(goal_store):
    """Test creating a system goal from proposal."""
    generator = GoalGenerator(goal_store=goal_store)

    proposal = GoalProposal(
        text="Improve test coverage",
        category=GoalCategory.MAINTENANCE,
        pattern_detected="coverage_drop",
        evidence={"old": 0.8, "new": 0.6},
        confidence=0.85,
        estimated_value=0.8,
        estimated_effort=0.4,
        estimated_risk=0.2,
        detector_name="coverage_detector"
    )

    goal = generator.create_system_goal(proposal)

    assert goal is not None
    assert goal.text == "Improve test coverage"
    assert goal.source == GoalSource.SYSTEM
    assert goal.created_by == "coverage_detector"
    assert goal.proposal_id == proposal.proposal_id
    assert goal.state == GoalState.PROPOSED  # Low confidence


def test_auto_approve_high_confidence(goal_store):
    """Test auto-approval for high confidence proposals."""
    generator = GoalGenerator(
        goal_store=goal_store,
        auto_approve_threshold=0.9
    )

    proposal = GoalProposal(
        text="Critical fix",
        category=GoalCategory.MAINTENANCE,
        pattern_detected="critical_issue",
        evidence={},
        confidence=0.95,  # Above threshold
        estimated_value=0.9,
        estimated_effort=0.3,
        estimated_risk=0.1,
        detector_name="critical_detector"
    )

    goal = generator.create_system_goal(proposal)

    assert goal.state == GoalState.ADOPTED
    assert goal.auto_approved is True


def test_evaluate_proposal_low_confidence(goal_store):
    """Test proposal rejection for low confidence."""
    generator = GoalGenerator(
        goal_store=goal_store,
        min_confidence=0.7
    )

    proposal = GoalProposal(
        text="Maybe fix this",
        category=GoalCategory.MAINTENANCE,
        pattern_detected="minor_issue",
        evidence={},
        confidence=0.5,  # Below threshold
        estimated_value=0.5,
        estimated_effort=0.5,
        estimated_risk=0.5,
        detector_name="test_detector"
    )

    approved, reason = generator.evaluate_proposal(proposal)

    assert not approved
    assert "confidence" in reason


def test_task_failure_detector_properties():
    """Test TaskFailureDetector configuration."""
    detector = TaskFailureDetector(
        failure_threshold=5,
        lookback_hours=48,
        scan_interval=30
    )

    assert detector.name() == "task_failure_detector"
    assert detector.scan_interval_minutes() == 30
    assert detector.enabled() is True


def test_get_telemetry(goal_store):
    """Test telemetry data collection."""
    generator = GoalGenerator(goal_store=goal_store)
    generator.add_detector(TaskFailureDetector())

    telemetry = generator.get_telemetry()

    assert "detectors_enabled" in telemetry
    assert "goals_created_today" in telemetry
    assert telemetry["detectors_total"] == 1
