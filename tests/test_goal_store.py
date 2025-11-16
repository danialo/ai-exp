"""Unit tests for GoalStore service."""

import json
import pytest
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from src.services.goal_store import (
    GoalStore,
    GoalDefinition,
    GoalCategory,
    GoalState,
    create_goal_store,
)


# === Fixtures ===

@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    try:
        Path(db_path).unlink()
    except:
        pass


@pytest.fixture
def goal_store(temp_db):
    """Create GoalStore instance with temp database."""
    # Create migration SQL manually since migration file may not exist
    conn = sqlite3.connect(temp_db)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            category TEXT NOT NULL,
            value REAL NOT NULL,
            effort REAL NOT NULL,
            risk REAL NOT NULL,
            horizon_min_min INTEGER NOT NULL,
            horizon_max_min INTEGER,
            aligns_with TEXT,
            contradicts TEXT,
            success_metrics TEXT,
            state TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT,
            version INTEGER NOT NULL,
            deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS goal_idempotency (
            key TEXT PRIMARY KEY,
            op TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_goals_state ON goals(state);
        CREATE INDEX IF NOT EXISTS idx_goals_category ON goals(category);
        CREATE INDEX IF NOT EXISTS idx_goals_updated_at ON goals(updated_at);
    """)
    conn.commit()
    conn.close()

    return GoalStore(temp_db)


@pytest.fixture
def sample_goal():
    """Create a sample goal for testing."""
    return GoalDefinition(
        id="goal_test123",
        text="Complete comprehensive code review",
        category=GoalCategory.INTROSPECTION,
        value=0.8,
        effort=0.4,
        risk=0.2,
        horizon_min_min=0,
        horizon_max_min=1440,  # 24 hours
        aligns_with=["belief_quality"],
        contradicts=["belief_speed"],
        success_metrics={"coverage": 0.9, "issues_found": 5.0}
    )


# === Basic CRUD Tests ===

def test_create_goal(goal_store, sample_goal):
    """Test creating a goal."""
    created = goal_store.create_goal(sample_goal)

    assert created.id == sample_goal.id
    assert created.text == sample_goal.text
    assert created.category == GoalCategory.INTROSPECTION
    assert created.value == 0.8
    assert created.effort == 0.4
    assert created.risk == 0.2
    assert created.state == GoalState.PROPOSED
    assert created.version == 0
    assert created.deleted_at is None


def test_create_goal_with_idempotency(goal_store, sample_goal):
    """Test idempotent goal creation."""
    # First creation
    goal1 = goal_store.create_goal(sample_goal, idempotency_key="key123")

    # Second creation with same key should return existing
    goal2 = goal_store.create_goal(sample_goal, idempotency_key="key123")

    assert goal1.id == goal2.id
    assert goal1.created_at == goal2.created_at


def test_get_goal(goal_store, sample_goal):
    """Test retrieving a goal by ID."""
    goal_store.create_goal(sample_goal)

    retrieved = goal_store.get_goal(sample_goal.id)

    assert retrieved is not None
    assert retrieved.id == sample_goal.id
    assert retrieved.text == sample_goal.text
    assert retrieved.aligns_with == ["belief_quality"]
    assert retrieved.contradicts == ["belief_speed"]
    assert retrieved.success_metrics == {"coverage": 0.9, "issues_found": 5.0}


def test_get_goal_not_found(goal_store):
    """Test retrieving non-existent goal returns None."""
    result = goal_store.get_goal("nonexistent")
    assert result is None


def test_list_goals(goal_store):
    """Test listing goals."""
    # Create multiple goals
    for i in range(5):
        goal = GoalDefinition(
            id=f"goal_{i}",
            text=f"Goal {i}",
            category=GoalCategory.INTROSPECTION if i % 2 == 0 else GoalCategory.EXPLORATION,
            value=0.5,
            effort=0.3,
            risk=0.1,
            horizon_min_min=0,
            state=GoalState.PROPOSED if i < 3 else GoalState.ADOPTED
        )
        goal_store.create_goal(goal)

    # List all goals
    all_goals = goal_store.list_goals(limit=10)
    assert len(all_goals) == 5

    # Filter by state
    proposed = goal_store.list_goals(state=GoalState.PROPOSED)
    assert len(proposed) == 3

    adopted = goal_store.list_goals(state=GoalState.ADOPTED)
    assert len(adopted) == 2

    # Filter by category
    introspection = goal_store.list_goals(category=GoalCategory.INTROSPECTION)
    assert len(introspection) == 3

    exploration = goal_store.list_goals(category=GoalCategory.EXPLORATION)
    assert len(exploration) == 2


def test_list_goals_pagination(goal_store):
    """Test goal listing with pagination."""
    # Create 10 goals
    for i in range(10):
        goal = GoalDefinition(
            id=f"goal_{i}",
            text=f"Goal {i}",
            category=GoalCategory.INTROSPECTION,
            value=0.5,
            effort=0.3,
            risk=0.1,
            horizon_min_min=0
        )
        goal_store.create_goal(goal)

    # Get first page
    page1 = goal_store.list_goals(limit=3, offset=0)
    assert len(page1) == 3

    # Get second page
    page2 = goal_store.list_goals(limit=3, offset=3)
    assert len(page2) == 3

    # Pages should be different
    assert page1[0].id != page2[0].id


def test_update_goal(goal_store, sample_goal):
    """Test updating a goal."""
    goal_store.create_goal(sample_goal)

    # Update goal
    updated = goal_store.update_goal(
        sample_goal.id,
        {"text": "Updated goal text", "value": 0.9},
        expected_version=0
    )

    assert updated is not None
    assert updated.text == "Updated goal text"
    assert updated.value == 0.9
    assert updated.version == 1
    assert updated.effort == 0.4  # Unchanged


def test_update_goal_version_conflict(goal_store, sample_goal):
    """Test optimistic locking with version conflict."""
    goal_store.create_goal(sample_goal)

    # Update with wrong version
    result = goal_store.update_goal(
        sample_goal.id,
        {"text": "New text"},
        expected_version=999  # Wrong version
    )

    assert result is None


def test_update_goal_complex_fields(goal_store, sample_goal):
    """Test updating complex fields (lists, dicts)."""
    goal_store.create_goal(sample_goal)

    updated = goal_store.update_goal(
        sample_goal.id,
        {
            "aligns_with": ["new_belief_1", "new_belief_2"],
            "success_metrics": {"new_metric": 10.0}
        },
        expected_version=0
    )

    assert updated.aligns_with == ["new_belief_1", "new_belief_2"]
    assert updated.success_metrics == {"new_metric": 10.0}


def test_abandon_goal(goal_store, sample_goal):
    """Test abandoning a goal."""
    goal_store.create_goal(sample_goal)

    abandoned = goal_store.abandon_goal(sample_goal.id)

    assert abandoned is not None
    assert abandoned.state == GoalState.ABANDONED
    assert abandoned.deleted_at is not None
    assert abandoned.version == 1


def test_abandon_goal_with_idempotency(goal_store, sample_goal):
    """Test idempotent goal abandonment."""
    goal_store.create_goal(sample_goal)

    # First abandon
    goal1 = goal_store.abandon_goal(sample_goal.id, idempotency_key="abandon_key")

    # Second abandon with same key should return existing
    goal2 = goal_store.abandon_goal(sample_goal.id, idempotency_key="abandon_key")

    assert goal1.state == goal2.state
    assert goal1.deleted_at == goal2.deleted_at


# === Scoring Tests ===

def test_compute_urgency_no_deadline():
    """Test urgency computation with no deadline."""
    created = datetime.now(timezone.utc)
    urgency = GoalStore.compute_urgency(created, None)
    assert urgency == 0.0


def test_compute_urgency_far_future():
    """Test urgency for distant deadline."""
    created = datetime.now(timezone.utc)
    horizon_max_min = 48 * 60  # 48 hours
    urgency = GoalStore.compute_urgency(created, horizon_max_min)
    assert urgency == 0.0  # More than 24 hours away


def test_compute_urgency_approaching():
    """Test urgency as deadline approaches."""
    # Created 12 hours ago with 24-hour deadline
    created = datetime.now(timezone.utc) - timedelta(hours=12)
    horizon_max_min = 24 * 60  # 24 hours

    urgency = GoalStore.compute_urgency(created, horizon_max_min)

    # 12 hours remaining = 0.5 urgency
    assert 0.4 < urgency < 0.6


def test_compute_urgency_overdue():
    """Test urgency for overdue goal."""
    created = datetime.now(timezone.utc) - timedelta(hours=48)
    horizon_max_min = 24 * 60  # 24 hours (expired)

    urgency = GoalStore.compute_urgency(created, horizon_max_min)
    assert urgency == -1.0  # Overdue


def test_score_goal_basic():
    """Test basic goal scoring."""
    goal = GoalDefinition(
        id="test",
        text="Test goal",
        category=GoalCategory.INTROSPECTION,
        value=0.8,
        effort=0.4,  # Inverted to 0.6
        risk=0.2,    # Inverted to 0.8
        horizon_min_min=0
    )

    weights = {
        "value_weight": 0.5,
        "effort_weight": 0.25,
        "risk_weight": 0.15,
        "urgency_weight": 0.05,
        "alignment_weight": 0.05
    }

    score = GoalStore.score_goal(goal, weights)

    # value: 0.5 * 0.8 = 0.4
    # effort: 0.25 * (1 - 0.4) = 0.15
    # risk: 0.15 * (1 - 0.2) = 0.12
    # urgency: 0 (no deadline)
    # alignment: 0 (no beliefs)
    # Total: 0.4 + 0.15 + 0.12 = 0.67
    assert 0.6 < score < 0.8


def test_score_goal_alignment_bonus():
    """Test alignment bonus in scoring."""
    goal = GoalDefinition(
        id="test",
        text="Test goal",
        category=GoalCategory.INTROSPECTION,
        value=0.5,
        effort=0.5,
        risk=0.5,
        horizon_min_min=0,
        aligns_with=["belief_1", "belief_2"]
    )

    weights = {
        "value_weight": 0.5,
        "effort_weight": 0.25,
        "risk_weight": 0.15,
        "urgency_weight": 0.05,
        "alignment_weight": 0.05
    }

    # With all beliefs active
    score_with = GoalStore.score_goal(
        goal,
        weights,
        active_beliefs=["belief_1", "belief_2"]
    )

    # Without active beliefs
    score_without = GoalStore.score_goal(
        goal,
        weights,
        active_beliefs=[]
    )

    assert score_with > score_without


def test_score_goal_contradiction_penalty():
    """Test contradiction penalty in scoring."""
    goal = GoalDefinition(
        id="test",
        text="Test goal",
        category=GoalCategory.INTROSPECTION,
        value=0.8,
        effort=0.2,
        risk=0.1,
        horizon_min_min=0,
        contradicts=["belief_x"]
    )

    weights = {
        "value_weight": 0.5,
        "effort_weight": 0.25,
        "risk_weight": 0.15,
        "urgency_weight": 0.05,
        "alignment_weight": 0.05
    }

    # With contradiction active
    score_with_contradiction = GoalStore.score_goal(
        goal,
        weights,
        active_beliefs=["belief_x"]
    )

    # Without contradiction
    score_without = GoalStore.score_goal(
        goal,
        weights,
        active_beliefs=[]
    )

    # Contradiction should heavily penalize score
    assert score_with_contradiction < score_without
    assert score_with_contradiction == 0.0  # Penalty should clamp to 0


def test_score_goal_urgency_factor():
    """Test urgency factor in scoring."""
    # Goal approaching deadline
    created = datetime.now(timezone.utc) - timedelta(hours=20)
    goal = GoalDefinition(
        id="test",
        text="Test goal",
        category=GoalCategory.INTROSPECTION,
        value=0.5,
        effort=0.5,
        risk=0.5,
        horizon_min_min=0,
        horizon_max_min=24 * 60,  # 24-hour deadline (4 hours remaining)
        created_at=created
    )

    weights = {
        "value_weight": 0.5,
        "effort_weight": 0.25,
        "risk_weight": 0.15,
        "urgency_weight": 0.05,
        "alignment_weight": 0.05
    }

    score_urgent = GoalStore.score_goal(goal, weights, now=datetime.now(timezone.utc))

    # Same goal but no deadline
    goal.horizon_max_min = None
    score_normal = GoalStore.score_goal(goal, weights)

    assert score_urgent > score_normal


# === Prioritization Tests ===

def test_prioritized_basic(goal_store):
    """Test prioritized goal listing."""
    # Create goals with different values
    for i in range(3):
        goal = GoalDefinition(
            id=f"goal_{i}",
            text=f"Goal {i}",
            category=GoalCategory.INTROSPECTION,
            value=0.3 + i * 0.3,  # 0.3, 0.6, 0.9
            effort=0.5,
            risk=0.5,
            horizon_min_min=0,
            state=GoalState.PROPOSED
        )
        goal_store.create_goal(goal)

    weights = {
        "value_weight": 1.0,  # Only value matters
        "effort_weight": 0.0,
        "risk_weight": 0.0
    }

    prioritized = goal_store.prioritized(
        state=GoalState.PROPOSED,
        weights=weights
    )

    assert len(prioritized) == 3
    # Highest value should be first
    assert prioritized[0][0].id == "goal_2"
    assert prioritized[1][0].id == "goal_1"
    assert prioritized[2][0].id == "goal_0"

    # Scores should be descending
    assert prioritized[0][1] >= prioritized[1][1]
    assert prioritized[1][1] >= prioritized[2][1]


def test_prioritized_filter_by_state(goal_store):
    """Test prioritized filtering by state."""
    # Create goals in different states
    for i in range(4):
        state = GoalState.PROPOSED if i < 2 else GoalState.ADOPTED
        goal = GoalDefinition(
            id=f"goal_{i}",
            text=f"Goal {i}",
            category=GoalCategory.INTROSPECTION,
            value=0.7,
            effort=0.3,
            risk=0.2,
            horizon_min_min=0,
            state=state
        )
        goal_store.create_goal(goal)

    # Get only PROPOSED goals
    proposed = goal_store.prioritized(state=GoalState.PROPOSED)
    assert len(proposed) == 2

    # Get only ADOPTED goals
    adopted = goal_store.prioritized(state=GoalState.ADOPTED)
    assert len(adopted) == 2


def test_prioritized_with_limit(goal_store):
    """Test prioritized with limit parameter."""
    # Create 10 goals
    for i in range(10):
        goal = GoalDefinition(
            id=f"goal_{i}",
            text=f"Goal {i}",
            category=GoalCategory.INTROSPECTION,
            value=0.5,
            effort=0.3,
            risk=0.2,
            horizon_min_min=0,
            state=GoalState.PROPOSED
        )
        goal_store.create_goal(goal)

    # Get top 5
    top5 = goal_store.prioritized(state=GoalState.PROPOSED, limit=5)
    assert len(top5) == 5


# === Adoption Tests ===

@patch('src.services.goal_store.append_event')
def test_adopt_goal_success(mock_append, goal_store, sample_goal):
    """Test successful goal adoption."""
    goal_store.create_goal(sample_goal)

    adopted, goal, details = goal_store.adopt_goal(
        sample_goal.id,
        active_belief_ids=[]
    )

    assert adopted is True
    assert goal is not None
    assert goal.state == GoalState.ADOPTED
    assert goal.version == 1
    assert details == {}

    # Verify ledger event was logged
    assert mock_append.call_count >= 1


@patch('src.services.goal_store.append_event')
def test_adopt_goal_contradiction_blocks(mock_append, goal_store, sample_goal):
    """Test goal adoption blocked by contradiction."""
    goal_store.create_goal(sample_goal)

    # Try to adopt with contradicting belief active
    adopted, goal, details = goal_store.adopt_goal(
        sample_goal.id,
        active_belief_ids=["belief_speed"]  # Contradicts this goal
    )

    assert adopted is False
    assert goal is not None
    assert goal.state == GoalState.PROPOSED  # State unchanged
    assert details["blocked_by_belief"] is True
    assert "belief_speed" in details["belief_ids"]
    assert details["reason"] == "contradiction"


@patch('src.services.goal_store.append_event')
def test_adopt_goal_idempotency(mock_append, goal_store, sample_goal):
    """Test idempotent goal adoption."""
    goal_store.create_goal(sample_goal)

    # First adoption
    adopted1, goal1, details1 = goal_store.adopt_goal(
        sample_goal.id,
        idempotency_key="adopt_key",
        active_belief_ids=[]
    )

    # Second adoption with same key
    adopted2, goal2, details2 = goal_store.adopt_goal(
        sample_goal.id,
        idempotency_key="adopt_key",
        active_belief_ids=[]
    )

    assert adopted1 is True
    assert adopted2 is True
    assert details2.get("reason") == "idempotent"


def test_adopt_goal_not_found(goal_store):
    """Test adopting non-existent goal."""
    adopted, goal, details = goal_store.adopt_goal("nonexistent")

    assert adopted is False
    assert goal is None
    assert details["reason"] == "not_found"


# === Factory Function Test ===

def test_create_goal_store(temp_db):
    """Test factory function."""
    store = create_goal_store(temp_db)
    assert isinstance(store, GoalStore)
    assert store.db_path == temp_db


# === Edge Cases ===

def test_empty_store_operations(goal_store):
    """Test operations on empty store."""
    goals = goal_store.list_goals()
    assert len(goals) == 0

    prioritized = goal_store.prioritized()
    assert len(prioritized) == 0

    result = goal_store.get_goal("nonexistent")
    assert result is None


def test_goal_with_empty_arrays(goal_store):
    """Test goal with empty arrays."""
    goal = GoalDefinition(
        id="empty_test",
        text="Empty arrays test",
        category=GoalCategory.MAINTENANCE,
        value=0.5,
        effort=0.5,
        risk=0.5,
        horizon_min_min=0,
        aligns_with=[],
        contradicts=[],
        success_metrics={}
    )

    created = goal_store.create_goal(goal)
    retrieved = goal_store.get_goal(created.id)

    assert retrieved.aligns_with == []
    assert retrieved.contradicts == []
    assert retrieved.success_metrics == {}


def test_goal_with_none_horizon(goal_store):
    """Test goal with no deadline."""
    goal = GoalDefinition(
        id="no_deadline",
        text="No deadline test",
        category=GoalCategory.EXPLORATION,
        value=0.6,
        effort=0.4,
        risk=0.3,
        horizon_min_min=0,
        horizon_max_min=None
    )

    created = goal_store.create_goal(goal)
    retrieved = goal_store.get_goal(created.id)

    assert retrieved.horizon_max_min is None


def test_score_goal_edge_values():
    """Test scoring with edge case values."""
    # Maximum value, minimum effort/risk
    goal_best = GoalDefinition(
        id="best",
        text="Best case",
        category=GoalCategory.INTROSPECTION,
        value=1.0,
        effort=0.0,
        risk=0.0,
        horizon_min_min=0
    )

    # Minimum value, maximum effort/risk
    goal_worst = GoalDefinition(
        id="worst",
        text="Worst case",
        category=GoalCategory.INTROSPECTION,
        value=0.0,
        effort=1.0,
        risk=1.0,
        horizon_min_min=0
    )

    weights = {
        "value_weight": 0.5,
        "effort_weight": 0.25,
        "risk_weight": 0.25
    }

    score_best = GoalStore.score_goal(goal_best, weights)
    score_worst = GoalStore.score_goal(goal_worst, weights)

    assert score_best > score_worst
    assert 0.0 <= score_best <= 1.0
    assert 0.0 <= score_worst <= 1.0
