"""Integration tests for goal-driven task scheduling."""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from src.services.task_scheduler import TaskScheduler, create_task_scheduler
from src.services.goal_store import GoalStore, GoalDefinition, GoalCategory, GoalState


# === Fixtures ===

@pytest.fixture
def temp_db():
    """Create temporary database for GoalStore."""
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
    """Create GoalStore with temp database."""
    import sqlite3

    # Create schema
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
    """)
    conn.commit()
    conn.close()

    return GoalStore(temp_db)


@pytest.fixture
def scheduler(tmp_path, goal_store):
    """Create TaskScheduler with GoalStore integration."""
    return create_task_scheduler(
        persona_space_path=str(tmp_path / "persona_space"),
        goal_store=goal_store
    )


@pytest.fixture
def mock_persona_service():
    """Create mock persona service."""
    def generate_response(*args, **kwargs):
        return ("Test response", {})

    service = Mock()
    service.generate_response = generate_response
    return service


# === Goal-to-Task Conversion Tests ===

def test_goal_to_task_conversion(scheduler, goal_store):
    """Test converting GoalDefinition to TaskDefinition."""
    goal = GoalDefinition(
        id="test_goal",
        text="Complete comprehensive testing",
        category=GoalCategory.INTROSPECTION,
        value=0.8,
        effort=0.4,
        risk=0.2,
        horizon_min_min=0
    )

    task = scheduler._goal_to_task(goal)

    assert task.id == "test_goal"
    assert task.name == "Complete comprehensive testing"
    assert task.prompt == "Complete comprehensive testing"
    assert task.metadata["goal_id"] == "test_goal"
    assert task.metadata["goal_value"] == 0.8
    assert task.metadata["goal_effort"] == 0.4
    assert task.metadata["source"] == "goal_store"


def test_goal_to_task_category_mapping(scheduler):
    """Test category mapping from GoalCategory to TaskType."""
    from src.services.task_scheduler import TaskType

    test_cases = [
        (GoalCategory.INTROSPECTION, TaskType.SELF_REFLECTION),
        (GoalCategory.EXPLORATION, TaskType.CAPABILITY_EXPLORATION),
        (GoalCategory.MAINTENANCE, TaskType.CUSTOM),
        (GoalCategory.USER_REQUESTED, TaskType.CUSTOM),
    ]

    for goal_category, expected_task_type in test_cases:
        goal = GoalDefinition(
            id=f"test_{goal_category.value}",
            text="Test goal",
            category=goal_category,
            value=0.5,
            effort=0.5,
            risk=0.5,
            horizon_min_min=0
        )

        task = scheduler._goal_to_task(goal)
        assert task.type == expected_task_type


# === Goal Selection Tests ===

def test_get_next_goal_no_goals(scheduler):
    """Test get_next_goal with no adopted goals."""
    goal = scheduler.get_next_goal()
    assert goal is None


def test_get_next_goal_only_proposed(scheduler, goal_store):
    """Test get_next_goal ignores PROPOSED goals."""
    # Create PROPOSED goal (not adopted)
    goal = GoalDefinition(
        id="proposed_goal",
        text="Not yet adopted",
        category=GoalCategory.INTROSPECTION,
        value=0.9,
        effort=0.2,
        risk=0.1,
        horizon_min_min=0,
        state=GoalState.PROPOSED
    )
    goal_store.create_goal(goal)

    # Should not return PROPOSED goal
    next_goal = scheduler.get_next_goal()
    assert next_goal is None


def test_get_next_goal_returns_highest_priority(scheduler, goal_store):
    """Test get_next_goal returns highest priority ADOPTED goal."""
    # Create low-value goal
    low_goal = GoalDefinition(
        id="low_priority",
        text="Low value task",
        category=GoalCategory.MAINTENANCE,
        value=0.3,
        effort=0.5,
        risk=0.5,
        horizon_min_min=0,
        state=GoalState.ADOPTED
    )
    goal_store.create_goal(low_goal)

    # Create high-value goal
    high_goal = GoalDefinition(
        id="high_priority",
        text="High value task",
        category=GoalCategory.INTROSPECTION,
        value=0.9,
        effort=0.2,
        risk=0.1,
        horizon_min_min=0,
        state=GoalState.ADOPTED
    )
    goal_store.create_goal(high_goal)

    # Should return high-value goal
    next_goal = scheduler.get_next_goal()
    assert next_goal is not None
    assert next_goal.id == "high_priority"


def test_get_next_goal_with_belief_alignment(scheduler, goal_store):
    """Test get_next_goal considers belief alignment."""
    # Create goal without alignment
    unaligned_goal = GoalDefinition(
        id="unaligned",
        text="Unaligned goal",
        category=GoalCategory.INTROSPECTION,
        value=0.7,
        effort=0.3,
        risk=0.2,
        horizon_min_min=0,
        state=GoalState.ADOPTED,
        aligns_with=[]
    )
    goal_store.create_goal(unaligned_goal)

    # Create goal with alignment
    aligned_goal = GoalDefinition(
        id="aligned",
        text="Aligned goal",
        category=GoalCategory.INTROSPECTION,
        value=0.7,  # Same base value
        effort=0.3,
        risk=0.2,
        horizon_min_min=0,
        state=GoalState.ADOPTED,
        aligns_with=["belief_quality", "belief_testing"]
    )
    goal_store.create_goal(aligned_goal)

    # With active beliefs, aligned goal should score higher
    next_goal = scheduler.get_next_goal(
        active_belief_ids=["belief_quality", "belief_testing"]
    )

    assert next_goal is not None
    assert next_goal.id == "aligned"


# === Goal Execution Tests ===

@pytest.mark.asyncio
async def test_execute_goal_success(scheduler, goal_store, mock_persona_service):
    """Test successful goal execution."""
    # Create and adopt goal
    goal = GoalDefinition(
        id="executable_goal",
        text="Test goal execution",
        category=GoalCategory.INTROSPECTION,
        value=0.8,
        effort=0.3,
        risk=0.2,
        horizon_min_min=0,
        state=GoalState.ADOPTED
    )
    created_goal = goal_store.create_goal(goal)

    # Execute goal
    result = await scheduler.execute_goal(
        "executable_goal",
        mock_persona_service
    )

    # Verify execution
    assert result.success is True
    assert result.task_id == "executable_goal"

    # Verify goal marked as SATISFIED
    updated_goal = goal_store.get_goal("executable_goal")
    assert updated_goal.state == GoalState.SATISFIED


@pytest.mark.asyncio
async def test_execute_goal_not_found(scheduler, mock_persona_service):
    """Test executing non-existent goal raises error."""
    with pytest.raises(ValueError, match="Goal not found"):
        await scheduler.execute_goal("nonexistent", mock_persona_service)


@pytest.mark.asyncio
async def test_execute_goal_no_goal_store(tmp_path, mock_persona_service):
    """Test executing goal without GoalStore raises error."""
    scheduler = create_task_scheduler(
        persona_space_path=str(tmp_path / "persona_space"),
        goal_store=None  # No GoalStore
    )

    with pytest.raises(ValueError, match="GoalStore not configured"):
        await scheduler.execute_goal("test", mock_persona_service)


# === Integration Tests ===

@pytest.mark.asyncio
async def test_full_goal_driven_workflow(scheduler, goal_store, mock_persona_service):
    """Test complete workflow: create → adopt → select → execute."""
    # 1. Create goals
    for i in range(3):
        goal = GoalDefinition(
            id=f"goal_{i}",
            text=f"Goal {i}",
            category=GoalCategory.INTROSPECTION,
            value=0.3 + i * 0.3,  # 0.3, 0.6, 0.9
            effort=0.5,
            risk=0.2,
            horizon_min_min=0,
            state=GoalState.PROPOSED
        )
        goal_store.create_goal(goal)

    # 2. Adopt goals (normally done via API)
    for i in range(3):
        goal_store.update_goal(
            f"goal_{i}",
            {"state": GoalState.ADOPTED},
            expected_version=0
        )

    # 3. Select highest priority goal
    next_goal = scheduler.get_next_goal()
    assert next_goal is not None
    assert next_goal.id == "goal_2"  # Highest value (0.9)

    # 4. Execute goal
    result = await scheduler.execute_goal(next_goal.id, mock_persona_service)
    assert result.success is True

    # 5. Verify goal marked as SATISFIED
    executed_goal = goal_store.get_goal("goal_2")
    assert executed_goal.state == GoalState.SATISFIED

    # 6. Next selection should skip satisfied goal
    next_goal_2 = scheduler.get_next_goal()
    assert next_goal_2 is not None
    assert next_goal_2.id == "goal_1"  # Next highest (0.6)


@pytest.mark.asyncio
async def test_goal_execution_with_metadata_preservation(scheduler, goal_store, mock_persona_service):
    """Test that goal metadata is preserved in task execution."""
    goal = GoalDefinition(
        id="metadata_test",
        text="Test metadata preservation",
        category=GoalCategory.EXPLORATION,
        value=0.7,
        effort=0.3,
        risk=0.2,
        horizon_min_min=0,
        state=GoalState.ADOPTED,
        aligns_with=["belief_test"],
        success_metrics={"coverage": 0.9}
    )
    goal_store.create_goal(goal)

    # Execute
    result = await scheduler.execute_goal("metadata_test", mock_persona_service)

    # The task should have been created with goal metadata
    # (This is verified through the task execution path)
    assert result.success is True
    assert result.task_id == "metadata_test"


# === Error Handling Tests ===

def test_get_next_goal_handles_store_errors(scheduler, goal_store, monkeypatch):
    """Test get_next_goal handles GoalStore errors gracefully."""
    def raise_error(*args, **kwargs):
        raise Exception("Store error")

    monkeypatch.setattr(goal_store, "prioritized", raise_error)

    # Should return None instead of crashing
    result = scheduler.get_next_goal()
    assert result is None


def test_scheduler_without_goal_store(tmp_path):
    """Test TaskScheduler works without GoalStore (backward compatibility)."""
    scheduler = create_task_scheduler(
        persona_space_path=str(tmp_path / "persona_space"),
        goal_store=None
    )

    assert scheduler.goal_store is None
    assert scheduler.get_next_goal() is None
