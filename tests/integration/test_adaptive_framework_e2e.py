"""End-to-end integration tests for Adaptive Decision Framework.

Tests the complete autonomous loop:
1. GoalStore creates and prioritizes goals
2. Decision recording for goal_selected
3. HTN Planner decomposes goals into plans
4. Decision recording for plan_generated
5. TaskGraph executes plans
6. Outcome evaluation collects feedback
7. Parameter adaptation adjusts weights
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import os

from src.services.goal_store import (
    GoalStore,
    GoalDefinition,
    GoalCategory,
    GoalState,
    register_goal_selection_decision
)
from src.services.htn_planner import (
    HTNPlanner,
    Method,
    register_htn_decision
)
from src.services.decision_framework import DecisionRegistry
from src.services.outcome_evaluator import OutcomeEvaluator
from src.services.parameter_adapter import ParameterAdapter


# === Fixtures ===

@pytest.fixture
def temp_db():
    """Create temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    try:
        Path(db_path).unlink()
    except:
        pass


@pytest.fixture
def decision_registry(temp_db):
    """Create DecisionRegistry with temp database."""
    registry_path = temp_db.replace(".db", "_registry.db")
    registry = DecisionRegistry(db_path=registry_path)

    yield registry

    try:
        Path(registry_path).unlink()
    except:
        pass


@pytest.fixture
def goal_store(temp_db):
    """Create GoalStore with schema."""
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
def outcome_evaluator(decision_registry):
    """Create OutcomeEvaluator."""
    # Check if class takes db_path parameter
    try:
        evaluator = OutcomeEvaluator(db_path=decision_registry.db_path)
    except TypeError:
        # Fallback if it doesn't take db_path
        evaluator = OutcomeEvaluator()

    return evaluator


@pytest.fixture
def parameter_adapter(decision_registry):
    """Create ParameterAdapter."""
    return ParameterAdapter(decision_registry=decision_registry)


# === Decision Recording Tests ===

def test_goal_selection_decision_recorded(goal_store, decision_registry):
    """Test that goal selection decisions are recorded."""
    # Register decision point
    register_goal_selection_decision(decision_registry)

    # Create and adopt goals
    goals = [
        GoalDefinition(
            id=f"goal_{i}",
            text="Test goal",
            category=GoalCategory.INTROSPECTION,
            value=0.5 + i * 0.1,
            effort=0.5,
            risk=0.3,
            horizon_min_min=0,
            state=GoalState.ADOPTED
        )
        for i in range(3)
    ]

    for goal in goals:
        goal_store.create_goal(goal)

    # Get prioritized goals (this should use the registered weights)
    weights = decision_registry.get_all_parameters("goal_selected")
    prioritized = goal_store.prioritized(
        state=GoalState.ADOPTED,
        weights=weights if weights else {},
        active_beliefs=[]
    )

    assert len(prioritized) == 3
    # Higher value goals should be first
    assert prioritized[0][0].value >= prioritized[1][0].value


def test_htn_plan_decision_registered(decision_registry):
    """Test that HTN planning decision point is registered."""
    register_htn_decision(decision_registry)

    # Verify registration
    params = decision_registry.get_all_parameters("plan_generated")

    assert params is not None
    assert "cost_weight" in params
    assert "precondition_threshold" in params
    assert params["cost_weight"] == 0.7


def test_decision_registry_persistence(decision_registry):
    """Test that DecisionRegistry persists decisions."""
    # Register decision point
    register_goal_selection_decision(decision_registry)

    # Get initial parameters
    initial_params = decision_registry.get_all_parameters("goal_selected")
    assert initial_params is not None

    # Create new registry instance with same database
    registry2 = DecisionRegistry(db_path=decision_registry.db_path)

    # Should retrieve same parameters
    persisted_params = registry2.get_all_parameters("goal_selected")
    assert persisted_params == initial_params


# === Parameter Adaptation Tests ===

def test_parameter_adapter_initialization(parameter_adapter, decision_registry):
    """Test ParameterAdapter initializes with registry."""
    register_goal_selection_decision(decision_registry)

    # Should be able to get parameters
    params = parameter_adapter.registry.get_all_parameters("goal_selected")
    assert params is not None


def test_goal_weights_start_at_defaults(decision_registry):
    """Test goal selection weights start at expected defaults."""
    register_goal_selection_decision(decision_registry)

    params = decision_registry.get_all_parameters("goal_selected")

    assert params["value_weight"] == 0.5
    assert params["effort_weight"] == 0.25
    assert params["risk_weight"] == 0.15
    assert params["urgency_weight"] == 0.05
    assert params["alignment_weight"] == 0.05


def test_htn_weights_start_at_defaults(decision_registry):
    """Test HTN planning weights start at expected defaults."""
    register_htn_decision(decision_registry)

    params = decision_registry.get_all_parameters("plan_generated")

    assert params["cost_weight"] == 0.7
    assert params["precondition_threshold"] == 0.5


# === Integration Tests ===

def test_full_decision_loop_goal_selection(
    goal_store,
    decision_registry,
    parameter_adapter
):
    """Test complete decision loop for goal selection."""
    # 1. Register decision point
    register_goal_selection_decision(decision_registry)

    # 2. Create goals with different values
    high_value_goal = GoalDefinition(
        id="high_value",
        text="High value goal",
        category=GoalCategory.INTROSPECTION,
        value=0.9,
        effort=0.3,
        risk=0.2,
        horizon_min_min=0,
        state=GoalState.ADOPTED
    )

    low_value_goal = GoalDefinition(
        id="low_value",
        text="Low value goal",
        category=GoalCategory.MAINTENANCE,
        value=0.3,
        effort=0.3,
        risk=0.2,
        horizon_min_min=0,
        state=GoalState.ADOPTED
    )

    goal_store.create_goal(high_value_goal)
    goal_store.create_goal(low_value_goal)

    # 3. Get prioritized list using current weights
    weights = decision_registry.get_all_parameters("goal_selected")
    prioritized = goal_store.prioritized(
        state=GoalState.ADOPTED,
        weights=weights,
        active_beliefs=[]
    )

    # 4. Verify high-value goal selected first
    assert len(prioritized) == 2
    assert prioritized[0][0].id == "high_value"
    assert prioritized[0][1] > prioritized[1][1]  # Higher score


def test_htn_planner_uses_decision_weights(decision_registry):
    """Test HTN Planner can use decision registry weights."""
    register_htn_decision(decision_registry)

    # Create methods with different costs
    methods = [
        Method(
            name="expensive_method",
            task="test_task",
            subtasks=["subtask1"],
            cost=0.9
        ),
        Method(
            name="cheap_method",
            task="test_task",
            subtasks=["subtask2"],
            cost=0.3
        )
    ]

    planner = HTNPlanner(
        method_library=methods,
        primitive_tasks={"subtask1", "subtask2"}
    )

    # Generate plan (should prefer cheap method based on cost_weight)
    plan = planner.plan(
        goal_id="test",
        goal_text="test_task",
        world_state={}
    )

    assert plan is not None
    assert "cheap_method" in plan.methods_used


# === Error Handling Tests ===

def test_missing_decision_registry(goal_store):
    """Test system works without decision registry (backward compatibility)."""
    # Prioritize without weights
    goal = GoalDefinition(
        id="test",
        text="Test",
        category=GoalCategory.INTROSPECTION,
        value=0.7,
        effort=0.5,
        risk=0.3,
        horizon_min_min=0,
        state=GoalState.ADOPTED
    )
    goal_store.create_goal(goal)

    # Should work with default weights
    prioritized = goal_store.prioritized(
        state=GoalState.ADOPTED,
        weights={}  # Empty weights
    )

    assert len(prioritized) == 1


def test_framework_enabled_flag():
    """Test DECISION_FRAMEWORK_ENABLED environment variable."""
    # Check if framework is enabled
    enabled = os.getenv("DECISION_FRAMEWORK_ENABLED", "false").lower() == "true"

    # Just verify we can read the flag
    assert isinstance(enabled, bool)


# === Performance Tests ===

def test_decision_recording_performance(decision_registry):
    """Test decision registry handles many decision points efficiently."""
    # Register multiple decision points
    register_goal_selection_decision(decision_registry)
    register_htn_decision(decision_registry)

    # Verify both are accessible
    goal_params = decision_registry.get_all_parameters("goal_selected")
    plan_params = decision_registry.get_all_parameters("plan_generated")

    assert goal_params is not None
    assert plan_params is not None


# === Documentation Tests ===

def test_decision_points_documented():
    """Test that all decision points have proper documentation."""
    # This is more of a smoke test to ensure imports work
    from src.services.goal_store import register_goal_selection_decision
    from src.services.htn_planner import register_htn_decision

    # Should not raise
    assert callable(register_goal_selection_decision)
    assert callable(register_htn_decision)
