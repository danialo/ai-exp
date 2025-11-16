"""Integration tests for HTN Planner with GoalStore and TaskGraph."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from src.services.htn_planner import (
    Method,
    Task,
    HTNPlanner,
    plan_to_task_graph,
    register_htn_decision
)
from src.services.goal_store import GoalStore, GoalDefinition, GoalCategory, GoalState
from src.services.task_graph import TaskGraph, TaskState
from src.services.decision_framework import DecisionRegistry


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
def test_methods():
    """Create test HTN methods."""
    return [
        Method(
            name="improve_quality",
            task="improve_code_quality",
            preconditions=["has_codebase"],
            subtasks=["analyze_code", "fix_issues", "verify_fixes"],
            cost=0.7
        ),
        Method(
            name="analyze_code",
            task="analyze_code",
            preconditions=[],
            subtasks=["run_linter", "check_complexity"],
            cost=0.3
        )
    ]


@pytest.fixture
def primitive_tasks():
    """Define primitive tasks."""
    return {
        "run_linter",
        "check_complexity",
        "fix_issues",
        "verify_fixes"
    }


# === HTN + GoalStore Integration Tests ===

def test_htn_plan_from_goal(test_methods, primitive_tasks, goal_store):
    """Test generating HTN plan from a GoalStore goal."""
    # Create and adopt a goal
    goal = GoalDefinition(
        id="goal_quality",
        text="improve_code_quality",
        category=GoalCategory.INTROSPECTION,
        value=0.8,
        effort=0.5,
        risk=0.3,
        horizon_min_min=0,
        state=GoalState.ADOPTED
    )
    goal_store.create_goal(goal)

    # Create HTN planner
    planner = HTNPlanner(
        method_library=test_methods,
        primitive_tasks=primitive_tasks
    )

    # Generate plan from goal
    plan = planner.plan(
        goal_id=goal.id,
        goal_text=goal.text,
        world_state={"has_codebase": True}
    )

    assert plan is not None
    assert plan.goal_id == "goal_quality"
    assert plan.goal_text == "improve_code_quality"
    assert len(plan.tasks) == 4  # run_linter, check_complexity, fix_issues, verify_fixes
    assert all(task.primitive for task in plan.tasks)


def test_htn_plan_with_goal_metadata(test_methods, primitive_tasks, goal_store):
    """Test that goal metadata is preserved in HTN plan."""
    goal = GoalDefinition(
        id="goal_metadata",
        text="improve_code_quality",
        category=GoalCategory.INTROSPECTION,
        value=0.9,
        effort=0.4,
        risk=0.2,
        horizon_min_min=0,
        state=GoalState.ADOPTED,
        aligns_with=["belief_quality"],
        success_metrics={"coverage": 0.8}
    )
    goal_store.create_goal(goal)

    planner = HTNPlanner(
        method_library=test_methods,
        primitive_tasks=primitive_tasks
    )

    plan = planner.plan(
        goal_id=goal.id,
        goal_text=goal.text,
        world_state={"has_codebase": True}
    )

    assert plan is not None
    # Goal metadata should be accessible via plan
    assert plan.goal_id == goal.id
    assert plan.goal_text == goal.text


# === HTN + TaskGraph Integration Tests ===

def test_plan_to_task_graph_conversion(test_methods, primitive_tasks):
    """Test converting HTN plan to TaskGraph."""
    planner = HTNPlanner(
        method_library=test_methods,
        primitive_tasks=primitive_tasks
    )

    plan = planner.plan(
        goal_id="goal_1",
        goal_text="improve_code_quality",
        world_state={"has_codebase": True}
    )

    assert plan is not None

    # Convert to TaskGraph
    task_graph = plan_to_task_graph(plan)

    assert task_graph is not None
    assert len(task_graph.nodes) == 4  # 4 primitive tasks
    assert all(task_graph.nodes[task_id].state == TaskState.PENDING
               for task_id in task_graph.nodes)


def test_task_graph_sequential_dependencies(test_methods, primitive_tasks):
    """Test TaskGraph has correct sequential dependencies from HTN plan."""
    planner = HTNPlanner(
        method_library=test_methods,
        primitive_tasks=primitive_tasks
    )

    plan = planner.plan(
        goal_id="goal_1",
        goal_text="improve_code_quality",
        world_state={"has_codebase": True}
    )

    task_graph = plan_to_task_graph(plan)

    # First task should have no dependencies
    first_task_id = plan.tasks[0].task_id
    assert len(task_graph.nodes[first_task_id].dependencies) == 0

    # Subsequent tasks should depend on previous task
    for i in range(1, len(plan.tasks)):
        current_id = plan.tasks[i].task_id
        prev_id = plan.tasks[i - 1].task_id

        assert prev_id in task_graph.nodes[current_id].dependencies


def test_task_graph_preserves_plan_metadata(test_methods, primitive_tasks):
    """Test TaskGraph tasks preserve HTN plan metadata."""
    planner = HTNPlanner(
        method_library=test_methods,
        primitive_tasks=primitive_tasks
    )

    plan = planner.plan(
        goal_id="goal_metadata",
        goal_text="improve_code_quality",
        world_state={"has_codebase": True}
    )

    task_graph = plan_to_task_graph(plan)

    # Check metadata is preserved in normalized_args
    for task in plan.tasks:
        node = task_graph.nodes[task.task_id]

        assert node.normalized_args["_plan_id"] == plan.plan_id
        assert node.normalized_args["_goal_id"] == plan.goal_id
        assert node.normalized_args["_goal_text"] == plan.goal_text


def test_task_graph_ready_tasks(test_methods, primitive_tasks):
    """Test TaskGraph correctly identifies ready tasks."""
    planner = HTNPlanner(
        method_library=test_methods,
        primitive_tasks=primitive_tasks
    )

    plan = planner.plan(
        goal_id="goal_1",
        goal_text="improve_code_quality",
        world_state={"has_codebase": True}
    )

    task_graph = plan_to_task_graph(plan)

    # Only first task should be ready (no dependencies)
    ready_tasks = task_graph.get_ready_tasks()

    assert len(ready_tasks) == 1
    # get_ready_tasks() returns task IDs (strings)
    assert ready_tasks[0] == plan.tasks[0].task_id


# === Full Pipeline Integration Tests ===

def test_full_pipeline_goal_to_taskgraph(goal_store, test_methods, primitive_tasks):
    """Test full pipeline: GoalStore → HTN Planner → TaskGraph."""
    # Step 1: Create and adopt goal
    goal = GoalDefinition(
        id="pipeline_test",
        text="improve_code_quality",
        category=GoalCategory.INTROSPECTION,
        value=0.8,
        effort=0.5,
        risk=0.3,
        horizon_min_min=0,
        state=GoalState.ADOPTED
    )
    goal_store.create_goal(goal)

    # Step 2: Get next goal (simulate autonomous selection)
    prioritized = goal_store.prioritized(
        state=GoalState.ADOPTED,
        limit=1
    )
    assert len(prioritized) == 1
    selected_goal, score = prioritized[0]

    # Step 3: Generate HTN plan
    planner = HTNPlanner(
        method_library=test_methods,
        primitive_tasks=primitive_tasks
    )

    plan = planner.plan(
        goal_id=selected_goal.id,
        goal_text=selected_goal.text,
        world_state={"has_codebase": True}
    )

    assert plan is not None

    # Step 4: Convert to TaskGraph
    task_graph = plan_to_task_graph(plan)

    assert task_graph is not None
    assert len(task_graph.nodes) == 4

    # Step 5: Verify execution readiness
    ready_tasks = task_graph.get_ready_tasks()
    assert len(ready_tasks) == 1


def test_multiple_goals_to_plans(goal_store, test_methods, primitive_tasks):
    """Test generating multiple HTN plans from different goals."""
    # Create multiple goals
    goals = [
        GoalDefinition(
            id=f"goal_{i}",
            text="improve_code_quality",
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

    planner = HTNPlanner(
        method_library=test_methods,
        primitive_tasks=primitive_tasks
    )

    # Generate plans for all goals
    plans = []
    for goal in goals:
        plan = planner.plan(
            goal_id=goal.id,
            goal_text=goal.text,
            world_state={"has_codebase": True}
        )
        plans.append(plan)

    assert all(plan is not None for plan in plans)
    assert len(plans) == 3

    # Each plan should be independent
    plan_ids = [plan.plan_id for plan in plans]
    assert len(set(plan_ids)) == 3  # All unique


# === Decision Framework Integration ===

def test_htn_decision_registration(temp_db):
    """Test HTN Planner decision framework integration."""
    registry = DecisionRegistry(db_path=temp_db)

    # Register decision point
    register_htn_decision(registry)

    # Verify parameters exist
    params = registry.get_all_parameters("plan_generated")

    assert params is not None
    assert "cost_weight" in params
    assert "precondition_threshold" in params


# === Error Handling Tests ===

def test_empty_plan_to_taskgraph():
    """Test plan_to_task_graph handles empty plan gracefully."""
    from src.services.htn_planner import Plan

    empty_plan = Plan(
        plan_id="empty",
        goal_id="goal_1",
        goal_text="test",
        tasks=[],
        total_cost=0.0
    )

    task_graph = plan_to_task_graph(empty_plan)

    assert task_graph is not None
    assert len(task_graph.nodes) == 0


def test_plan_failure_with_unsatisfied_preconditions(goal_store, test_methods, primitive_tasks):
    """Test HTN planning fails gracefully when preconditions not met."""
    goal = GoalDefinition(
        id="impossible_goal",
        text="improve_code_quality",
        category=GoalCategory.INTROSPECTION,
        value=0.8,
        effort=0.5,
        risk=0.3,
        horizon_min_min=0,
        state=GoalState.ADOPTED
    )
    goal_store.create_goal(goal)

    planner = HTNPlanner(
        method_library=test_methods,
        primitive_tasks=primitive_tasks
    )

    # Precondition not satisfied
    plan = planner.plan(
        goal_id=goal.id,
        goal_text=goal.text,
        world_state={"has_codebase": False}  # Precondition not met
    )

    assert plan is None


# === Performance Tests ===

def test_large_plan_conversion():
    """Test converting large HTN plan to TaskGraph."""
    # Create a plan with many tasks
    large_plan_tasks = [
        Task(task_id=f"task_{i}", task_name=f"action_{i}", primitive=True)
        for i in range(100)
    ]

    from src.services.htn_planner import Plan

    large_plan = Plan(
        plan_id="large",
        goal_id="goal_1",
        goal_text="large_goal",
        tasks=large_plan_tasks,
        total_cost=50.0
    )

    task_graph = plan_to_task_graph(large_plan)

    assert len(task_graph.nodes) == 100
    # First task has no dependencies
    assert len(task_graph.nodes["task_0"].dependencies) == 0
    # Last task depends on previous
    assert "task_98" in task_graph.nodes["task_99"].dependencies
