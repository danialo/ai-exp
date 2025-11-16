"""Unit tests for HTN Planner."""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from src.services.htn_planner import (
    Method,
    Task,
    Plan,
    HTNPlanner,
    register_htn_decision
)


# === Fixture Data ===

@pytest.fixture
def simple_methods():
    """Create simple test methods for basic planning."""
    return [
        Method(
            name="improve_quality_basic",
            task="improve_code_quality",
            preconditions=["has_codebase"],
            subtasks=["run_linter", "fix_issues", "run_tests"],
            cost=0.5
        ),
        Method(
            name="run_tests_basic",
            task="run_tests",
            preconditions=[],
            subtasks=["setup_env", "execute_tests", "report_results"],
            cost=0.3
        )
    ]


@pytest.fixture
def complex_methods():
    """Create methods with multiple decomposition paths."""
    return [
        # High-cost method for improving quality
        Method(
            name="improve_quality_thorough",
            task="improve_code_quality",
            preconditions=["has_codebase"],
            subtasks=["analyze_code", "refactor", "run_tests", "document"],
            cost=0.9
        ),
        # Low-cost method (should be preferred)
        Method(
            name="improve_quality_quick",
            task="improve_code_quality",
            preconditions=["has_codebase"],
            subtasks=["run_linter", "fix_critical"],
            cost=0.3
        ),
        # Method with unsatisfied precondition
        Method(
            name="improve_quality_with_ci",
            task="improve_code_quality",
            preconditions=["has_ci_pipeline"],  # Not satisfied
            subtasks=["run_ci", "review_results"],
            cost=0.2
        )
    ]


@pytest.fixture
def primitive_tasks():
    """Define which tasks are primitive (executable)."""
    return {
        "run_linter",
        "fix_issues",
        "fix_critical",
        "analyze_code",
        "refactor",
        "document",
        "setup_env",
        "execute_tests",
        "report_results",
        "run_ci",
        "review_results"
    }


@pytest.fixture
def mock_belief_kernel():
    """Create mock BeliefKernel for testing precondition checking."""
    kernel = Mock()

    # Define belief responses
    def get_belief(belief_id):
        beliefs = {
            "has_codebase": Mock(confidence=0.9),
            "has_ci_pipeline": Mock(confidence=0.3),  # Below threshold
            "belief_quality": Mock(confidence=0.8)
        }
        return beliefs.get(belief_id, None)

    kernel.get_belief = get_belief
    return kernel


# === Data Structure Tests ===

def test_method_creation():
    """Test Method dataclass creation and validation."""
    method = Method(
        name="test_method",
        task="test_task",
        preconditions=["precond1"],
        subtasks=["subtask1", "subtask2"],
        cost=0.5
    )

    assert method.name == "test_method"
    assert method.task == "test_task"
    assert method.preconditions == ["precond1"]
    assert method.subtasks == ["subtask1", "subtask2"]
    assert method.cost == 0.5


def test_method_validation():
    """Test Method validation catches invalid parameters."""
    # Empty name
    with pytest.raises(ValueError, match="name cannot be empty"):
        Method(name="", task="test", subtasks=["sub1"])

    # Empty task
    with pytest.raises(ValueError, match="task cannot be empty"):
        Method(name="test", task="", subtasks=["sub1"])

    # Invalid cost
    with pytest.raises(ValueError, match="cost must be in"):
        Method(name="test", task="test", subtasks=["sub1"], cost=1.5)

    # No subtasks
    with pytest.raises(ValueError, match="at least one subtask"):
        Method(name="test", task="test", subtasks=[])


def test_task_creation():
    """Test Task dataclass creation and validation."""
    task = Task(
        task_id="task_1",
        task_name="test_task",
        primitive=True,
        parameters={"param1": "value1"}
    )

    assert task.task_id == "task_1"
    assert task.task_name == "test_task"
    assert task.primitive is True
    assert task.parameters == {"param1": "value1"}


def test_task_validation():
    """Test Task validation catches invalid parameters."""
    # Empty task_id
    with pytest.raises(ValueError, match="Task ID cannot be empty"):
        Task(task_id="", task_name="test")

    # Empty task_name
    with pytest.raises(ValueError, match="Task name cannot be empty"):
        Task(task_id="test", task_name="")


def test_plan_creation():
    """Test Plan dataclass creation."""
    tasks = [
        Task(task_id="t1", task_name="task1", primitive=True),
        Task(task_id="t2", task_name="task2", primitive=True)
    ]

    plan = Plan(
        plan_id="plan_1",
        goal_id="goal_1",
        goal_text="Test goal",
        tasks=tasks,
        total_cost=0.5,
        methods_used=["method1"]
    )

    assert plan.plan_id == "plan_1"
    assert plan.goal_id == "goal_1"
    assert plan.goal_text == "Test goal"
    assert len(plan.tasks) == 2
    assert plan.total_cost == 0.5
    assert plan.methods_used == ["method1"]
    assert isinstance(plan.created_at, datetime)


# === HTNPlanner Basic Tests ===

def test_planner_initialization(simple_methods, primitive_tasks):
    """Test HTNPlanner initialization."""
    planner = HTNPlanner(
        belief_kernel=None,
        method_library=simple_methods,
        primitive_tasks=primitive_tasks
    )

    assert len(planner.methods) == 2
    assert len(planner.primitive_tasks) == len(primitive_tasks)
    assert "run_linter" in planner.primitive_tasks


def test_add_method(primitive_tasks):
    """Test adding methods dynamically."""
    planner = HTNPlanner(primitive_tasks=primitive_tasks)

    assert len(planner.methods) == 0

    method = Method(
        name="new_method",
        task="new_task",
        subtasks=["sub1"]
    )
    planner.add_method(method)

    assert len(planner.methods) == 1
    assert "new_task" in planner._method_index


def test_add_primitive_task():
    """Test adding primitive tasks dynamically."""
    planner = HTNPlanner()

    assert len(planner.primitive_tasks) == 0

    planner.add_primitive_task("new_task")

    assert "new_task" in planner.primitive_tasks
    assert planner.is_primitive("new_task") is True


def test_is_primitive(primitive_tasks):
    """Test primitive task checking."""
    planner = HTNPlanner(primitive_tasks=primitive_tasks)

    assert planner.is_primitive("run_linter") is True
    assert planner.is_primitive("fix_issues") is True
    assert planner.is_primitive("improve_code_quality") is False


# === Planning Tests ===

def test_simple_plan_single_decomposition(simple_methods, primitive_tasks):
    """Test planning with single-level decomposition."""
    planner = HTNPlanner(
        method_library=simple_methods,
        primitive_tasks=primitive_tasks
    )

    # Goal: improve_code_quality
    # Should decompose to: run_linter, fix_issues, run_tests
    # run_tests further decomposes to: setup_env, execute_tests, report_results
    plan = planner.plan(
        goal_id="goal_1",
        goal_text="improve_code_quality",
        world_state={"has_codebase": True}
    )

    assert plan is not None
    assert len(plan.tasks) == 5  # run_linter, fix_issues, setup_env, execute_tests, report_results
    assert plan.tasks[0].task_name == "run_linter"
    assert plan.tasks[1].task_name == "fix_issues"
    assert plan.tasks[2].task_name == "setup_env"
    assert plan.tasks[3].task_name == "execute_tests"
    assert plan.tasks[4].task_name == "report_results"
    assert plan.total_cost == 0.8  # 0.5 + 0.3
    assert "improve_quality_basic" in plan.methods_used
    assert "run_tests_basic" in plan.methods_used


def test_primitive_goal_no_decomposition(primitive_tasks):
    """Test planning when goal is already primitive."""
    planner = HTNPlanner(primitive_tasks=primitive_tasks)

    # Goal is primitive - no decomposition needed
    plan = planner.plan(
        goal_id="goal_1",
        goal_text="run_linter",
        world_state={}
    )

    assert plan is not None
    assert len(plan.tasks) == 1
    assert plan.tasks[0].task_name == "run_linter"
    assert plan.tasks[0].primitive is True
    assert plan.total_cost == 0.0  # No methods used
    assert len(plan.methods_used) == 0


def test_cost_based_method_selection(complex_methods, primitive_tasks):
    """Test planner selects lowest-cost applicable method."""
    planner = HTNPlanner(
        method_library=complex_methods,
        primitive_tasks=primitive_tasks
    )

    # Three methods available, but improve_quality_quick has lowest cost
    plan = planner.plan(
        goal_id="goal_1",
        goal_text="improve_code_quality",
        world_state={"has_codebase": True}
    )

    assert plan is not None
    assert "improve_quality_quick" in plan.methods_used
    assert "improve_quality_thorough" not in plan.methods_used
    assert plan.total_cost == 0.3  # Lowest cost method


def test_no_applicable_methods(simple_methods, primitive_tasks):
    """Test planning fails when no methods applicable."""
    planner = HTNPlanner(
        method_library=simple_methods,
        primitive_tasks=primitive_tasks
    )

    # Precondition not satisfied
    plan = planner.plan(
        goal_id="goal_1",
        goal_text="improve_code_quality",
        world_state={"has_codebase": False}  # Precondition not met
    )

    assert plan is None


def test_unknown_task_no_methods(primitive_tasks):
    """Test planning fails for unknown compound task."""
    planner = HTNPlanner(primitive_tasks=primitive_tasks)

    # No methods for "unknown_task"
    plan = planner.plan(
        goal_id="goal_1",
        goal_text="unknown_task",
        world_state={}
    )

    assert plan is None


def test_constraint_satisfaction(simple_methods, primitive_tasks):
    """Test plan constraint checking."""
    planner = HTNPlanner(
        method_library=simple_methods,
        primitive_tasks=primitive_tasks
    )

    def max_three_tasks(tasks):
        """Constraint: plan must have at most 3 tasks."""
        return len(tasks) <= 3

    # Plan will have 6 tasks - should fail constraint
    plan = planner.plan(
        goal_id="goal_1",
        goal_text="improve_code_quality",
        world_state={"has_codebase": True},
        constraints=[max_three_tasks]
    )

    assert plan is None


def test_constraint_success(primitive_tasks):
    """Test plan passes constraint."""
    planner = HTNPlanner(primitive_tasks=primitive_tasks)

    def any_tasks(tasks):
        """Constraint: plan must have at least 1 task."""
        return len(tasks) >= 1

    plan = planner.plan(
        goal_id="goal_1",
        goal_text="run_linter",
        world_state={},
        constraints=[any_tasks]
    )

    assert plan is not None
    assert "any_tasks" in plan.constraints_satisfied


def test_max_depth_protection():
    """Test planning stops at max depth to prevent infinite recursion."""
    # Create circular decomposition
    methods = [
        Method(
            name="loop1",
            task="task_a",
            subtasks=["task_b"],
            cost=0.1
        ),
        Method(
            name="loop2",
            task="task_b",
            subtasks=["task_a"],
            cost=0.1
        )
    ]

    planner = HTNPlanner(method_library=methods, primitive_tasks=set())

    # Should hit max_depth and return None
    plan = planner.plan(
        goal_id="goal_1",
        goal_text="task_a",
        world_state={},
        max_depth=10
    )

    assert plan is None


# === Precondition Tests ===

def test_world_state_preconditions(simple_methods, primitive_tasks):
    """Test precondition checking via world_state."""
    planner = HTNPlanner(
        method_library=simple_methods,
        primitive_tasks=primitive_tasks
    )

    # Precondition satisfied
    plan = planner.plan(
        goal_id="goal_1",
        goal_text="improve_code_quality",
        world_state={"has_codebase": True}
    )
    assert plan is not None

    # Precondition not satisfied (False)
    plan = planner.plan(
        goal_id="goal_2",
        goal_text="improve_code_quality",
        world_state={"has_codebase": False}
    )
    assert plan is None

    # Precondition not satisfied (missing)
    plan = planner.plan(
        goal_id="goal_3",
        goal_text="improve_code_quality",
        world_state={}
    )
    assert plan is None


def test_belief_kernel_preconditions(simple_methods, primitive_tasks, mock_belief_kernel):
    """Test precondition checking via BeliefKernel."""
    planner = HTNPlanner(
        belief_kernel=mock_belief_kernel,
        method_library=simple_methods,
        primitive_tasks=primitive_tasks
    )

    # Belief "has_codebase" has confidence 0.9 (>= 0.5) - satisfied
    plan = planner.plan(
        goal_id="goal_1",
        goal_text="improve_code_quality",
        world_state={}
    )
    assert plan is not None


def test_belief_kernel_precondition_low_confidence(complex_methods, primitive_tasks, mock_belief_kernel):
    """Test precondition fails when belief confidence too low."""
    planner = HTNPlanner(
        belief_kernel=mock_belief_kernel,
        method_library=complex_methods,
        primitive_tasks=primitive_tasks
    )

    # Belief "has_ci_pipeline" has confidence 0.3 (< 0.5) - not satisfied
    # Only method requiring CI should be filtered out
    plan = planner.plan(
        goal_id="goal_1",
        goal_text="improve_code_quality",
        world_state={}
    )

    assert plan is not None
    # Should use quick method (cost 0.3) since CI method excluded
    assert plan.total_cost == 0.3


def test_preconditions_satisfied_mixed(primitive_tasks, mock_belief_kernel):
    """Test precondition checking with both beliefs and world_state."""
    method = Method(
        name="mixed_precond",
        task="test_task",
        preconditions=["has_codebase", "has_tests"],  # One belief, one world_state
        subtasks=["subtask1"],
        cost=0.5
    )

    # Add subtask1 as primitive
    primitives = primitive_tasks | {"subtask1"}

    planner = HTNPlanner(
        belief_kernel=mock_belief_kernel,
        method_library=[method],
        primitive_tasks=primitives
    )

    # Belief satisfied, world_state satisfied
    plan = planner.plan(
        goal_id="goal_1",
        goal_text="test_task",
        world_state={"has_tests": True}
    )
    assert plan is not None

    # Belief satisfied, world_state not satisfied
    plan = planner.plan(
        goal_id="goal_2",
        goal_text="test_task",
        world_state={"has_tests": False}
    )
    assert plan is None


# === Edge Cases ===

def test_empty_method_library():
    """Test planner with no methods."""
    planner = HTNPlanner(method_library=[], primitive_tasks={"task1"})

    # Primitive task works
    plan = planner.plan(goal_id="goal_1", goal_text="task1", world_state={})
    assert plan is not None

    # Compound task fails
    plan = planner.plan(goal_id="goal_2", goal_text="compound", world_state={})
    assert plan is None


def test_task_metadata_preserved():
    """Test that task metadata is preserved during decomposition."""
    method = Method(
        name="test_method",
        task="compound_task",
        subtasks=["primitive1", "primitive2"],
        cost=0.5
    )

    planner = HTNPlanner(
        method_library=[method],
        primitive_tasks={"primitive1", "primitive2"}
    )

    plan = planner.plan(
        goal_id="goal_1",
        goal_text="compound_task",
        world_state={}
    )

    assert plan is not None
    # Check that subtasks have parent metadata
    for task in plan.tasks:
        assert "method" in task.metadata
        assert task.metadata["method"] == "test_method"


# === Integration Tests ===

def test_register_htn_decision():
    """Test HTN decision point registration."""
    import tempfile
    from src.services.decision_framework import DecisionRegistry

    # Create temporary registry
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        registry = DecisionRegistry(db_path=f.name)

        # Should not raise
        register_htn_decision(registry)

        # Verify parameters can be retrieved
        params = registry.get_all_parameters("plan_generated")
        assert params is not None
        assert "cost_weight" in params
        assert "precondition_threshold" in params
        assert params["cost_weight"] == 0.7
