"""Integration test: HTN Planning → TaskGraph → Execution Engine.

Tests the full flow from goal decomposition through task execution.
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path

from src.services.htn_planner import HTNPlanner, Method, Task, Plan
from src.services.task_graph import TaskGraph, TaskNode
from src.services.task_execution_engine import TaskExecutionEngine
from src.services.task_executors.base import RunContext, ExecutionResult, TaskExecutor


# === Mock Executor for Integration Test ===

class MockFileCreator(TaskExecutor):
    """Mock executor that simulates file creation."""
    actions = {"create_file", "write_file"}

    def __init__(self):
        self.files_created = []

    async def admit(self, task, ctx):
        return True, ""

    async def preflight(self, task, ctx):
        return True, ""

    async def execute(self, task, ctx):
        args = task.normalized_args
        filepath = args.get("path", "test.txt")
        content = args.get("content", "")

        # Simulate file creation
        self.files_created.append(filepath)
        await asyncio.sleep(0.01)  # Simulate I/O

        return ExecutionResult(
            success=True,
            stdout=f"Created {filepath}",
            artifacts={"path": filepath, "size": len(content)}
        )

    async def postcondition(self, task, ctx, res):
        return True, ""


class MockTestRunner(TaskExecutor):
    """Mock executor that simulates running tests."""
    actions = {"run_tests", "verify"}

    async def admit(self, task, ctx):
        return True, ""

    async def preflight(self, task, ctx):
        return True, ""

    async def execute(self, task, ctx):
        await asyncio.sleep(0.01)  # Simulate test run

        return ExecutionResult(
            success=True,
            stdout="All tests passed",
            artifacts={"tests_run": 10, "tests_passed": 10}
        )

    async def postcondition(self, task, ctx, res):
        return True, ""


# === Helper: Convert HTN Plan to TaskGraph ===

def plan_to_taskgraph(plan: Plan, graph_id: str = "integration-test", sequential: bool = True) -> TaskGraph:
    """Convert HTN Plan to TaskGraph.

    Args:
        plan: HTN Plan to convert
        graph_id: Graph identifier
        sequential: If True, creates dependency chain; if False, tasks run in parallel

    Creates a sequential dependency chain from the ordered task list (if sequential=True).
    """
    graph = TaskGraph(
        graph_id=graph_id,
        max_retry_tokens=5,
        graph_timeout_ms=300000  # 5 minutes
    )

    prev_task_id = None

    for idx, task in enumerate(plan.tasks):
        task_id = task.task_id
        action_name = task.task_name  # Use task_name directly as action name

        # Build dependencies: each task depends on the previous one (if sequential)
        deps = [prev_task_id] if (prev_task_id and sequential) else []

        # Ensure parameters dict exists
        params = task.parameters if task.parameters else {}

        graph.add_task(
            task_id=task_id,
            action_name=action_name,
            normalized_args=params,
            resource_ids=[],
            version="1.0",
            dependencies=deps,
            priority=0.5,
            max_retries=2
        )

        prev_task_id = task_id

    return graph


# === Integration Test ===

@pytest.mark.asyncio
async def test_full_flow_htn_to_execution():
    """Integration test: Goal → HTN Plan → TaskGraph → Execution → Completion.

    Flow:
    1. Define decomposition methods
    2. Create HTN planner and generate plan
    3. Convert plan to TaskGraph
    4. Execute tasks with TaskExecutionEngine
    5. Verify all tasks completed successfully
    """

    # Step 1: Define HTN decomposition methods
    # Use action names directly as subtask names for simplicity
    methods = [
        Method(
            name="implement_feature_simple",
            task="implement_feature",
            preconditions=[],
            subtasks=["create_file", "create_file", "run_tests"],  # Sequential tasks
            cost=0.5
        )
    ]

    # Define which task names are primitive (executable)
    # These match the executor action names
    primitive_task_names = {
        "create_file",
        "run_tests"
    }

    # Step 2: Create HTN planner and generate plan
    planner = HTNPlanner(
        belief_kernel=None,  # Not needed for this simple test
        method_library=methods,
        primitive_tasks=primitive_task_names
    )

    # Use the goal task name that matches our method
    plan = planner.plan(
        goal_id="goal-1",
        goal_text="implement_feature",  # This matches the method.task
        world_state={},
        constraints=[]
    )

    assert plan is not None, "HTN planner should generate a plan"
    assert len(plan.tasks) == 3, "Plan should have 3 primitive tasks (2 create_file, 1 run_tests)"
    assert all(task.primitive for task in plan.tasks), "All tasks should be primitive"

    # Step 3: Convert plan to TaskGraph
    graph = plan_to_taskgraph(plan, graph_id="integration-test", sequential=True)

    assert len(graph.nodes) == 3, "TaskGraph should have 3 nodes"

    # Verify dependency chain - tasks have auto-generated IDs like "goal-1.0", "goal-1.1", "goal-1.2"
    task_ids = list(graph.nodes.keys())
    first_task = graph.nodes[task_ids[0]]
    second_task = graph.nodes[task_ids[1]]
    third_task = graph.nodes[task_ids[2]]

    assert len(first_task.dependencies) == 0, "First task has no dependencies"
    assert task_ids[0] in second_task.dependencies, "Second task depends on first"
    assert task_ids[1] in third_task.dependencies, "Third task depends on second"

    # Step 4: Set up executors and run execution engine
    file_creator = MockFileCreator()
    test_runner = MockTestRunner()
    executors = [file_creator, test_runner]

    engine = TaskExecutionEngine(
        graph=graph,
        executors=executors,
        max_concurrent=2
    )

    # Create RunContext factory
    workdir = tempfile.gettempdir()

    def make_context(task: TaskNode) -> RunContext:
        return RunContext(
            trace_id=f"trace-{task.task_id}",
            span_id=f"span-{task.task_id}",
            workdir=workdir,
            timeout_ms=10000,
            env={},
            monotonic=time.monotonic,
            ledger=None,
            breaker=graph,
            caps={}
        )

    # Run the engine
    await engine.run(make_context)

    # Step 5: Verify completion
    stats = graph.get_stats()

    assert stats["states"]["succeeded"] == 3, "All 3 tasks should succeed"
    assert stats["states"].get("failed", 0) == 0, "No tasks should fail"

    # Verify all tasks succeeded
    for task_id, node in graph.nodes.items():
        assert node.state.value == "succeeded", f"Task {task_id} should succeed"

    # Verify executors were called
    # First two tasks are create_file, third is run_tests
    assert len(file_creator.files_created) == 2, f"Should create 2 files, got {len(file_creator.files_created)}"

    print("✅ Full integration test passed: HTN → TaskGraph → Execution → Success")


@pytest.mark.asyncio
async def test_parallel_execution_from_htn():
    """Test that independent HTN tasks execute in parallel."""

    # Define method with parallel subtasks - all are create_file actions
    methods = [
        Method(
            name="parallel_setup",
            task="setup_environment",
            preconditions=[],
            subtasks=["create_file", "create_file", "create_file"],  # Three parallel file creates
            cost=0.3
        )
    ]

    primitive_task_names = {"create_file"}

    planner = HTNPlanner(
        method_library=methods,
        primitive_tasks=primitive_task_names
    )

    plan = planner.plan(
        goal_id="goal-parallel",
        goal_text="setup_environment",  # Matches method.task
        world_state={}
    )

    assert plan is not None
    assert len(plan.tasks) == 3, "Should have 3 parallel tasks"

    # Convert to graph with NO dependencies (parallel execution)
    graph = plan_to_taskgraph(plan, graph_id="parallel-test", sequential=False)

    # Track execution timing
    execution_times = {}

    class TimedFileCreator(TaskExecutor):
        actions = {"create_file"}

        async def admit(self, task, ctx):
            return True, ""

        async def preflight(self, task, ctx):
            return True, ""

        async def execute(self, task, ctx):
            start = time.time()
            await asyncio.sleep(0.05)  # Simulate work
            execution_times[task.task_id] = time.time() - start
            return ExecutionResult(success=True, stdout="done")

        async def postcondition(self, task, ctx, res):
            return True, ""

    engine = TaskExecutionEngine(
        graph=graph,
        executors=[TimedFileCreator()],
        max_concurrent=10  # Allow parallel execution
    )

    workdir = tempfile.gettempdir()

    def make_context(task: TaskNode) -> RunContext:
        return RunContext(
            trace_id=f"trace-{task.task_id}",
            span_id=f"span-{task.task_id}",
            workdir=workdir,
            timeout_ms=10000,
            env={},
            monotonic=time.monotonic,
            ledger=None,
            breaker=graph,
            caps={}
        )

    start_time = time.time()
    await engine.run(make_context)
    total_time = time.time() - start_time

    # Verify all tasks succeeded
    stats = graph.get_stats()
    assert stats["states"]["succeeded"] == 3

    # Verify parallel execution: total time should be ~0.05s, not ~0.15s
    assert total_time < 0.15, f"Parallel execution should take <0.15s, took {total_time:.3f}s"

    print(f"✅ Parallel execution test passed: {total_time:.3f}s for 3 tasks")
