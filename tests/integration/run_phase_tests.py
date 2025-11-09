#!/usr/bin/env python3
"""Standalone test runner for autonomous coding pipeline.

Runs all phase tests without requiring pytest.
"""
import asyncio
import sys
import time
import traceback
from pathlib import Path

# Add project to path
sys.path.insert(0, "/home/d/git/ai-exp")

from src.services.task_graph import TaskGraph, TaskState
from src.services.task_execution_engine import TaskExecutionEngine
from src.services.task_executors.base import RunContext
from src.services.task_executors.code_modification import CodeModificationExecutor
from src.services.task_executors.test_runner import TestExecutor
from src.services.task_executors.shell_command import ShellCommandExecutor
from src.services.code_access import create_code_access_service
from src.services.goal_execution_service import GoalExecutionService
from src.services.goal_store import (
    create_goal_store,
    GoalDefinition,
    GoalCategory,
    GoalState,
    GoalSource
)
from uuid import uuid4


class TestRunner:
    """Test runner with pass/fail tracking."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, name, test_func):
        """Run a single test function."""
        print(f"\n  Testing: {name}...", end=" ")
        try:
            if asyncio.iscoroutinefunction(test_func):
                asyncio.run(test_func())
            else:
                test_func()
            print("✓ PASS")
            self.passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {e}")
            self.failed += 1
            self.errors.append((name, str(e)))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            self.failed += 1
            self.errors.append((name, traceback.format_exc()))

    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"Results: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailures:")
            for name, error in self.errors:
                print(f"  - {name}")
                print(f"    {error[:200]}")
        return self.failed == 0


# Test fixtures
project_root = Path("/home/d/git/ai-exp")
code_access = create_code_access_service(
    project_root=project_root,
    max_file_size_kb=100,
    auto_branch=False
)
executors = [
    CodeModificationExecutor(code_access),
    TestExecutor(),
    ShellCommandExecutor()
]


def make_context(task):
    """Create RunContext for task."""
    return RunContext(
        trace_id=f"test-{uuid4().hex[:8]}",
        span_id=f"task-{task.task_id}",
        workdir=str(project_root),
        timeout_ms=task.task_timeout_ms or 30000,
        env={},
        monotonic=time.monotonic,
        ledger=None,
        breaker=None,
        caps={}
    )


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 TESTS: TaskExecutionEngine + Executors
# ═══════════════════════════════════════════════════════════════════

async def test_phase1_single_file_creation():
    """Test creating a single file."""
    graph = TaskGraph(graph_id="test-single", max_retry_tokens=5, graph_timeout_ms=30000)

    graph.add_task(
        task_id="create_test",
        action_name="create_file",
        normalized_args={
            "file_path": "tests/generated/phase1_single.py",
            "content": "# Test\ndef test(): return True\n",
            "reason": "Test"
        },
        resource_ids=[], version="1.0", dependencies=[], priority=0.5, max_retries=2
    )

    engine = TaskExecutionEngine(graph=graph, executors=executors, max_concurrent=1)
    await engine.run(make_context)

    stats = graph.get_stats()
    assert stats['states']['succeeded'] == 1, f"Expected 1 succeeded, got {stats['states']}"
    assert Path("tests/generated/phase1_single.py").exists()


async def test_phase1_sequential_tasks():
    """Test sequential task execution."""
    graph = TaskGraph(graph_id="test-seq", max_retry_tokens=10, graph_timeout_ms=60000)

    graph.add_task(
        task_id="task1", action_name="create_file",
        normalized_args={"file_path": "tests/generated/phase1_seq1.py", "content": "# 1\n", "reason": "T"},
        resource_ids=[], version="1.0", dependencies=[], priority=0.5, max_retries=2
    )
    graph.add_task(
        task_id="task2", action_name="create_file",
        normalized_args={"file_path": "tests/generated/phase1_seq2.py", "content": "# 2\n", "reason": "T"},
        resource_ids=[], version="1.0", dependencies=["task1"], priority=0.5, max_retries=2
    )

    engine = TaskExecutionEngine(graph=graph, executors=executors, max_concurrent=2)
    await engine.run(make_context)

    stats = graph.get_stats()
    assert stats['states']['succeeded'] == 2


async def test_phase1_shell_executor():
    """Test shell command execution."""
    graph = TaskGraph(graph_id="test-shell", max_retry_tokens=5, graph_timeout_ms=30000)

    graph.add_task(
        task_id="echo", action_name="shell_command",
        normalized_args={"cmd": "echo test"},  # ShellCommandExecutor expects string
        resource_ids=[], version="1.0", dependencies=[], priority=0.5, max_retries=2
    )

    engine = TaskExecutionEngine(graph=graph, executors=executors, max_concurrent=1)
    await engine.run(make_context)

    stats = graph.get_stats()
    assert stats['states']['succeeded'] == 1


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 TESTS: GoalExecutionService (HTN Planning)
# ═══════════════════════════════════════════════════════════════════

async def test_phase2_execute_implement_feature():
    """Test executing implement_feature goal."""
    service = GoalExecutionService(
        code_access=code_access,
        workdir=str(project_root),
        max_concurrent=2
    )

    result = await service.execute_goal(
        goal_text="implement_feature",
        context={},
        timeout_ms=60000
    )

    assert result.goal_text == "implement_feature"
    assert result.total_tasks >= 3
    assert len(result.completed_tasks) >= 2


async def test_phase2_htn_methods():
    """Test HTN methods are used."""
    service = GoalExecutionService(
        code_access=code_access,
        workdir=str(project_root),
        max_concurrent=2
    )

    result = await service.execute_goal(
        goal_text="fix_bug",
        context={},
        timeout_ms=60000
    )

    assert "fix_bug_simple" in result.methods_used


async def test_phase2_file_paths():
    """Test file paths are in allowed directory."""
    service = GoalExecutionService(
        code_access=code_access,
        workdir=str(project_root),
        max_concurrent=2
    )

    result = await service.execute_goal(
        goal_text="implement_feature",
        context={},
        timeout_ms=60000
    )

    for task in result.completed_tasks:
        if "file_path" in task.artifacts:
            path = task.artifacts["file_path"]
            assert path.startswith("tests/generated/"), f"Invalid path: {path}"


# ═══════════════════════════════════════════════════════════════════
# PHASE 3 TESTS: GoalStore Integration
# ═══════════════════════════════════════════════════════════════════

def test_phase3_create_goal():
    """Test creating a goal."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    goal_store = create_goal_store(db_path)

    goal = GoalDefinition(
        id=str(uuid4()),
        text="implement_feature",
        category=GoalCategory.USER_REQUESTED,
        value=0.8, effort=0.3, risk=0.2,
        horizon_min_min=0
    )

    created = goal_store.create_goal(goal)
    assert created.state == GoalState.PROPOSED

    goal_store.close()
    Path(db_path).unlink(missing_ok=True)


def test_phase3_adopt_goal():
    """Test adopting a goal."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    goal_store = create_goal_store(db_path)

    goal = GoalDefinition(
        id=str(uuid4()),
        text="implement_feature",
        category=GoalCategory.USER_REQUESTED,
        value=0.8, effort=0.3, risk=0.2,
        horizon_min_min=0
    )

    goal_store.create_goal(goal)
    adopted, adopted_goal, _ = goal_store.adopt_goal(goal.id)

    assert adopted is True
    assert adopted_goal.state == GoalState.ADOPTED

    goal_store.close()
    Path(db_path).unlink(missing_ok=True)


async def test_phase3_execute_goal():
    """Test executing goal via GoalStore."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    goal_store = create_goal_store(db_path)

    goal = GoalDefinition(
        id=str(uuid4()),
        text="add_tests",
        category=GoalCategory.USER_REQUESTED,
        value=0.5, effort=0.5, risk=0.5,
        horizon_min_min=0
    )

    goal_store.create_goal(goal)
    goal_store.adopt_goal(goal.id)

    result = await goal_store.execute_goal(
        goal_id=goal.id,
        code_access_service=code_access,
        timeout_ms=60000
    )

    assert result.total_tasks >= 2

    # Check metadata
    final_goal = goal_store.get_goal(goal.id)
    assert "execution_result" in final_goal.metadata

    goal_store.close()
    Path(db_path).unlink(missing_ok=True)


async def test_phase3_state_transitions():
    """Test goal state transitions."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    goal_store = create_goal_store(db_path)

    goal = GoalDefinition(
        id=str(uuid4()),
        text="implement_feature",
        category=GoalCategory.USER_REQUESTED,
        value=0.8, effort=0.3, risk=0.2,
        horizon_min_min=0
    )

    # PROPOSED
    created = goal_store.create_goal(goal)
    assert created.state == GoalState.PROPOSED

    # ADOPTED
    adopted, adopted_goal, _ = goal_store.adopt_goal(goal.id)
    assert adopted_goal.state == GoalState.ADOPTED

    # EXECUTING → SATISFIED/ADOPTED
    await goal_store.execute_goal(
        goal_id=goal.id,
        code_access_service=code_access,
        timeout_ms=60000
    )

    final = goal_store.get_goal(goal.id)
    assert final.state in [GoalState.SATISFIED, GoalState.ADOPTED]

    goal_store.close()
    Path(db_path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("AUTONOMOUS CODING PIPELINE - TEST BATTERY")
    print("="*70)

    # Phase 1
    print("\n" + "─"*70)
    print("PHASE 1: TaskExecutionEngine + Executors")
    print("─"*70)
    runner1 = TestRunner()
    runner1.run_test("Single file creation", test_phase1_single_file_creation)
    runner1.run_test("Sequential tasks", test_phase1_sequential_tasks)
    runner1.run_test("Shell executor", test_phase1_shell_executor)
    phase1_ok = runner1.summary()

    # Phase 2
    print("\n" + "─"*70)
    print("PHASE 2: GoalExecutionService (HTN Planning)")
    print("─"*70)
    runner2 = TestRunner()
    runner2.run_test("Execute implement_feature", test_phase2_execute_implement_feature)
    runner2.run_test("HTN methods", test_phase2_htn_methods)
    runner2.run_test("File paths", test_phase2_file_paths)
    phase2_ok = runner2.summary()

    # Phase 3
    print("\n" + "─"*70)
    print("PHASE 3: GoalStore Integration")
    print("─"*70)
    runner3 = TestRunner()
    runner3.run_test("Create goal", test_phase3_create_goal)
    runner3.run_test("Adopt goal", test_phase3_adopt_goal)
    runner3.run_test("Execute goal", test_phase3_execute_goal)
    runner3.run_test("State transitions", test_phase3_state_transitions)
    phase3_ok = runner3.summary()

    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)

    total_pass = runner1.passed + runner2.passed + runner3.passed
    total_fail = runner1.failed + runner2.failed + runner3.failed

    print(f"Phase 1: {runner1.passed} passed, {runner1.failed} failed")
    print(f"Phase 2: {runner2.passed} passed, {runner2.failed} failed")
    print(f"Phase 3: {runner3.passed} passed, {runner3.failed} failed")
    print(f"\nTOTAL: {total_pass} passed, {total_fail} failed")

    if phase1_ok and phase2_ok and phase3_ok:
        print("\n✓ ALL TESTS PASSED - AUTONOMOUS CODING PIPELINE VERIFIED")
        print("="*70)
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
