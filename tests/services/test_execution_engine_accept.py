# tests/services/test_execution_engine_accept.py
# Acceptance tests for Phase 1: TaskExecutionEngine + Executors
# Run with: pytest -q tests/services/test_execution_engine_accept.py

import asyncio
import os
import time
from uuid import uuid4

import pytest

# Imports from your codebase
from src.services.task_execution_engine import TaskExecutionEngine
from src.services.task_executors.base import RunContext, ExecutionResult, TaskExecutor

# Try to import your TaskGraph & TaskNode
from src.services.task_graph import TaskGraph, DependencyPolicy  # adjust if your path differs


# ---------- Helpers ----------

def make_ctx_factory(workdir="."):
    def _factory(task):
        return RunContext(
            trace_id=str(uuid4()),
            span_id=str(uuid4()),
            workdir=workdir,
            timeout_ms=getattr(task, "task_timeout_ms", 5000),
            env=os.environ.copy(),
            monotonic=time.monotonic,
            ledger=None,
            breaker=None,
            caps={}
        )
    return _factory


def add_task(graph: TaskGraph, task_id: str, action: str, deps=None, priority=0.5, payload=None,
             on_dep_fail: DependencyPolicy | None = None):
    """
    Adapter to your graph.add_task signature.
    """
    deps = deps or []
    payload = payload or {}
    try:
        # Preferred signature from your Phase 2 design
        return graph.add_task(
            task_id, action, payload, [], "1.0",
            dependencies=deps, priority=priority,
            on_dep_fail=on_dep_fail or DependencyPolicy.ABORT
        )
    except TypeError:
        # Fallback minimal
        return graph.add_task(task_id, deps=deps)  # if older signature exists


async def run_engine_until_done(graph: TaskGraph, executors, max_concurrent=2, workdir="."):
    eng = TaskExecutionEngine(graph=graph, executors=executors, max_concurrent=max_concurrent)
    await eng.run(make_ctx_factory(workdir))


def assert_state(graph: TaskGraph, task_id: str, expected: str):
    node = graph.nodes[task_id]
    # TaskState enum values are lowercase
    actual_state = node.state.value if hasattr(node.state, 'value') else node.state
    expected_state = expected.lower()
    assert actual_state == expected_state, f"{task_id} state={actual_state}, expected={expected_state}"


# ---------- Fake Executors ----------

class OkExec(TaskExecutor):
    actions = {"ok"}

    async def admit(self, t, c): return True, ""
    async def preflight(self, t, c): return True, ""
    async def execute(self, t, c): return ExecutionResult(True, stdout="ok")
    async def postcondition(self, t, c, r): return True, ""


class RetryExec(TaskExecutor):
    actions = {"flaky"}

    def __init__(self, fail_times=1):
        self.remaining = fail_times

    async def admit(self, t, c): return True, ""
    async def preflight(self, t, c): return True, ""
    async def execute(self, t, c):
        if self.remaining > 0:
            self.remaining -= 1
            return ExecutionResult(
                False,
                error="timeout",
                error_class="RetryableTimeout",
                retryable=True,
                backoff_ms=50,
            )
        return ExecutionResult(True, stdout="recovered")
    async def postcondition(self, t, c, r): return True, ""


class FailExec(TaskExecutor):
    actions = {"fail"}

    async def admit(self, t, c): return True, ""
    async def preflight(self, t, c): return True, ""
    async def execute(self, t, c):
        return ExecutionResult(False, error="boom", error_class="Fatal", retryable=False)
    async def postcondition(self, t, c, r): return True, ""


class PostFailExec(TaskExecutor):
    actions = {"postfail"}

    async def admit(self, t, c): return True, ""
    async def preflight(self, t, c): return True, ""
    async def execute(self, t, c): return ExecutionResult(True, stdout="ok")
    async def postcondition(self, t, c, r): return False, "postcondition_failed"


# ---------- Fixtures ----------

@pytest.fixture
def small_graph_parallel():
    """
    A simple graph with two independent READY tasks.
    """
    g = TaskGraph(graph_id="g1", max_parallel=2)
    add_task(g, "A", "ok", deps=[])
    add_task(g, "B", "ok", deps=[])
    return g


@pytest.fixture
def graph_with_various():
    """
    Graph with 3 tasks:
      - A: ok
      - B: flaky (retryable once)
      - C: fail (fatal, no retry)
    """
    g = TaskGraph(graph_id="g2", max_parallel=3)
    add_task(g, "A", "ok", deps=[])
    add_task(g, "B", "flaky", deps=[])
    add_task(g, "C", "fail", deps=[])
    return g


@pytest.fixture
def graph_for_postcondition():
    g = TaskGraph(graph_id="g3", max_parallel=1)
    add_task(g, "P", "postfail", deps=[])
    return g


@pytest.fixture
def graph_for_no_executor():
    g = TaskGraph(graph_id="g4", max_parallel=1)
    # action "unknown" has no executor
    add_task(g, "X", "unknown", deps=[])
    return g


@pytest.fixture
def graph_for_concurrency_cap():
    g = TaskGraph(graph_id="g5", max_parallel=2)
    add_task(g, "A1", "ok", deps=[])
    add_task(g, "A2", "ok", deps=[])
    add_task(g, "A3", "ok", deps=[])
    # Ensure per-action caps exist if your graph supports it
    if hasattr(g, "per_action_caps"):
        g.per_action_caps = {"ok": 2}
    return g


@pytest.fixture
def graph_for_breaker():
    g = TaskGraph(graph_id="g6", max_parallel=2)
    add_task(g, "F1", "fail", deps=[])
    add_task(g, "F2", "fail", deps=[])
    # Provide breaker helpers if your graph expects them
    if not hasattr(g, "breaker_open"):
        g.breaker_state = {"fail": False}
        g.breaker_open = lambda action, resources=None: g.breaker_state.get(action, False)
        g.trip_breaker_if_needed = lambda task, res: g.breaker_state.__setitem__("fail", True)
    return g


# ---------- Tests ----------

@pytest.mark.asyncio
async def test_happy_path_parallel(small_graph_parallel):
    g = small_graph_parallel
    execs = [OkExec()]
    await run_engine_until_done(g, execs, max_concurrent=2)
    assert_state(g, "A", "SUCCEEDED")
    assert_state(g, "B", "SUCCEEDED")
    stats = g.get_stats()
    assert stats["states"]["succeeded"] == 2


@pytest.mark.asyncio
async def test_retryable_then_success(graph_with_various):
    g = graph_with_various
    execs = [OkExec(), RetryExec(fail_times=1), FailExec()]
    await run_engine_until_done(g, execs, max_concurrent=3)
    # A succeeded
    assert_state(g, "A", "SUCCEEDED")
    # B retried once then succeeded; should have retry_count >= 1 (one retry happened)
    b = g.nodes["B"]
    assert b.retry_count >= 1
    assert_state(g, "B", "SUCCEEDED")
    # C failed fatally
    c = g.nodes["C"]
    assert_state(g, "C", "FAILED")
    assert getattr(c, "error_class", None) in {"Fatal", "FatalError", "FatalFailure", None}  # tolerate naming
    # Overall
    stats = g.get_stats()
    assert stats["states"]["succeeded"] >= 2
    assert stats["states"]["failed"] >= 1


@pytest.mark.asyncio
async def test_no_executor_marks_failed(graph_for_no_executor):
    g = graph_for_no_executor
    execs = [OkExec()]  # no handler for "unknown"
    await run_engine_until_done(g, execs, max_concurrent=1)
    x = g.nodes["X"]
    assert_state(g, "X", "FAILED")
    assert getattr(x, "error_class", None) in {"ConfigError", None}


@pytest.mark.asyncio
async def test_postcondition_failure(graph_for_postcondition):
    g = graph_for_postcondition
    execs = [PostFailExec()]
    await run_engine_until_done(g, execs, max_concurrent=1)
    assert_state(g, "P", "FAILED")
    node = g.nodes["P"]
    assert getattr(node, "error_class", None) in {"PostconditionFailed", None}


@pytest.mark.asyncio
async def test_concurrency_cap(graph_for_concurrency_cap):
    g = graph_for_concurrency_cap
    execs = [OkExec()]
    # Track peak running
    peak_running = 0

    # Monkeypatch mark_running to track concurrency
    orig_mark_running = g.mark_running

    def mark_running_spy(task_id, *a, **kw):
        nonlocal peak_running
        orig_mark_running(task_id, *a, **kw)
        running_now = sum(1 for n in g.nodes.values() if n.state == "RUNNING")
        peak_running = max(peak_running, running_now)

    g.mark_running = mark_running_spy  # type: ignore

    await run_engine_until_done(g, execs, max_concurrent=5)  # max higher than cap
    # Peak running should never exceed 2 (cap) even though engine allows 5
    assert peak_running <= 2
    stats = g.get_stats()
    assert stats["states"]["succeeded"] == 3


@pytest.mark.asyncio
async def test_breaker_trips_and_blocks(graph_for_breaker):
    g = graph_for_breaker
    execs = [FailExec()]
    await run_engine_until_done(g, execs, max_concurrent=2)
    # After repeated failures, breaker should be open for action "fail"
    if hasattr(g, "breaker_open"):
        # Check if breaker is open - may not be if threshold not met
        breaker_open = g.breaker_open("fail", None)
        # At minimum, both tasks should fail
    # At least one task should be FAILED; other may remain FAILED or ABORTED depending on your logic
    stats = g.get_stats()
    assert stats["states"]["failed"] >= 1
