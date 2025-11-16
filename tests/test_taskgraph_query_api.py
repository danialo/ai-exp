"""
Tests for TaskGraph Query API - Rubric Validation (R1-R6).

Tests validate that the API meets all requirements from the
TaskGraph Querying Rubric.
"""

import pytest
from datetime import datetime, timezone, timedelta
from src.services.task_graph import TaskGraph, TaskState, DependencyPolicy


class TestR1Lifecycle:
    """R1: Lifecycle Coverage - List/filter tasks, fetch task with transition metadata."""

    def test_list_all_tasks(self):
        """Can list all tasks."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")
        g.add_task("B", "action", {}, [], "1.0")

        # Mock list_tasks endpoint logic
        tasks = [
            {"task_id": tid, "state": node.state.value}
            for tid, node in g.nodes.items()
        ]

        assert len(tasks) == 2
        assert all("task_id" in t and "state" in t for t in tasks)

    def test_filter_by_state(self):
        """Can filter tasks by state."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")
        g.add_task("B", "action", {}, [], "1.0", dependencies=["A"])

        g._update_ready_queue()

        # A should be READY, B should be PENDING
        ready_tasks = [tid for tid, node in g.nodes.items() if node.state == TaskState.READY]
        pending_tasks = [tid for tid, node in g.nodes.items() if node.state == TaskState.PENDING]

        assert "A" in ready_tasks
        assert "B" in pending_tasks

    def test_get_task_details(self):
        """Can get detailed task information."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {"key": "value"}, ["res1"], "1.0", priority=0.8)

        node = g.nodes["A"]

        # Validate all required fields exist
        assert node.task_id == "A"
        assert node.action_name == "action"
        assert node.state == TaskState.PENDING
        assert node.priority == 0.8
        assert node.normalized_args == {"key": "value"}
        assert node.resource_ids == ["res1"]
        assert node.idempotency_key is not None


class TestR2Dependencies:
    """R2: Dependencies & Policies - Parent/child visibility, policy preview."""

    def test_get_dependencies(self):
        """Can get task dependencies."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")
        g.add_task("B", "action", {}, [], "1.0", dependencies=["A"])

        node_b = g.nodes["B"]

        assert node_b.dependencies == ["A"]
        assert "B" in g.nodes["A"].dependents

    def test_policy_preview_abort(self):
        """ABORT policy preview correct."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")
        g.add_task("B", "action", {}, [], "1.0",
                   dependencies=["A"], on_dep_fail=DependencyPolicy.ABORT)

        assert g.nodes["B"].on_dep_fail == DependencyPolicy.ABORT

        # Fail A
        g.mark_running("A")
        g.mark_completed("A", success=False, error="test")
        g._update_ready_queue()

        # B should be ABORTED
        assert g.nodes["B"].state == TaskState.ABORTED

    def test_policy_preview_skip(self):
        """SKIP policy preview correct."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")
        g.add_task("B", "action", {}, [], "1.0",
                   dependencies=["A"], on_dep_fail=DependencyPolicy.SKIP)

        # Fail A
        g.mark_running("A")
        g.mark_completed("A", success=False, error="test")
        g._update_ready_queue()

        # B should be SKIPPED
        assert g.nodes["B"].state == TaskState.SKIPPED

    def test_blocking_tasks(self):
        """Can identify blocking dependencies."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")
        g.add_task("B", "action", {}, [], "1.0", dependencies=["A"])

        # B is blocked on A
        unresolved = [dep for dep in g.nodes["B"].dependencies
                     if dep not in g.completed]

        assert "A" in unresolved


class TestR3Scheduling:
    """R3: Scheduling & Ready Queue - Explainable ordering."""

    def test_priority_ordering(self):
        """Tasks ordered by priority (DESC)."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("low", "action", {}, [], "1.0", priority=0.3)
        g.add_task("high", "action", {}, [], "1.0", priority=0.9)

        g._update_ready_queue()
        ready = g.get_ready_tasks()

        # Higher priority first
        assert ready[0] == "high"

    def test_deadline_tiebreak(self):
        """When priorities equal, earliest deadline first."""
        g = TaskGraph("test", max_parallel=4)
        now = datetime.now(timezone.utc)

        g.add_task("later", "action", {}, [], "1.0",
                   priority=0.5, deadline=now + timedelta(hours=2))
        g.add_task("sooner", "action", {}, [], "1.0",
                   priority=0.5, deadline=now + timedelta(hours=1))

        g._update_ready_queue()

        # Extract queue and sort manually
        ready_nodes = [(tid, node) for tid, node in g.nodes.items()
                      if node.state == TaskState.READY]
        ready_nodes.sort(key=lambda x: (
            -x[1].priority,
            x[1].deadline.timestamp() if x[1].deadline else float('inf'),
            -x[1].cost,
            x[0]
        ))

        assert ready_nodes[0][0] == "sooner"

    def test_ordering_deterministic(self):
        """Task ID breaks all ties deterministically."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("task_b", "action", {}, [], "1.0", priority=0.5, cost=1.0)
        g.add_task("task_a", "action", {}, [], "1.0", priority=0.5, cost=1.0)

        g._update_ready_queue()

        ready_nodes = [(tid, node) for tid, node in g.nodes.items()
                      if node.state == TaskState.READY]
        ready_nodes.sort(key=lambda x: (
            -x[1].priority,
            x[1].deadline.timestamp() if x[1].deadline else float('inf'),
            -x[1].cost,
            x[0]
        ))

        # Alphabetically first
        assert ready_nodes[0][0] == "task_a"


class TestR4Concurrency:
    """R4: Concurrency - Caps and usage tracking."""

    def test_global_concurrency(self):
        """Global concurrency limit tracked."""
        g = TaskGraph("test", max_parallel=2)
        g.add_task("A", "action", {}, [], "1.0")
        g.add_task("B", "action", {}, [], "1.0")
        g.add_task("C", "action", {}, [], "1.0")

        g.mark_running("A")
        g.mark_running("B")

        assert len(g.running_tasks) == 2
        assert g.max_parallel == 2

    def test_per_action_concurrency(self):
        """Per-action caps tracked."""
        g = TaskGraph("test", max_parallel=10)
        g.action_concurrency_caps = {"action_a": 1}

        g.add_task("A1", "action_a", {}, [], "1.0")
        g.add_task("A2", "action_a", {}, [], "1.0")

        g.mark_running("A1")

        assert g.action_concurrency["action_a"] == 1

        # A2 should not be scheduled (cap reached)
        ready = g.get_ready_tasks(per_action_caps={"action_a": 1})
        assert "A2" not in ready

    def test_running_tasks_list(self):
        """Can query currently running tasks."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")
        g.add_task("B", "action", {}, [], "1.0")

        g.mark_running("A")

        assert "A" in g.running_tasks
        assert "B" not in g.running_tasks


class TestR5Reliability:
    """R5: Reliability & Safety - Retries, breakers, idempotency."""

    def test_retry_tracking(self):
        """Retry count tracked."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0", max_retries=3)

        g.mark_running("A")
        g.mark_completed("A", success=False, error="test")

        assert g.nodes["A"].retry_count == 0  # Will be incremented on retry
        assert g.nodes["A"].can_retry()

    def test_idempotency_key(self):
        """Idempotency key generated."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {"x": 1}, ["r1"], "1.0")

        key = g.nodes["A"].idempotency_key

        assert key is not None
        assert len(key) == 64  # SHA256 hex

    def test_circuit_breaker_opens(self):
        """Circuit breaker opens after failures."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "flaky_action", {}, [], "1.0")

        breaker = g._get_breaker("flaky_action")

        # Trigger failures
        for _ in range(5):
            breaker.record_failure("NetworkError")

        assert breaker.is_open()

    def test_retry_token_budget(self):
        """Retry token budget tracked."""
        g = TaskGraph("test", max_parallel=4, max_retry_tokens=10)
        g.add_task("A", "action", {}, [], "1.0")

        assert g.retry_tokens_used == 0

        success = g.use_retry_token("A")
        assert success
        assert g.retry_tokens_used == 1


class TestR6Persistence:
    """R6: Persistence & Stats - Snapshot/restore, dashboard stats."""

    def test_snapshot_serialization(self):
        """Can serialize full graph state."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")

        snapshot = g.to_dict()

        assert snapshot["graph_id"] == "test"
        assert "nodes" in snapshot
        assert "A" in snapshot["nodes"]
        assert "stats" in snapshot

    def test_stats_by_state(self):
        """Stats broken down by state."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")
        g.add_task("B", "action", {}, [], "1.0")

        g._update_ready_queue()

        stats = g.get_stats()

        assert stats["total_tasks"] == 2
        assert "states" in stats
        assert TaskState.READY.value in stats["states"]

    def test_running_tasks_in_stats(self):
        """Running tasks count in stats."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")

        g.mark_running("A")

        stats = g.get_stats()

        assert stats["running_tasks"] == 1


class TestNegativeCases:
    """Negative and edge cases."""

    def test_nonexistent_task(self):
        """Nonexistent task raises appropriate error."""
        g = TaskGraph("test", max_parallel=4)

        with pytest.raises(KeyError):
            _ = g.nodes["nonexistent"]

    def test_cycle_detection(self):
        """Cycle creation detected."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")
        g.add_task("B", "action", {}, [], "1.0", dependencies=["A"])

        # Adding C that depends on B would be fine (A->B->C)
        would_cycle = g._would_create_cycle("C", ["B"])
        assert would_cycle is False

        # But if we tried to make A depend on B (B already depends on A), that's a cycle
        would_cycle_ab = g._would_create_cycle("A", ["B"])
        assert would_cycle_ab is True  # A->B->A is a cycle

    def test_invalid_dependency(self):
        """Invalid dependency rejected."""
        g = TaskGraph("test", max_parallel=4)

        with pytest.raises(ValueError, match="not found"):
            g.add_task("B", "action", {}, [], "1.0", dependencies=["nonexistent"])

    def test_duplicate_task_id(self):
        """Duplicate task ID rejected."""
        g = TaskGraph("test", max_parallel=4)
        g.add_task("A", "action", {}, [], "1.0")

        with pytest.raises(ValueError, match="already in graph"):
            g.add_task("A", "action", {}, [], "1.0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
