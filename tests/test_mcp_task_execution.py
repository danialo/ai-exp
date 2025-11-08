"""Tests for TaskExecutionTooling (MCP task execution helpers)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest

from src.memory.raw_store import RawStore
from src.pipeline.task_experience import create_task_execution_experience
from src.mcp.task_execution_server import TaskExecutionTooling


@pytest.fixture()
def temp_raw_store(tmp_path: Path) -> Iterator[RawStore]:
    store = RawStore(tmp_path / "task_exec.db")
    try:
        yield store
    finally:
        store.close()


def _store_task(
    raw_store: RawStore,
    *,
    task_id: str,
    started_at: datetime,
    ended_at: datetime,
    status: str,
    trace_id: str,
    span_id: str,
    attempt: int,
    backfilled: bool = False,
    error: dict | None = None,
    retry_of: str | None = None,
) -> None:
    experience = create_task_execution_experience(
        task_id=task_id,
        task_slug=task_id,
        task_name=task_id.title(),
        task_type="test",
        scheduled_vs_manual="scheduled",
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        response_text=f"Response for {task_id}",
        error=error,
        parent_experience_ids=[],
        retrieval_metadata={"memory_count": 0, "source": []},
        files_written=[f"/tmp/{task_id}_{attempt}.log"] if status == "failed" else [],
        task_config={},
        trace_id=trace_id,
        span_id=span_id,
        attempt=attempt,
        retry_of=retry_of,
    )

    if backfilled:
        experience.content.structured["backfilled"] = True

    raw_store.append_experience(experience)


def test_tasks_list_filters(temp_raw_store: RawStore):
    tooling = TaskExecutionTooling(temp_raw_store)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)

    _store_task(
        temp_raw_store,
        task_id="alpha",
        started_at=base - timedelta(hours=4),
        ended_at=base - timedelta(hours=4, minutes=30),
        status="success",
        trace_id="trace-alpha",
        span_id="span-alpha-1",
        attempt=1,
    )

    _store_task(
        temp_raw_store,
        task_id="beta",
        started_at=base - timedelta(hours=2),
        ended_at=base - timedelta(hours=2, minutes=20),
        status="failed",
        trace_id="trace-beta",
        span_id="span-beta-1",
        attempt=1,
        error={"type": "ValueError", "message": "boom", "stack_hash": "hash1"},
    )

    _store_task(
        temp_raw_store,
        task_id="gamma",
        started_at=base - timedelta(hours=1),
        ended_at=base - timedelta(hours=1, minutes=10),
        status="failed",
        trace_id="trace-gamma",
        span_id="span-gamma-1",
        attempt=1,
        error={"type": "RuntimeError", "message": "oops", "stack_hash": "hash2"},
        backfilled=True,
    )

    result = tooling.tasks_list(
        status="failed",
        limit=5,
        since=(base - timedelta(hours=3)).isoformat(),
        backfilled=False,
    )

    assert result["has_more"] is False
    assert result["total"] == 1
    assert result["executions"][0]["task_id"] == "beta"
    assert result["executions"][0]["error"]["stack_hash"] == "hash1"


def test_tasks_by_trace(temp_raw_store: RawStore):
    tooling = TaskExecutionTooling(temp_raw_store)
    base = datetime(2025, 1, 2, tzinfo=timezone.utc)

    _store_task(
        temp_raw_store,
        task_id="delta",
        started_at=base - timedelta(hours=2),
        ended_at=base - timedelta(hours=2, minutes=30),
        status="failed",
        trace_id="trace-delta",
        span_id="span-delta-1",
        attempt=1,
        error={"type": "RuntimeError", "message": "fail", "stack_hash": "stackA"},
    )

    _store_task(
        temp_raw_store,
        task_id="delta",
        started_at=base - timedelta(hours=1),
        ended_at=base - timedelta(hours=1, minutes=30),
        status="success",
        trace_id="trace-delta",
        span_id="span-delta-2",
        attempt=2,
        retry_of="span-delta-1",
    )

    result = tooling.tasks_by_trace(trace_id="trace-delta")

    assert result["retry_count"] == 1
    assert result["final_status"] == "success"
    assert [execution["attempt"] for execution in result["executions"]] == [1, 2]


def test_tasks_last_failed_unique_errors(temp_raw_store: RawStore):
    tooling = TaskExecutionTooling(temp_raw_store)
    base = datetime(2025, 1, 3, tzinfo=timezone.utc)

    for idx in range(4):
        stack = "stackX" if idx < 2 else f"stack{idx}"
        _store_task(
            temp_raw_store,
            task_id=f"task_{idx}",
            started_at=base - timedelta(minutes=idx * 5),
            ended_at=base - timedelta(minutes=idx * 5 - 1),
            status="failed",
            trace_id=f"trace-{idx}",
            span_id=f"span-{idx}",
            attempt=1,
            error={"type": "RuntimeError", "message": "msg", "stack_hash": stack},
        )

    result = tooling.tasks_last_failed(limit=3, unique_errors=True)

    hashes = [failure["error"]["stack_hash"] for failure in result["failures"]]
    assert len(hashes) == len(set(hashes))
    assert len(hashes) <= 3

    pattern_hashes = {pattern["stack_hash"] for pattern in result["error_patterns"]}
    assert {"stackX", "stack2", "stack3"}.issubset(pattern_hashes)
