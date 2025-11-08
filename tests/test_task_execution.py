"""Tests for task execution tracking (Phase 1).

Run these tests after implementing Phase 1 to verify all components work correctly.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path

from src.memory.models import ExperienceType, CaptureMethod, ExperienceModel
from src.memory.raw_store import RawStore
from src.pipeline.task_experience import create_task_execution_experience


def test_task_execution_enum_exists():
    """Test that TASK_EXECUTION enum is defined."""
    assert ExperienceType.TASK_EXECUTION.value == "task_execution"


def test_capture_method_enums_exist():
    """Test that task capture methods are defined."""
    assert CaptureMethod.SCHEDULED_TASK.value == "scheduled_task"
    assert CaptureMethod.MANUAL_TASK.value == "manual_task"


def test_create_task_execution_experience():
    """Test creating a task execution experience."""
    started_at = datetime(2025, 11, 5, 18, 29, 12, tzinfo=timezone.utc)
    ended_at = datetime(2025, 11, 5, 18, 29, 13, tzinfo=timezone.utc)

    experience = create_task_execution_experience(
        task_id="test_task",
        task_slug="test_task",
        task_name="Test Task",
        task_type="test",
        scheduled_vs_manual="scheduled",
        started_at=started_at,
        ended_at=ended_at,
        status="success",
        response_text="Test response",
        error=None,
        parent_experience_ids=["exp_123", "exp_456"],
        retrieval_metadata={"memory_count": 2, "source": ["experiences"]},
        files_written=[],
        task_config={"type": "test"},
        trace_id="test-trace-id",
        span_id="test-span-id",
        attempt=1,
        retry_of=None,
    )

    # Verify experience type
    assert experience.type == ExperienceType.TASK_EXECUTION

    # Verify structured content fields
    assert experience.content.structured["task_id"] == "test_task"
    assert experience.content.structured["status"] == "success"
    assert experience.content.structured["duration_ms"] == 1000  # 1 second
    assert experience.content.structured["trace_id"] == "test-trace-id"
    assert experience.content.structured["span_id"] == "test-span-id"

    # Verify parents
    assert len(experience.parents) == 2
    assert "exp_123" in experience.parents

    # Verify retrieval metadata
    assert experience.content.structured["retrieval"]["memory_count"] == 2

    # Verify PII scrubbing flag
    assert experience.content.structured["scrubbed"] is True


def test_raw_store_has_new_methods(tmp_path):
    """Test that RawStore has new methods for task execution tracking."""
    db_path = tmp_path / "test.db"
    raw_store = RawStore(db_path)

    # Test that methods exist
    assert hasattr(raw_store, "append_experience_idempotent")
    assert hasattr(raw_store, "list_task_executions")
    assert hasattr(raw_store, "get_by_trace_id")
    assert hasattr(raw_store, "_create_task_indexes")


def test_idempotent_insert(tmp_path):
    """Test idempotent experience insertion."""
    db_path = tmp_path / "test.db"
    raw_store = RawStore(db_path)

    started_at = datetime.now(timezone.utc)
    ended_at = datetime.now(timezone.utc)

    experience = create_task_execution_experience(
        task_id="test_idempotent",
        task_slug="test_idempotent",
        task_name="Test Idempotent",
        task_type="test",
        scheduled_vs_manual="scheduled",
        started_at=started_at,
        ended_at=ended_at,
        status="success",
        response_text="Test",
        error=None,
        parent_experience_ids=[],
        retrieval_metadata={"memory_count": 0, "source": []},
        files_written=[],
        task_config={},
        trace_id=None,
        span_id=None,
        attempt=1,
        retry_of=None,
    )

    idempotency_key = experience.content.structured["idempotency_key"]

    # Insert first time
    exp_id_1 = raw_store.append_experience_idempotent(experience, idempotency_key)

    # Insert second time - should return same ID
    exp_id_2 = raw_store.append_experience_idempotent(experience, idempotency_key)

    assert exp_id_1 == exp_id_2
    print(f"✅ Idempotency works: {exp_id_1} == {exp_id_2}")


def test_list_task_executions(tmp_path):
    """Test querying task executions."""
    db_path = tmp_path / "test.db"
    raw_store = RawStore(db_path)

    # Create multiple task executions
    for i in range(3):
        started_at = datetime.now(timezone.utc)
        ended_at = datetime.now(timezone.utc)

        experience = create_task_execution_experience(
            task_id=f"test_task_{i}",
            task_slug=f"test_task_{i}",
            task_name=f"Test Task {i}",
            task_type="test",
            scheduled_vs_manual="scheduled",
            started_at=started_at,
            ended_at=ended_at,
            status="success",
            response_text=f"Response {i}",
            error=None,
            parent_experience_ids=[],
            retrieval_metadata={"memory_count": 0, "source": []},
            files_written=[],
            task_config={},
            trace_id=None,
            span_id=None,
            attempt=1,
            retry_of=None,
        )

        idempotency_key = experience.content.structured["idempotency_key"]
        raw_store.append_experience_idempotent(experience, idempotency_key)

    # Query all task executions
    results = raw_store.list_task_executions(limit=10)
    assert len(results) == 3

    # Query specific task
    results = raw_store.list_task_executions(task_id="test_task_0", limit=10)
    assert len(results) == 1
    assert results[0].content.structured["task_id"] == "test_task_0"


if __name__ == "__main__":
    # Run basic tests
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        print("Running task execution tracking tests...")
        print("")

        print("Test 1: Enum definitions...")
        test_task_execution_enum_exists()
        test_capture_method_enums_exist()
        print("✅ PASS")

        print("\nTest 2: Create task execution experience...")
        test_create_task_execution_experience()
        print("✅ PASS")

        print("\nTest 3: RawStore methods exist...")
        test_raw_store_has_new_methods(tmp_path)
        print("✅ PASS")

        print("\nTest 4: Idempotent insertion...")
        test_idempotent_insert(tmp_path)
        print("✅ PASS")

        print("\nTest 5: List task executions...")
        test_list_task_executions(tmp_path)
        print("✅ PASS")

        print("\n" + "=" * 60)
        print("All tests PASSED!")
        print("=" * 60)
