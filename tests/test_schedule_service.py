"""Unit tests for ScheduleService."""

import json
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.services.schedule_service import (
    ScheduleService,
    Schedule,
    ScheduleStatus,
    SafetyTier,
    create_schedule_service,
)


@pytest.fixture
def temp_schedules_dir():
    """Create temporary directory for schedules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def schedule_service(temp_schedules_dir):
    """Create ScheduleService with temp directory."""
    return create_schedule_service(temp_schedules_dir)


class TestScheduleCreation:
    """Test schedule creation and ID generation."""

    def test_create_schedule(self, schedule_service):
        """Test creating a basic schedule."""
        schedule = schedule_service.create(
            name="daily_reflection",
            cron="0 3 * * *",
            target_tool="execute_goal",
            payload={"goal_text": "Run daily reflection"},
            safety_tier=SafetyTier.READ_ONLY,
            per_day_budget=1,
        )

        assert schedule.id.startswith("sch_")
        assert len(schedule.id) == 12  # sch_ + 8 hex chars
        assert schedule.name == "daily_reflection"
        assert schedule.cron == "0 3 * * *"
        assert schedule.status == ScheduleStatus.ACTIVE
        assert schedule.safety_tier == SafetyTier.READ_ONLY
        assert schedule.run_budget.per_day == 1

    def test_deterministic_id_generation(self):
        """Test that same schedule produces same ID."""
        id1 = Schedule.generate_id(
            "test", "0 3 * * *", "tool", {"arg": "value"}
        )
        id2 = Schedule.generate_id(
            "test", "0 3 * * *", "tool", {"arg": "value"}
        )
        assert id1 == id2

    def test_different_schedule_different_id(self):
        """Test that different schedules produce different IDs."""
        id1 = Schedule.generate_id(
            "test", "0 3 * * *", "tool", {"arg": "value"}
        )
        id2 = Schedule.generate_id(
            "test", "0 4 * * *", "tool", {"arg": "value"}  # Different cron
        )
        assert id1 != id2

    def test_duplicate_schedule_raises(self, schedule_service):
        """Test that creating duplicate schedule raises error."""
        schedule_service.create(
            name="test",
            cron="0 3 * * *",
            target_tool="tool",
            payload={},
        )

        with pytest.raises(ValueError, match="already exists"):
            schedule_service.create(
                name="test",
                cron="0 3 * * *",
                target_tool="tool",
                payload={},
            )

    def test_invalid_cron_raises(self, schedule_service):
        """Test that invalid cron expression raises error."""
        with pytest.raises(ValueError, match="Invalid cron"):
            schedule_service.create(
                name="test",
                cron="invalid cron",
                target_tool="tool",
                payload={},
            )


class TestCronParsing:
    """Test cron expression parsing and next run calculation."""

    def test_compute_next_run(self):
        """Test computing next run time from cron."""
        base_time = datetime(2025, 11, 11, 12, 0, 0, tzinfo=timezone.utc)
        next_run = Schedule.compute_next_run("0 3 * * *", base_time)

        assert next_run.hour == 3
        assert next_run.minute == 0
        assert next_run.day == 12  # Next day

    def test_compute_next_run_handles_dst(self):
        """Test that next run calculation handles timezone correctly."""
        cron = "0 3 * * *"
        next_run = Schedule.compute_next_run(cron)

        # Should be timezone-aware
        assert next_run.tzinfo is not None
        assert next_run > datetime.now(timezone.utc)


class TestScheduleRetrieval:
    """Test schedule retrieval operations."""

    def test_get_existing_schedule(self, schedule_service):
        """Test retrieving existing schedule."""
        created = schedule_service.create(
            name="test", cron="0 3 * * *", target_tool="tool", payload={}
        )

        retrieved = schedule_service.get(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "test"

    def test_get_nonexistent_schedule(self, schedule_service):
        """Test retrieving nonexistent schedule returns None."""
        result = schedule_service.get("sch_notfound")
        assert result is None

    def test_list_all_schedules(self, schedule_service):
        """Test listing all schedules."""
        schedule_service.create(
            name="test1", cron="0 3 * * *", target_tool="tool", payload={}
        )
        schedule_service.create(
            name="test2", cron="0 4 * * *", target_tool="tool", payload={}
        )

        all_schedules = schedule_service.list_all()
        assert len(all_schedules) == 2

    def test_list_by_status(self, schedule_service):
        """Test filtering schedules by status."""
        s1 = schedule_service.create(
            name="test1", cron="0 3 * * *", target_tool="tool", payload={}
        )
        s2 = schedule_service.create(
            name="test2", cron="0 4 * * *", target_tool="tool", payload={}
        )

        schedule_service.pause(s1.id)

        active = schedule_service.list_all(status=ScheduleStatus.ACTIVE)
        paused = schedule_service.list_all(status=ScheduleStatus.PAUSED)

        assert len(active) == 1
        assert len(paused) == 1
        assert active[0].id == s2.id
        assert paused[0].id == s1.id


class TestScheduleModification:
    """Test schedule modification operations."""

    def test_modify_cron(self, schedule_service):
        """Test modifying cron expression."""
        schedule = schedule_service.create(
            name="test", cron="0 3 * * *", target_tool="tool", payload={}
        )

        original_next_run = schedule.next_run_at

        modified = schedule_service.modify(schedule.id, cron="0 4 * * *")

        assert modified.cron == "0 4 * * *"
        assert modified.next_run_at != original_next_run

    def test_modify_payload(self, schedule_service):
        """Test modifying payload."""
        schedule = schedule_service.create(
            name="test",
            cron="0 3 * * *",
            target_tool="tool",
            payload={"arg": "old"},
        )

        modified = schedule_service.modify(schedule.id, payload={"arg": "new"})

        assert modified.payload == {"arg": "new"}

    def test_modify_nonexistent_raises(self, schedule_service):
        """Test modifying nonexistent schedule raises error."""
        with pytest.raises(ValueError, match="not found"):
            schedule_service.modify("sch_notfound", cron="0 3 * * *")


class TestSchedulePauseResume:
    """Test pause and resume operations."""

    def test_pause_schedule(self, schedule_service):
        """Test pausing active schedule."""
        schedule = schedule_service.create(
            name="test", cron="0 3 * * *", target_tool="tool", payload={}
        )

        assert schedule.status == ScheduleStatus.ACTIVE

        paused = schedule_service.pause(schedule.id)

        assert paused.status == ScheduleStatus.PAUSED

    def test_pause_already_paused(self, schedule_service):
        """Test pausing already paused schedule is idempotent."""
        schedule = schedule_service.create(
            name="test", cron="0 3 * * *", target_tool="tool", payload={}
        )

        schedule_service.pause(schedule.id)
        schedule_service.pause(schedule.id)  # Should not raise

    def test_resume_schedule(self, schedule_service):
        """Test resuming paused schedule."""
        schedule = schedule_service.create(
            name="test", cron="0 3 * * *", target_tool="tool", payload={}
        )

        schedule_service.pause(schedule.id)
        resumed = schedule_service.resume(schedule.id)

        assert resumed.status == ScheduleStatus.ACTIVE

    def test_resume_recomputes_next_run(self, schedule_service):
        """Test that resume recomputes next_run_at."""
        schedule = schedule_service.create(
            name="test", cron="0 3 * * *", target_tool="tool", payload={}
        )

        schedule_service.pause(schedule.id)

        # Manually set next_run_at to past to ensure recompute changes it
        paused = schedule_service.get(schedule.id)
        original_next_run = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        paused.next_run_at = original_next_run
        schedule_service.index[schedule.id] = paused

        resumed = schedule_service.resume(schedule.id)

        # Next run should be recomputed from current time
        assert resumed.next_run_at != original_next_run


class TestDueSchedules:
    """Test listing due schedules."""

    def test_list_due_schedules(self, schedule_service):
        """Test listing schedules that are due to run."""
        # Create schedule with past next_run_at
        schedule = schedule_service.create(
            name="test", cron="0 3 * * *", target_tool="tool", payload={}
        )

        # Manually set next_run_at to past
        schedule.next_run_at = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        schedule_service.index[schedule.id] = schedule

        due = schedule_service.list_due()

        assert len(due) == 1
        assert due[0].id == schedule.id

    def test_paused_schedules_not_due(self, schedule_service):
        """Test that paused schedules are not returned as due."""
        schedule = schedule_service.create(
            name="test", cron="0 3 * * *", target_tool="tool", payload={}
        )

        # Set to past and pause
        schedule.next_run_at = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        schedule_service.index[schedule.id] = schedule
        schedule_service.pause(schedule.id)

        due = schedule_service.list_due()

        assert len(due) == 0


class TestBudgetTracking:
    """Test budget tracking and enforcement."""

    def test_mark_executed_updates_next_run(self, schedule_service):
        """Test that mark_executed updates next_run_at."""
        schedule = schedule_service.create(
            name="test", cron="*/5 * * * *", target_tool="tool", payload={}  # Every 5 minutes
        )

        # Set next_run_at to a past time to ensure it advances
        past_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()
        schedule.next_run_at = past_time
        schedule_service.index[schedule.id] = schedule

        schedule_service.mark_executed(schedule.id)

        updated = schedule_service.get(schedule.id)
        # Should be computed from now, not from past
        assert updated.next_run_at != past_time
        assert datetime.fromisoformat(updated.next_run_at) > datetime(2020, 1, 1, tzinfo=timezone.utc)

    def test_mark_executed_increments_budget(self, schedule_service):
        """Test that mark_executed increments consumed budget."""
        schedule = schedule_service.create(
            name="test",
            cron="0 3 * * *",
            target_tool="tool",
            payload={},
            per_day_budget=5,
        )

        assert schedule.run_budget.consumed == 0

        schedule_service.mark_executed(schedule.id)

        updated = schedule_service.get(schedule.id)
        assert updated.run_budget.consumed == 1

    def test_check_budget_available(self, schedule_service):
        """Test checking budget availability."""
        schedule = schedule_service.create(
            name="test",
            cron="0 3 * * *",
            target_tool="tool",
            payload={},
            per_day_budget=2,
        )

        assert schedule_service.check_budget(schedule) is True

        schedule_service.mark_executed(schedule.id)
        schedule_service.mark_executed(schedule.id)

        updated = schedule_service.get(schedule.id)
        assert schedule_service.check_budget(updated) is False

    def test_budget_resets_daily(self, schedule_service):
        """Test that budget resets on new day."""
        schedule = schedule_service.create(
            name="test",
            cron="0 3 * * *",
            target_tool="tool",
            payload={},
            per_day_budget=1,
        )

        # Mark as executed
        schedule_service.mark_executed(schedule.id)

        updated = schedule_service.get(schedule.id)
        assert updated.run_budget.consumed == 1

        # Simulate new day by manipulating last_reset
        updated.run_budget.last_reset = datetime(
            2020, 1, 1, tzinfo=timezone.utc
        ).isoformat()
        schedule_service.index[updated.id] = updated

        # Should have budget again
        assert schedule_service.check_budget(updated) is True


class TestPersistence:
    """Test persistence layer."""

    def test_index_persists(self, temp_schedules_dir):
        """Test that index is persisted to disk."""
        service = create_schedule_service(temp_schedules_dir)

        service.create(
            name="test", cron="0 3 * * *", target_tool="tool", payload={}
        )

        # Check index file exists
        index_path = Path(temp_schedules_dir) / "index.json"
        assert index_path.exists()

        # Verify content
        with open(index_path, "r") as f:
            index_data = json.load(f)

        assert len(index_data) == 1

    def test_index_loads_on_init(self, temp_schedules_dir):
        """Test that index is loaded on service initialization."""
        # Create schedule with first service
        service1 = create_schedule_service(temp_schedules_dir)
        schedule = service1.create(
            name="test", cron="0 3 * * *", target_tool="tool", payload={}
        )

        # Create new service instance (should load from disk)
        service2 = create_schedule_service(temp_schedules_dir)

        retrieved = service2.get(schedule.id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_ndjson_chain_appends(self, temp_schedules_dir):
        """Test that events are appended to NDJSON chain."""
        service = create_schedule_service(temp_schedules_dir)

        service.create(
            name="test", cron="0 3 * * *", target_tool="tool", payload={}
        )

        # Check chain file exists
        chain_files = list(Path(temp_schedules_dir).glob("*.ndjson.gz"))
        assert len(chain_files) == 1
