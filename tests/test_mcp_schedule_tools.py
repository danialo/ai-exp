"""Integration tests for MCP schedule tools."""

import json
import tempfile
from pathlib import Path

import pytest

from src.mcp.tools.schedule import ScheduleTools, create_schedule_tools
from src.services.schedule_service import (
    SafetyTier,
    ScheduleStatus,
    create_schedule_service,
)


@pytest.fixture
def temp_schedules_dir():
    """Create temporary directory for schedules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def schedule_tools(temp_schedules_dir):
    """Create ScheduleTools with temp directory."""
    service = create_schedule_service(temp_schedules_dir)
    return create_schedule_tools(schedule_service=service)


class TestScheduleToolsCreate:
    """Test schedule.create tool."""

    def test_create_schedule_success(self, schedule_tools):
        """Test successful schedule creation."""
        payload = {
            "name": "daily_backup",
            "cron_expression": "0 2 * * *",
            "target_tool": "backup_data",
            "payload": {"target": "/data"},
            "safety_tier": 1,
            "per_day_budget": 2,
        }

        result = schedule_tools.create(payload)

        assert result["success"] is True
        assert "schedule_id" in result
        assert result["schedule"]["name"] == "daily_backup"
        assert result["schedule"]["cron"] == "0 2 * * *"
        assert result["schedule"]["safety_tier"] == 1
        assert result["schedule"]["run_budget"]["per_day"] == 2

    def test_create_schedule_missing_fields(self, schedule_tools):
        """Test creation fails with missing required fields."""
        payload = {"name": "incomplete"}

        result = schedule_tools.create(payload)

        assert result["success"] is False
        assert "Missing required fields" in result["error"]

    def test_create_schedule_invalid_cron(self, schedule_tools):
        """Test creation fails with invalid cron expression."""
        payload = {
            "name": "bad_cron",
            "cron_expression": "invalid cron",
            "target_tool": "tool",
            "payload": {},
        }

        result = schedule_tools.create(payload)

        assert result["success"] is False
        assert "Invalid cron" in result["error"]

    def test_create_duplicate_schedule(self, schedule_tools):
        """Test creation fails for duplicate schedule."""
        payload = {
            "name": "test",
            "cron_expression": "0 3 * * *",
            "target_tool": "tool",
            "payload": {},
        }

        # Create first time - should succeed
        result1 = schedule_tools.create(payload)
        assert result1["success"] is True

        # Create second time - should fail
        result2 = schedule_tools.create(payload)
        assert result2["success"] is False
        assert "already exists" in result2["error"]


class TestScheduleToolsModify:
    """Test schedule.modify tool."""

    def test_modify_cron(self, schedule_tools):
        """Test modifying cron expression."""
        # Create schedule
        create_payload = {
            "name": "test",
            "cron_expression": "0 3 * * *",
            "target_tool": "tool",
            "payload": {},
        }
        create_result = schedule_tools.create(create_payload)
        schedule_id = create_result["schedule_id"]

        # Modify cron
        modify_payload = {
            "schedule_id": schedule_id,
            "cron_expression": "0 4 * * *",
        }
        result = schedule_tools.modify(modify_payload)

        assert result["success"] is True
        assert result["schedule"]["cron"] == "0 4 * * *"

    def test_modify_payload(self, schedule_tools):
        """Test modifying tool payload."""
        # Create schedule
        create_payload = {
            "name": "test",
            "cron_expression": "0 3 * * *",
            "target_tool": "tool",
            "payload": {"arg": "old"},
        }
        create_result = schedule_tools.create(create_payload)
        schedule_id = create_result["schedule_id"]

        # Modify payload
        modify_payload = {
            "schedule_id": schedule_id,
            "payload": {"arg": "new"},
        }
        result = schedule_tools.modify(modify_payload)

        assert result["success"] is True
        assert result["schedule"]["payload"] == {"arg": "new"}

    def test_modify_nonexistent(self, schedule_tools):
        """Test modifying nonexistent schedule fails."""
        payload = {
            "schedule_id": "sch_notfound",
            "cron_expression": "0 3 * * *",
        }
        result = schedule_tools.modify(payload)

        assert result["success"] is False
        assert "not found" in result["error"]


class TestScheduleToolsPauseResume:
    """Test schedule.pause and schedule.resume tools."""

    def test_pause_schedule(self, schedule_tools):
        """Test pausing active schedule."""
        # Create schedule
        create_payload = {
            "name": "test",
            "cron_expression": "0 3 * * *",
            "target_tool": "tool",
            "payload": {},
        }
        create_result = schedule_tools.create(create_payload)
        schedule_id = create_result["schedule_id"]

        # Pause it
        pause_payload = {"schedule_id": schedule_id}
        result = schedule_tools.pause(pause_payload)

        assert result["success"] is True
        assert result["status"] == "paused"

    def test_resume_schedule(self, schedule_tools):
        """Test resuming paused schedule."""
        # Create and pause schedule
        create_payload = {
            "name": "test",
            "cron_expression": "0 3 * * *",
            "target_tool": "tool",
            "payload": {},
        }
        create_result = schedule_tools.create(create_payload)
        schedule_id = create_result["schedule_id"]

        schedule_tools.pause({"schedule_id": schedule_id})

        # Resume it
        resume_payload = {"schedule_id": schedule_id}
        result = schedule_tools.resume(resume_payload)

        assert result["success"] is True
        assert result["status"] == "active"
        assert "next_run_at" in result


class TestScheduleToolsList:
    """Test schedule.list tool."""

    def test_list_all_schedules(self, schedule_tools):
        """Test listing all schedules."""
        # Create two schedules
        schedule_tools.create(
            {
                "name": "test1",
                "cron_expression": "0 3 * * *",
                "target_tool": "tool",
                "payload": {},
            }
        )
        schedule_tools.create(
            {
                "name": "test2",
                "cron_expression": "0 4 * * *",
                "target_tool": "tool",
                "payload": {},
            }
        )

        result = schedule_tools.list_schedules({})

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["schedules"]) == 2

    def test_list_by_status(self, schedule_tools):
        """Test filtering schedules by status."""
        # Create two schedules
        s1 = schedule_tools.create(
            {
                "name": "test1",
                "cron_expression": "0 3 * * *",
                "target_tool": "tool",
                "payload": {},
            }
        )
        s2 = schedule_tools.create(
            {
                "name": "test2",
                "cron_expression": "0 4 * * *",
                "target_tool": "tool",
                "payload": {},
            }
        )

        # Pause first one
        schedule_tools.pause({"schedule_id": s1["schedule_id"]})

        # List active only
        active_result = schedule_tools.list_schedules({"status": "active"})
        assert active_result["success"] is True
        assert active_result["count"] == 1
        assert active_result["schedules"][0]["status"] == "active"

        # List paused only
        paused_result = schedule_tools.list_schedules({"status": "paused"})
        assert paused_result["success"] is True
        assert paused_result["count"] == 1
        assert paused_result["schedules"][0]["status"] == "paused"


class TestSafetyTierAndBudgets:
    """Test safety tier and budget functionality."""

    def test_schedule_with_safety_tier(self, schedule_tools):
        """Test creating schedules with different safety tiers."""
        # Tier 0: Read-only
        tier0 = schedule_tools.create(
            {
                "name": "readonly",
                "cron_expression": "*/5 * * * *",
                "target_tool": "tasks_list",
                "payload": {},
                "safety_tier": 0,
            }
        )
        assert tier0["success"] is True
        assert tier0["schedule"]["safety_tier"] == 0

        # Tier 1: Local write
        tier1 = schedule_tools.create(
            {
                "name": "local_write",
                "cron_expression": "0 * * * *",
                "target_tool": "execute_goal",
                "payload": {"goal": "test"},
                "safety_tier": 1,
            }
        )
        assert tier1["success"] is True
        assert tier1["schedule"]["safety_tier"] == 1

        # Tier 2: External
        tier2 = schedule_tools.create(
            {
                "name": "external",
                "cron_expression": "0 0 * * *",
                "target_tool": "deploy_to_prod",
                "payload": {},
                "safety_tier": 2,
            }
        )
        assert tier2["success"] is True
        assert tier2["schedule"]["safety_tier"] == 2

    def test_budget_in_schedule(self, schedule_tools):
        """Test that budget is set correctly."""
        result = schedule_tools.create(
            {
                "name": "limited",
                "cron_expression": "*/10 * * * *",
                "target_tool": "tool",
                "payload": {},
                "per_day_budget": 10,
            }
        )

        assert result["success"] is True
        assert result["schedule"]["run_budget"]["per_day"] == 10
        assert result["schedule"]["run_budget"]["consumed"] == 0

    def test_list_shows_budget_remaining(self, schedule_tools):
        """Test that list shows remaining budget."""
        schedule_tools.create(
            {
                "name": "budgeted",
                "cron_expression": "*/5 * * * *",
                "target_tool": "tool",
                "payload": {},
                "per_day_budget": 5,
            }
        )

        result = schedule_tools.list_schedules({})
        assert result["success"] is True
        assert result["schedules"][0]["budget_remaining"] == 5
