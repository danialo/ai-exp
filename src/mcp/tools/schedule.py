"""MCP tools for autonomous scheduling with safety tiers and budget enforcement.

Tools:
- astra.schedule.create: Create a new scheduled task
- astra.schedule.modify: Modify existing schedule
- astra.schedule.pause: Pause a schedule
- astra.schedule.resume: Resume a paused schedule
"""

import json
import logging
from typing import Any, Dict, Optional

from src.services.schedule_service import (
    SafetyTier,
    ScheduleService,
    ScheduleStatus,
    create_schedule_service,
)

logger = logging.getLogger(__name__)


class ScheduleTools:
    """MCP tool handlers for schedule management."""

    def __init__(self, schedule_service: Optional[ScheduleService] = None):
        """Initialize schedule tools.

        Args:
            schedule_service: ScheduleService instance (creates default if None)
        """
        self.schedule_service = schedule_service or create_schedule_service()

    def create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new schedule.

        Args:
            payload: {
                "name": str,
                "cron_expression": str,
                "target_tool": str,
                "payload": dict,
                "safety_tier": int (0, 1, or 2),
                "per_day_budget": int (default: 4)
            }

        Returns:
            {
                "success": bool,
                "schedule_id": str,
                "schedule": dict,
                "error": str (if failed)
            }
        """
        try:
            # Validate required fields
            required = ["name", "cron_expression", "target_tool", "payload"]
            missing = [f for f in required if f not in payload]
            if missing:
                return {
                    "success": False,
                    "error": f"Missing required fields: {', '.join(missing)}",
                }

            # Extract fields
            name = payload["name"]
            cron = payload["cron_expression"]
            target_tool = payload["target_tool"]
            tool_payload = payload["payload"]
            safety_tier = SafetyTier(payload.get("safety_tier", 1))  # Default: LOCAL_WRITE
            per_day_budget = payload.get("per_day_budget", 4)

            # Create schedule
            schedule = self.schedule_service.create(
                name=name,
                cron=cron,
                target_tool=target_tool,
                payload=tool_payload,
                safety_tier=safety_tier,
                per_day_budget=per_day_budget,
            )

            logger.info(f"Created schedule via MCP: {schedule.id} (name: {name})")

            return {
                "success": True,
                "schedule_id": schedule.id,
                "schedule": {
                    "id": schedule.id,
                    "name": schedule.name,
                    "cron": schedule.cron,
                    "target_tool": schedule.target_tool,
                    "payload": schedule.payload,
                    "next_run_at": schedule.next_run_at,
                    "status": schedule.status.value,
                    "safety_tier": int(schedule.safety_tier),
                    "run_budget": {
                        "per_day": schedule.run_budget.per_day,
                        "consumed": schedule.run_budget.consumed,
                    },
                },
            }

        except ValueError as e:
            logger.warning(f"Schedule creation failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error creating schedule: {e}", exc_info=True)
            return {"success": False, "error": f"Internal error: {e}"}

    def modify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an existing schedule.

        Args:
            payload: {
                "schedule_id": str,
                "cron_expression": str (optional),
                "target_tool": str (optional),
                "payload": dict (optional),
                "per_day_budget": int (optional)
            }

        Returns:
            {
                "success": bool,
                "schedule_id": str,
                "schedule": dict,
                "error": str (if failed)
            }
        """
        try:
            # Validate required fields
            if "schedule_id" not in payload:
                return {"success": False, "error": "Missing required field: schedule_id"}

            schedule_id = payload["schedule_id"]

            # Build updates dict
            updates = {}
            if "cron_expression" in payload:
                updates["cron"] = payload["cron_expression"]
            if "target_tool" in payload:
                updates["target_tool"] = payload["target_tool"]
            if "payload" in payload:
                updates["payload"] = payload["payload"]
            if "per_day_budget" in payload:
                updates["per_day_budget"] = payload["per_day_budget"]

            if not updates:
                return {"success": False, "error": "No updates provided"}

            # Modify schedule
            schedule = self.schedule_service.modify(schedule_id, **updates)

            logger.info(f"Modified schedule via MCP: {schedule_id}")

            return {
                "success": True,
                "schedule_id": schedule.id,
                "schedule": {
                    "id": schedule.id,
                    "name": schedule.name,
                    "cron": schedule.cron,
                    "target_tool": schedule.target_tool,
                    "payload": schedule.payload,
                    "next_run_at": schedule.next_run_at,
                    "status": schedule.status.value,
                    "safety_tier": int(schedule.safety_tier),
                    "run_budget": {
                        "per_day": schedule.run_budget.per_day,
                        "consumed": schedule.run_budget.consumed,
                    },
                },
            }

        except ValueError as e:
            logger.warning(f"Schedule modification failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error modifying schedule: {e}", exc_info=True)
            return {"success": False, "error": f"Internal error: {e}"}

    def pause(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Pause a schedule.

        Args:
            payload: {
                "schedule_id": str
            }

        Returns:
            {
                "success": bool,
                "schedule_id": str,
                "status": str,
                "error": str (if failed)
            }
        """
        try:
            if "schedule_id" not in payload:
                return {"success": False, "error": "Missing required field: schedule_id"}

            schedule_id = payload["schedule_id"]

            # Pause schedule
            schedule = self.schedule_service.pause(schedule_id)

            logger.info(f"Paused schedule via MCP: {schedule_id}")

            return {
                "success": True,
                "schedule_id": schedule.id,
                "status": schedule.status.value,
            }

        except ValueError as e:
            logger.warning(f"Schedule pause failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error pausing schedule: {e}", exc_info=True)
            return {"success": False, "error": f"Internal error: {e}"}

    def resume(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Resume a paused schedule.

        Args:
            payload: {
                "schedule_id": str
            }

        Returns:
            {
                "success": bool,
                "schedule_id": str,
                "status": str,
                "next_run_at": str,
                "error": str (if failed)
            }
        """
        try:
            if "schedule_id" not in payload:
                return {"success": False, "error": "Missing required field: schedule_id"}

            schedule_id = payload["schedule_id"]

            # Resume schedule
            schedule = self.schedule_service.resume(schedule_id)

            logger.info(f"Resumed schedule via MCP: {schedule_id}")

            return {
                "success": True,
                "schedule_id": schedule.id,
                "status": schedule.status.value,
                "next_run_at": schedule.next_run_at,
            }

        except ValueError as e:
            logger.warning(f"Schedule resume failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error resuming schedule: {e}", exc_info=True)
            return {"success": False, "error": f"Internal error: {e}"}

    def list_schedules(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """List all schedules, optionally filtered.

        Args:
            payload: {
                "status": str (optional: "active" or "paused")
            }

        Returns:
            {
                "success": bool,
                "schedules": list[dict],
                "count": int
            }
        """
        try:
            payload = payload or {}
            status = None
            if "status" in payload:
                status_str = payload["status"]
                if status_str == "active":
                    status = ScheduleStatus.ACTIVE
                elif status_str == "paused":
                    status = ScheduleStatus.PAUSED

            schedules = self.schedule_service.list_all(status=status)

            return {
                "success": True,
                "schedules": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "cron": s.cron,
                        "target_tool": s.target_tool,
                        "next_run_at": s.next_run_at,
                        "status": s.status.value,
                        "safety_tier": int(s.safety_tier),
                        "budget_remaining": s.run_budget.per_day - s.run_budget.consumed,
                    }
                    for s in schedules
                ],
                "count": len(schedules),
            }

        except Exception as e:
            logger.error(f"Unexpected error listing schedules: {e}", exc_info=True)
            return {"success": False, "error": f"Internal error: {e}"}


def create_schedule_tools(
    schedule_service: Optional[ScheduleService] = None,
) -> ScheduleTools:
    """Factory function to create ScheduleTools.

    Args:
        schedule_service: Optional ScheduleService instance

    Returns:
        Initialized ScheduleTools
    """
    return ScheduleTools(schedule_service=schedule_service)
