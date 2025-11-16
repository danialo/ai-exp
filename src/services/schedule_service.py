"""Schedule service for autonomous task scheduling with safety tiers.

Persistence:
- Append-only NDJSON chain: var/schedules/YYYY-MM.ndjson.gz
- Compact KV index: var/schedules/index.json

Safety Tiers:
- Tier 0: Read-only introspection (auto-run)
- Tier 1: Local writes with per-day budgets (auto-run, capped)
- Tier 2: External side effects (requires approval token)
"""

import gzip
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from croniter import croniter

logger = logging.getLogger(__name__)


class ScheduleStatus(str, Enum):
    """Schedule execution status."""

    ACTIVE = "active"
    PAUSED = "paused"


class SafetyTier(int, Enum):
    """Safety tiers for schedule operations."""

    READ_ONLY = 0  # Introspection tools (auto-run)
    LOCAL_WRITE = 1  # Repo writes with budgets (auto-run, capped)
    EXTERNAL = 2  # External side effects (requires approval)


@dataclass
class RunBudget:
    """Per-day execution budget."""

    per_day: int = 4  # Max runs per day
    consumed: int = 0  # Runs consumed today
    last_reset: Optional[str] = None  # ISO8601 timestamp of last reset


@dataclass
class Schedule:
    """Scheduled task definition."""

    id: str  # sch_<sha8>
    name: str
    cron: str  # 5-field or 6-field cron expression
    target_tool: str  # Tool to invoke (e.g., "execute_goal")
    payload: Dict  # Tool arguments
    next_run_at: str  # ISO8601 timestamp
    status: ScheduleStatus
    created_at: str  # ISO8601 timestamp
    updated_at: str  # ISO8601 timestamp
    safety_tier: SafetyTier
    run_budget: RunBudget = field(default_factory=RunBudget)

    @staticmethod
    def generate_id(name: str, cron: str, target_tool: str, payload: Dict) -> str:
        """Generate deterministic schedule ID from components.

        Args:
            name: Schedule name
            cron: Cron expression
            target_tool: Target tool name
            payload: Tool payload

        Returns:
            Deterministic ID: sch_<sha8>
        """
        # Sort payload keys for deterministic hashing
        payload_json = json.dumps(payload, sort_keys=True)
        composite = f"{name}:{cron}:{target_tool}:{payload_json}"
        hash_digest = hashlib.sha256(composite.encode()).hexdigest()
        return f"sch_{hash_digest[:8]}"

    @staticmethod
    def compute_next_run(
        cron: str, base_time: Optional[datetime] = None
    ) -> datetime:
        """Compute next run time from cron expression.

        Args:
            cron: Cron expression (5 or 6 fields)
            base_time: Base time for calculation (defaults to now)

        Returns:
            Next run datetime in UTC

        Raises:
            ValueError: If cron expression is invalid
        """
        if base_time is None:
            base_time = datetime.now(timezone.utc)

        try:
            cron_iter = croniter(cron, base_time)
            next_run = cron_iter.get_next(datetime)
            # Ensure timezone-aware
            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=timezone.utc)
            return next_run
        except Exception as e:
            raise ValueError(f"Invalid cron expression '{cron}': {e}")


class ScheduleService:
    """Service for managing scheduled tasks with NDJSON persistence."""

    def __init__(self, schedules_dir: str = "var/schedules"):
        """Initialize schedule service.

        Args:
            schedules_dir: Directory for schedule persistence
        """
        self.schedules_dir = Path(schedules_dir)
        self.schedules_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.schedules_dir / "index.json"

        # In-memory index: schedule_id -> Schedule
        self.index: Dict[str, Schedule] = {}

        # Load index from disk
        self._load_index()

    def _load_index(self) -> None:
        """Load schedule index from disk."""
        if not self.index_path.exists():
            logger.info("No schedule index found, starting fresh")
            return

        try:
            with open(self.index_path, "r") as f:
                index_data = json.load(f)

            for schedule_id, schedule_dict in index_data.items():
                # Convert dict to Schedule object
                schedule_dict["status"] = ScheduleStatus(schedule_dict["status"])
                schedule_dict["safety_tier"] = SafetyTier(schedule_dict["safety_tier"])
                schedule_dict["run_budget"] = RunBudget(**schedule_dict["run_budget"])

                self.index[schedule_id] = Schedule(**schedule_dict)

            logger.info(f"Loaded {len(self.index)} schedules from index")

        except Exception as e:
            logger.error(f"Failed to load schedule index: {e}")
            # Continue with empty index

    def _save_index(self) -> None:
        """Save schedule index to disk."""
        try:
            # Convert Schedule objects to dicts
            index_data = {}
            for schedule_id, schedule in self.index.items():
                schedule_dict = asdict(schedule)
                # Convert enums to strings
                schedule_dict["status"] = schedule.status.value
                schedule_dict["safety_tier"] = int(schedule.safety_tier)
                index_data[schedule_id] = schedule_dict

            # Write atomically (write to temp, then rename)
            temp_path = self.index_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(index_data, f, indent=2)

            temp_path.replace(self.index_path)

            logger.debug(f"Saved {len(self.index)} schedules to index")

        except Exception as e:
            logger.error(f"Failed to save schedule index: {e}")

    def _append_to_chain(self, event: Dict) -> None:
        """Append event to NDJSON chain.

        Args:
            event: Event to append (create, modify, pause, resume)
        """
        now = datetime.now(timezone.utc)
        year_month = now.strftime("%Y-%m")
        chain_path = self.schedules_dir / f"{year_month}.ndjson.gz"

        try:
            # Add timestamp to event
            event["_timestamp"] = now.isoformat()

            # Append to gzipped NDJSON
            with gzip.open(chain_path, "at", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

            logger.debug(f"Appended event to chain: {chain_path}")

        except Exception as e:
            logger.error(f"Failed to append to chain: {e}")

    def create(
        self,
        name: str,
        cron: str,
        target_tool: str,
        payload: Dict,
        safety_tier: SafetyTier = SafetyTier.LOCAL_WRITE,
        per_day_budget: int = 4,
    ) -> Schedule:
        """Create a new schedule.

        Args:
            name: Human-readable schedule name
            cron: Cron expression (5 or 6 fields)
            target_tool: Tool to invoke
            payload: Tool arguments
            safety_tier: Safety tier (0, 1, or 2)
            per_day_budget: Max runs per day

        Returns:
            Created schedule

        Raises:
            ValueError: If schedule already exists or cron is invalid
        """
        # Generate deterministic ID
        schedule_id = Schedule.generate_id(name, cron, target_tool, payload)

        if schedule_id in self.index:
            raise ValueError(f"Schedule already exists: {schedule_id} (name: {name})")

        # Compute next run time
        next_run = Schedule.compute_next_run(cron)

        now = datetime.now(timezone.utc).isoformat()

        # Create schedule
        schedule = Schedule(
            id=schedule_id,
            name=name,
            cron=cron,
            target_tool=target_tool,
            payload=payload,
            next_run_at=next_run.isoformat(),
            status=ScheduleStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            safety_tier=safety_tier,
            run_budget=RunBudget(per_day=per_day_budget, consumed=0, last_reset=now),
        )

        # Add to index
        self.index[schedule_id] = schedule
        self._save_index()

        # Append to chain
        self._append_to_chain(
            {"event": "schedule_created", "schedule_id": schedule_id, "schedule": asdict(schedule)}
        )

        logger.info(f"Created schedule: {schedule_id} (name: {name})")
        return schedule

    def get(self, schedule_id: str) -> Optional[Schedule]:
        """Get schedule by ID.

        Args:
            schedule_id: Schedule ID

        Returns:
            Schedule or None if not found
        """
        return self.index.get(schedule_id)

    def list_all(self, status: Optional[ScheduleStatus] = None) -> List[Schedule]:
        """List all schedules, optionally filtered by status.

        Args:
            status: Filter by status (None = all)

        Returns:
            List of schedules
        """
        schedules = list(self.index.values())
        if status is not None:
            schedules = [s for s in schedules if s.status == status]
        return sorted(schedules, key=lambda s: s.created_at)

    def list_due(self, now: Optional[datetime] = None) -> List[Schedule]:
        """List schedules that are due to run.

        Args:
            now: Current time (defaults to now UTC)

        Returns:
            List of due schedules (active and past next_run_at)
        """
        if now is None:
            now = datetime.now(timezone.utc)

        due_schedules = []
        for schedule in self.index.values():
            if schedule.status != ScheduleStatus.ACTIVE:
                continue

            next_run = datetime.fromisoformat(schedule.next_run_at)
            if next_run <= now:
                due_schedules.append(schedule)

        return sorted(due_schedules, key=lambda s: s.next_run_at)

    def modify(self, schedule_id: str, **updates) -> Schedule:
        """Modify an existing schedule.

        Args:
            schedule_id: Schedule ID
            **updates: Fields to update (cron, target_tool, payload, per_day_budget)

        Returns:
            Updated schedule

        Raises:
            ValueError: If schedule not found or update invalid
        """
        schedule = self.get(schedule_id)
        if not schedule:
            raise ValueError(f"Schedule not found: {schedule_id}")

        # Apply updates
        modified = False
        for key, value in updates.items():
            if key == "cron" and value != schedule.cron:
                # Recompute next_run_at
                schedule.cron = value
                schedule.next_run_at = Schedule.compute_next_run(value).isoformat()
                modified = True
            elif key == "target_tool" and value != schedule.target_tool:
                schedule.target_tool = value
                modified = True
            elif key == "payload" and value != schedule.payload:
                schedule.payload = value
                modified = True
            elif key == "per_day_budget" and value != schedule.run_budget.per_day:
                schedule.run_budget.per_day = value
                modified = True

        if modified:
            schedule.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_index()
            self._append_to_chain(
                {"event": "schedule_modified", "schedule_id": schedule_id, "updates": updates}
            )
            logger.info(f"Modified schedule: {schedule_id}")

        return schedule

    def pause(self, schedule_id: str) -> Schedule:
        """Pause a schedule.

        Args:
            schedule_id: Schedule ID

        Returns:
            Updated schedule

        Raises:
            ValueError: If schedule not found
        """
        schedule = self.get(schedule_id)
        if not schedule:
            raise ValueError(f"Schedule not found: {schedule_id}")

        if schedule.status == ScheduleStatus.PAUSED:
            return schedule  # Already paused

        schedule.status = ScheduleStatus.PAUSED
        schedule.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_index()
        self._append_to_chain({"event": "schedule_paused", "schedule_id": schedule_id})

        logger.info(f"Paused schedule: {schedule_id}")
        return schedule

    def resume(self, schedule_id: str) -> Schedule:
        """Resume a paused schedule.

        Args:
            schedule_id: Schedule ID

        Returns:
            Updated schedule

        Raises:
            ValueError: If schedule not found
        """
        schedule = self.get(schedule_id)
        if not schedule:
            raise ValueError(f"Schedule not found: {schedule_id}")

        if schedule.status == ScheduleStatus.ACTIVE:
            return schedule  # Already active

        schedule.status = ScheduleStatus.ACTIVE
        # Recompute next_run_at from now
        schedule.next_run_at = Schedule.compute_next_run(schedule.cron).isoformat()
        schedule.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_index()
        self._append_to_chain({"event": "schedule_resumed", "schedule_id": schedule_id})

        logger.info(f"Resumed schedule: {schedule_id}")
        return schedule

    def mark_executed(self, schedule_id: str) -> None:
        """Mark schedule as executed and update next_run_at.

        Args:
            schedule_id: Schedule ID

        Raises:
            ValueError: If schedule not found
        """
        schedule = self.get(schedule_id)
        if not schedule:
            raise ValueError(f"Schedule not found: {schedule_id}")

        # Update next_run_at
        schedule.next_run_at = Schedule.compute_next_run(schedule.cron).isoformat()

        # Increment budget consumed
        now = datetime.now(timezone.utc)
        last_reset = (
            datetime.fromisoformat(schedule.run_budget.last_reset)
            if schedule.run_budget.last_reset
            else now
        )

        # Reset budget if new day
        if now.date() > last_reset.date():
            schedule.run_budget.consumed = 0
            schedule.run_budget.last_reset = now.isoformat()

        schedule.run_budget.consumed += 1
        schedule.updated_at = now.isoformat()

        self._save_index()
        self._append_to_chain({"event": "schedule_executed", "schedule_id": schedule_id})

        logger.debug(f"Marked schedule executed: {schedule_id}")

    def check_budget(self, schedule: Schedule) -> bool:
        """Check if schedule has budget remaining for today.

        Args:
            schedule: Schedule to check

        Returns:
            True if budget available, False if exhausted
        """
        now = datetime.now(timezone.utc)
        last_reset = (
            datetime.fromisoformat(schedule.run_budget.last_reset)
            if schedule.run_budget.last_reset
            else now
        )

        # Reset if new day
        if now.date() > last_reset.date():
            return True  # Budget will reset on execution

        return schedule.run_budget.consumed < schedule.run_budget.per_day


def create_schedule_service(schedules_dir: str = "var/schedules") -> ScheduleService:
    """Factory function to create ScheduleService.

    Args:
        schedules_dir: Directory for schedule persistence

    Returns:
        Initialized ScheduleService
    """
    return ScheduleService(schedules_dir=schedules_dir)
