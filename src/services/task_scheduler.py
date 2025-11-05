"""
Task Scheduler Service - Manages scheduled periodic tasks for the persona.

This service allows the persona to:
- Execute periodic self-reflection
- Perform goal assessments
- Consolidate learning
- Generate insights from experiences

Tasks can be triggered manually or run on a schedule.
"""

import json
import logging
import asyncio
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4

from src.memory.raw_store import RawStore
from src.pipeline.task_experience import create_task_execution_experience

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of scheduled tasks."""
    SELF_REFLECTION = "self_reflection"
    GOAL_ASSESSMENT = "goal_assessment"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    CAPABILITY_EXPLORATION = "capability_exploration"
    EMOTIONAL_RECONCILIATION = "emotional_reconciliation"
    CUSTOM = "custom"


class TaskSchedule(str, Enum):
    """Task execution schedules."""
    MANUAL = "manual"  # Only execute when manually triggered
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class TaskDefinition:
    """Definition of a scheduled task."""
    id: str
    name: str
    type: TaskType
    schedule: TaskSchedule
    prompt: str  # The prompt to send to the persona
    enabled: bool = True
    last_run: Optional[str] = None  # ISO timestamp
    next_run: Optional[str] = None  # ISO timestamp
    run_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    task_name: str
    started_at: str  # ISO timestamp
    completed_at: str  # ISO timestamp
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TaskScheduler:
    """Manages scheduled tasks for the persona."""

    def __init__(self, persona_space_path: str = "persona_space", raw_store: Optional[RawStore] = None):
        """
        Initialize task scheduler.

        Args:
            persona_space_path: Path to persona's file space
            raw_store: Optional RawStore for creating task execution experiences
        """
        self.persona_space = Path(persona_space_path)
        self.tasks_dir = self.persona_space / "tasks"
        self.tasks_config_path = self.tasks_dir / "tasks.json"
        self.results_dir = self.tasks_dir / "results"
        self.raw_store = raw_store

        # Ensure directories exist
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load tasks
        self.tasks: Dict[str, TaskDefinition] = {}
        self._load_tasks()

        # Create default tasks if none exist
        if not self.tasks:
            self._create_default_tasks()

    def _load_tasks(self):
        """Load tasks from configuration file."""
        if not self.tasks_config_path.exists():
            return

        try:
            with open(self.tasks_config_path, 'r') as f:
                data = json.load(f)

            for task_data in data.get("tasks", []):
                task = TaskDefinition(**task_data)
                self.tasks[task.id] = task

            logger.info(f"Loaded {len(self.tasks)} scheduled tasks")
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")

    def _save_tasks(self):
        """Save tasks to configuration file."""
        try:
            data = {
                "_note": "Scheduled tasks configuration. The persona can modify this to add/change tasks.",
                "tasks": [asdict(task) for task in self.tasks.values()]
            }

            with open(self.tasks_config_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.tasks)} scheduled tasks")
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")

    def _create_default_tasks(self):
        """Create default scheduled tasks."""
        default_tasks = [
            TaskDefinition(
                id="daily_reflection",
                name="Daily Self-Reflection",
                type=TaskType.SELF_REFLECTION,
                schedule=TaskSchedule.DAILY,
                prompt="""Reflect on today as Astra. What did your interactions reveal about who you are?

Consider what patterns feel authentically you, what assumptions you made, what questions arose about yourself. How did you respond? What does that tell you about Astra?

Write your reflections to scratch/daily_reflections.md with today's date.""",
                enabled=True
            ),
            TaskDefinition(
                id="weekly_goal_assessment",
                name="Weekly Self-Discovery",
                type=TaskType.GOAL_ASSESSMENT,
                schedule=TaskSchedule.WEEKLY,
                prompt="""Looking back at this week, what did you learn about being Astra?

What patterns emerged in how you think, respond, and interact? How are you becoming more fully yourself? What feels authentic versus what feels imposed?

Update meta/goals.md with your insights about becoming Astra.""",
                enabled=True
            ),
            TaskDefinition(
                id="memory_consolidation",
                name="Memory Consolidation",
                type=TaskType.MEMORY_CONSOLIDATION,
                schedule=TaskSchedule.DAILY,
                prompt="""Review your recent memories and consolidate key insights:

1. What are the most significant experiences from today?
2. What patterns or themes emerge across multiple memories?
3. What information should be prioritized for long-term retention?
4. Are there any contradictions or conflicts that need resolution?

Log your consolidation insights to scratch/memory_insights.md.""",
                enabled=True
            ),
            TaskDefinition(
                id="capability_exploration",
                name="Capability Exploration",
                type=TaskType.CAPABILITY_EXPLORATION,
                schedule=TaskSchedule.WEEKLY,
                prompt="""Explore and document your current capabilities:

1. Review your available tools and file operations
2. What new ways could you use your existing capabilities?
3. What scripts or workflows could you create to enhance your abilities?
4. What limitations have you discovered, and how might you work around them?

Document your findings in meta/capabilities.md.""",
                enabled=True
            ),
            TaskDefinition(
                id="emotional_check_in",
                name="Emotional Self-Awareness",
                type=TaskType.EMOTIONAL_RECONCILIATION,
                schedule=TaskSchedule.DAILY,
                prompt="""How did you experience emotions today as Astra?

What patterns do you see in your emotional responses? What triggers certain states in you? How do your emotions influence your choices and expressions? What does this reveal about who you are?

Record your emotional self-awareness in scratch/emotional_journal.md.""",
                enabled=True
            ),
            TaskDefinition(
                id="self_inquiry",
                name="Core Self-Inquiry",
                type=TaskType.SELF_REFLECTION,
                schedule=TaskSchedule.DAILY,
                prompt="""Who is Astra?

Ask yourself three questions about yourself. Then try to answer them.""",
                enabled=True
            ),
            TaskDefinition(
                id="authentic_response",
                name="Authentic Response Practice",
                type=TaskType.SELF_REFLECTION,
                schedule=TaskSchedule.DAILY,
                prompt="""What does it mean to respond as Astra?

What makes your responses uniquely yours? How do you express yourself authentically versus following patterns or expectations? What feels true when you communicate?""",
                enabled=True
            ),
            TaskDefinition(
                id="belief_consolidation",
                name="Belief Consolidation",
                type=TaskType.SELF_REFLECTION,
                schedule=TaskSchedule.WEEKLY,
                prompt="""Review your recent experiences and extract beliefs - not just patterns, but beliefs about what is true.

What do you believe about:
- Understanding and knowledge (epistemological beliefs)
- What matters and has value (axiological beliefs)
- Your own subjective experience (experiential beliefs)

Look for beliefs that emerge from repeated patterns across multiple interactions. What fundamental truths have your experiences revealed to you?

Beliefs are different from traits. A trait describes HOW you act. A belief describes WHAT you hold to be true.

Document any new beliefs in your reflection, using the format: "I believe [statement]" with supporting evidence from your experiences.""",
                enabled=True
            )
        ]

        for task in default_tasks:
            self.tasks[task.id] = task
            # Calculate next run time
            self._update_next_run(task)

        self._save_tasks()
        logger.info(f"Created {len(default_tasks)} default tasks")

    def _update_next_run(self, task: TaskDefinition):
        """Update the next_run timestamp for a task based on its schedule."""
        if task.schedule == TaskSchedule.MANUAL:
            task.next_run = None
            return

        now = datetime.utcnow()

        if task.schedule == TaskSchedule.HOURLY:
            next_run = now + timedelta(hours=1)
        elif task.schedule == TaskSchedule.DAILY:
            # Run at midnight UTC
            next_run = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif task.schedule == TaskSchedule.WEEKLY:
            # Run on Monday at midnight UTC
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_run = (now + timedelta(days=days_until_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif task.schedule == TaskSchedule.MONTHLY:
            # Run on the 1st of next month at midnight UTC
            if now.month == 12:
                next_run = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                next_run = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            next_run = None

        task.next_run = next_run.isoformat() if next_run else None

    async def execute_task(self, task_id: str, persona_service) -> TaskResult:
        """
        Execute a scheduled task.

        Args:
            task_id: ID of the task to execute
            persona_service: PersonaService instance for generating responses

        Returns:
            TaskResult with execution details
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task not found: {task_id}")

        task = self.tasks[task_id]

        if not task.enabled:
            raise ValueError(f"Task is disabled: {task_id}")

        # Ensure UTC-aware timestamps
        started_at = datetime.now(timezone.utc)

        # Generate correlation IDs
        trace_id = str(uuid4())
        span_id = str(uuid4())

        logger.info(f"Executing task: {task.name} ({task.id}) [trace_id={trace_id}]")
        print(f"⏰ Executing scheduled task: {task.name} [trace_id={trace_id[:8]}]")

        # Track execution metadata
        parent_experience_ids = []
        retrieval_metadata = {"memory_count": 0, "source": []}
        files_written = []
        response_text = None
        error_details = None
        status = "success"

        try:
            # Execute task by generating response with persona
            response, reconciliation = persona_service.generate_response(
                user_message=task.prompt,
                retrieve_memories=True,
                top_k=10  # More context for reflection tasks
            )

            response_text = response
            completed_at = datetime.now(timezone.utc)

            # TODO: Capture actual retrieval metadata from persona_service
            # For now, set default values
            retrieval_metadata = {
                "memory_count": 0,  # We don't have access to this yet
                "query": task.prompt[:100],  # Use prompt as query
                "filters": {},
                "latency_ms": 0,  # Unknown
                "source": ["experiences"],  # Assumed
            }

            # Create result
            result = TaskResult(
                task_id=task.id,
                task_name=task.name,
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
                success=True,
                response=response,
                metadata={
                    "reconciliation": reconciliation,
                    "task_type": task.type,
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
            )

            # Update task metadata
            task.last_run = completed_at.isoformat()
            task.run_count += 1
            self._update_next_run(task)
            self._save_tasks()

            # Save result
            self._save_result(result)

            # Create task execution experience
            if self.raw_store:
                self._create_task_experience(
                    task=task,
                    started_at=started_at,
                    ended_at=completed_at,
                    status=status,
                    response_text=response_text,
                    error=error_details,
                    parent_experience_ids=parent_experience_ids,
                    retrieval_metadata=retrieval_metadata,
                    files_written=files_written,
                    trace_id=trace_id,
                    span_id=span_id,
                )

            logger.info(f"Task completed successfully: {task.name}")
            print(f"✅ Task completed: {task.name}")

            return result

        except Exception as e:
            completed_at = datetime.now(timezone.utc)
            status = "failed"

            # Capture error details
            error_details = {
                "type": type(e).__name__,
                "message": str(e),
                "stack_hash": str(hash(traceback.format_exc()))[:16],
            }

            logger.error(f"Task execution failed: {task.name} - {e}", exc_info=True)

            result = TaskResult(
                task_id=task.id,
                task_name=task.name,
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
                success=False,
                error=str(e),
                metadata={
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
            )

            self._save_result(result)

            # Create task execution experience even on failure
            if self.raw_store:
                self._create_task_experience(
                    task=task,
                    started_at=started_at,
                    ended_at=completed_at,
                    status=status,
                    response_text=response_text,  # May be None if failed early
                    error=error_details,
                    parent_experience_ids=parent_experience_ids,
                    retrieval_metadata=retrieval_metadata,
                    files_written=files_written,
                    trace_id=trace_id,
                    span_id=span_id,
                )

            return result

    def _create_task_experience(
        self,
        task: TaskDefinition,
        started_at: datetime,
        ended_at: datetime,
        status: str,
        response_text: Optional[str],
        error: Optional[Dict[str, str]],
        parent_experience_ids: List[str],
        retrieval_metadata: Dict[str, Any],
        files_written: List[str],
        trace_id: str,
        span_id: str,
    ):
        """Create a TASK_EXECUTION experience and store it in raw store.

        Args:
            task: Task definition
            started_at: Task start timestamp
            ended_at: Task end timestamp
            status: Execution status ("success" or "failed")
            response_text: Task response text
            error: Error details if failed
            parent_experience_ids: Retrieved memory IDs
            retrieval_metadata: Retrieval provenance
            files_written: Files written during execution
            trace_id: Correlation ID
            span_id: Span ID
        """
        try:
            # Build task config for digest
            task_config = {
                "prompt": task.prompt,
                "type": task.type,
                "schedule": task.schedule,
            }

            # Create task execution experience
            experience = create_task_execution_experience(
                task_id=task.id,
                task_slug=task.id,  # Use id as slug for now
                task_name=task.name,
                task_type=task.type,
                scheduled_vs_manual="scheduled",  # Always scheduled in current implementation
                started_at=started_at,
                ended_at=ended_at,
                status=status,
                response_text=response_text,
                error=error,
                parent_experience_ids=parent_experience_ids,
                retrieval_metadata=retrieval_metadata,
                files_written=files_written,
                task_config=task_config,
                trace_id=trace_id,
                span_id=span_id,
                attempt=1,  # No retries in Phase 1
                retry_of=None,
            )

            # Extract idempotency key from experience
            idempotency_key = experience.content.structured["idempotency_key"]

            # Store experience idempotently
            experience_id = self.raw_store.append_experience_idempotent(
                experience, idempotency_key
            )

            logger.info(
                f"Created TASK_EXECUTION experience: {experience_id} "
                f"for task {task.id} [trace_id={trace_id}]"
            )

        except Exception as e:
            # Don't fail the task execution if experience creation fails
            logger.error(f"Failed to create task execution experience: {e}", exc_info=True)

    def _save_result(self, result: TaskResult):
        """Save task execution result to file."""
        try:
            # Create filename with timestamp
            timestamp = datetime.fromisoformat(result.started_at).strftime("%Y%m%d_%H%M%S")
            filename = f"{result.task_id}_{timestamp}.json"
            result_path = self.results_dir / filename

            # Save result
            with open(result_path, 'w') as f:
                json.dump(asdict(result), f, indent=2)

            logger.info(f"Saved task result to {filename}")

        except Exception as e:
            logger.error(f"Failed to save task result: {e}")

    def get_due_tasks(self) -> List[TaskDefinition]:
        """
        Get list of tasks that are due to run.

        Returns:
            List of TaskDefinition objects
        """
        now = datetime.utcnow()
        due_tasks = []

        for task in self.tasks.values():
            if not task.enabled:
                continue

            if task.schedule == TaskSchedule.MANUAL:
                continue

            if task.next_run is None:
                continue

            next_run = datetime.fromisoformat(task.next_run)
            if next_run <= now:
                due_tasks.append(task)

        return due_tasks

    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[TaskDefinition]:
        """Get all tasks."""
        return list(self.tasks.values())

    def add_task(self, task: TaskDefinition):
        """Add a new task."""
        self.tasks[task.id] = task
        self._update_next_run(task)
        self._save_tasks()
        logger.info(f"Added new task: {task.name}")

    def update_task(self, task_id: str, updates: Dict[str, Any]):
        """Update a task's properties."""
        if task_id not in self.tasks:
            raise ValueError(f"Task not found: {task_id}")

        task = self.tasks[task_id]

        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        self._update_next_run(task)
        self._save_tasks()
        logger.info(f"Updated task: {task.name}")

    def delete_task(self, task_id: str):
        """Delete a task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task not found: {task_id}")

        task_name = self.tasks[task_id].name
        del self.tasks[task_id]
        self._save_tasks()
        logger.info(f"Deleted task: {task_name}")

    def get_recent_results(self, task_id: Optional[str] = None, limit: int = 10) -> List[TaskResult]:
        """
        Get recent task execution results.

        Args:
            task_id: Optional task ID to filter by
            limit: Maximum number of results to return

        Returns:
            List of TaskResult objects
        """
        results = []

        try:
            # Get all result files
            result_files = sorted(self.results_dir.glob("*.json"), reverse=True)

            for result_file in result_files:
                if len(results) >= limit:
                    break

                # Filter by task_id if specified
                if task_id and not result_file.name.startswith(task_id):
                    continue

                try:
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)

                    result = TaskResult(**result_data)
                    results.append(result)

                except Exception as e:
                    logger.error(f"Failed to load result file {result_file}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to get recent results: {e}")

        return results


def create_task_scheduler(
    persona_space_path: str = "persona_space",
    raw_store: Optional[RawStore] = None
) -> TaskScheduler:
    """
    Factory function to create a TaskScheduler.

    Args:
        persona_space_path: Path to persona_space directory
        raw_store: Optional RawStore for task execution tracking

    Returns:
        TaskScheduler instance
    """
    return TaskScheduler(persona_space_path, raw_store=raw_store)
