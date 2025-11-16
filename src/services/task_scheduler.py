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
from src.services.decision_framework import Parameter
from src.services.task_graph import TaskGraph, TaskNode, TaskState, DependencyPolicy
from src.services.task_executor import TaskExecutor, TaskStatus as ExecutorTaskStatus

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of scheduled tasks."""
    # Cognitive tasks (LLM-based)
    SELF_REFLECTION = "self_reflection"
    GOAL_ASSESSMENT = "goal_assessment"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    CAPABILITY_EXPLORATION = "capability_exploration"
    EMOTIONAL_RECONCILIATION = "emotional_reconciliation"
    CUSTOM = "custom"

    # Code access tasks (file operations)
    CODE_READ = "code_read"          # Read source files
    CODE_ANALYZE = "code_analyze"    # Analyze code patterns
    CODE_MODIFY = "code_modify"      # Modify source files
    CODE_TEST = "code_test"          # Run tests


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

    def __init__(
        self,
        persona_space_path: str = "persona_space",
        raw_store: Optional[RawStore] = None,
        abort_monitor: Optional[Any] = None,
        decision_framework: Optional[Any] = None,
        parameter_adapter: Optional[Any] = None,
        task_executor: Optional[TaskExecutor] = None,
        goal_store: Optional[Any] = None,
        code_access_service: Optional[Any] = None
    ):
        """
        Initialize task scheduler.

        Args:
            persona_space_path: Path to persona's file space
            raw_store: Optional RawStore for creating task execution experiences
            abort_monitor: Optional AbortConditionMonitor for safety checks
            decision_framework: Optional DecisionFramework for adaptive task selection
            parameter_adapter: Optional ParameterAdapter for learning from outcomes
            task_executor: Optional TaskExecutor for robust task execution (created if not provided)
            goal_store: Optional GoalStore for goal-driven task selection (Phase 1)
            code_access_service: Optional CodeAccessService for code modification tasks
        """
        self.persona_space = Path(persona_space_path)
        self.tasks_dir = self.persona_space / "tasks"
        self.tasks_config_path = self.tasks_dir / "tasks.json"
        self.results_dir = self.tasks_dir / "results"
        self.raw_store = raw_store
        self.abort_monitor = abort_monitor
        self.decision_framework = decision_framework
        self.code_access_service = code_access_service
        self.parameter_adapter = parameter_adapter
        self.goal_store = goal_store

        # Create TaskExecutor for robust execution (Phase 2)
        if task_executor is None:
            task_executor = TaskExecutor(
                abort_monitor=abort_monitor,
                decision_registry=decision_framework.registry if decision_framework else None,
                raw_store=raw_store
            )
        self.task_executor = task_executor

        # Ensure directories exist
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load tasks
        self.tasks: Dict[str, TaskDefinition] = {}
        self._load_tasks()

        # Create default tasks if none exist
        if not self.tasks:
            self._create_default_tasks()

        # Register task selection decision point
        self._register_decision_point()

        # Track adaptations
        self.executions_since_adaptation = 0
        self.adaptation_interval = 10  # Adapt every 10 task executions

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

    def _register_decision_point(self):
        """Register task_selected decision point with DecisionFramework."""
        if not self.decision_framework:
            return

        # Define parameters for task selection decisions
        parameters = {
            "urgency_threshold": Parameter(
                name="urgency_threshold",
                current_value=0.7,
                min_value=0.3,
                max_value=0.95,
                step_size=0.05,
                adaptation_rate=0.15
            ),
            "coherence_required": Parameter(
                name="coherence_required",
                current_value=0.6,
                min_value=0.4,
                max_value=0.9,
                step_size=0.05,
                adaptation_rate=0.1
            ),
        }

        self.decision_framework.registry.register_decision(
            decision_id="task_selected",
            subsystem="task_scheduler",
            description="Autonomous task selection and execution",
            parameters=parameters,
            success_metrics=["coherence_delta", "dissonance_delta", "satisfaction_score"],
            context_features=["task_type", "time_since_last_run", "current_coherence"]
        )
        logger.info("Registered task_selected decision point")

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

    async def _execute_code_task(
        self,
        task: TaskDefinition,
        trace_id: str
    ) -> tuple[str, dict]:
        """Execute a code access task (CODE_READ, CODE_MODIFY, CODE_TEST).

        Args:
            task: The task to execute
            trace_id: Trace ID for logging

        Returns:
            (response_text, metadata) - Task result and metadata

        Raises:
            ValueError: If code_access_service not available or task params invalid
        """
        if not self.code_access_service:
            raise ValueError("CodeAccessService not available - cannot execute code tasks")

        # Extract task parameters from prompt
        # For now, prompt should be a dict-like string or actual dict in metadata
        # TODO: Better parameter extraction
        metadata = task.metadata or {}

        if task.type == TaskType.CODE_READ:
            # Read a source file
            file_path = metadata.get("file_path")
            if not file_path:
                raise ValueError("CODE_READ task requires 'file_path' in metadata")

            content, error = await self.code_access_service.read_file(file_path)
            if error:
                raise ValueError(f"Failed to read {file_path}: {error}")

            return content, {
                "file_path": file_path,
                "file_size": len(content),
                "task_type": "code_read",
            }

        elif task.type == TaskType.CODE_TEST:
            # Run tests
            test_pattern = metadata.get("test_pattern")

            test_result = await self.code_access_service.run_tests(test_pattern)

            response = f"""Test Results:
- Passed: {test_result.passed}
- Total: {test_result.total_tests}
- Passed: {test_result.passed_tests}
- Failed: {test_result.failed_tests}
- Duration: {test_result.duration_seconds:.2f}s

Output:
{test_result.output[:500]}
"""

            return response, {
                "test_passed": test_result.passed,
                "total_tests": test_result.total_tests,
                "task_type": "code_test",
            }

        elif task.type == TaskType.CODE_MODIFY:
            # Modify a source file
            file_path = metadata.get("file_path")
            new_content = metadata.get("new_content")
            reason = metadata.get("reason", "Code modification from task")
            goal_id = metadata.get("goal_id", "unknown")

            if not file_path or not new_content:
                raise ValueError("CODE_MODIFY task requires 'file_path' and 'new_content' in metadata")

            modification, error = await self.code_access_service.modify_file(
                file_path=file_path,
                new_content=new_content,
                reason=reason,
                goal_id=goal_id,
            )

            if error:
                raise ValueError(f"Failed to modify {file_path}: {error}")

            # Commit the modification
            commit_success, commit_error = await self.code_access_service.commit_modification(modification)
            if not commit_success:
                raise ValueError(f"Failed to commit modification: {commit_error}")

            response = f"""Code Modified:
- File: {file_path}
- Modification ID: {modification.id}
- Branch: {modification.branch_name}
- Status: {modification.status.value}

Reason: {reason}

Next steps:
1. Run tests: CODE_TEST task
2. Request approval
3. Merge to main
"""

            return response, {
                "modification_id": modification.id,
                "branch_name": modification.branch_name,
                "file_path": file_path,
                "task_type": "code_modify",
            }

        else:
            raise ValueError(f"Unknown code task type: {task.type}")

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

        # Safety check: Abort if dangerous conditions detected
        if self.abort_monitor:
            should_abort, abort_reason = self.abort_monitor.check_abort_conditions()
            if should_abort:
                error_msg = f"Task execution aborted due to safety condition: {abort_reason}"
                logger.warning(error_msg)
                print(f"⚠️  {error_msg}")

                # Return failed result
                now = datetime.now(timezone.utc)
                return TaskResult(
                    task_id=task.id,
                    task_name=task.name,
                    started_at=now.isoformat(),
                    completed_at=now.isoformat(),
                    success=False,
                    error=error_msg,
                    metadata={
                        "abort_reason": abort_reason,
                        "safety_aborted": True
                    }
                )

        # Ensure UTC-aware timestamps
        started_at = datetime.now(timezone.utc)

        # Generate correlation IDs
        trace_id = str(uuid4())
        span_id = str(uuid4())

        # Record task selection decision
        decision_record_id = None
        if self.decision_framework:
            context = {
                "task_type": task.type,
                "task_id": task.id,
                "time_since_last_run": (
                    (started_at - datetime.fromisoformat(task.last_run)).total_seconds() / 3600
                    if task.last_run else None
                ),
                "trace_id": trace_id
            }
            params = self.decision_framework.registry.get_all_parameters("task_selected")
            if params:
                decision_record_id = self.decision_framework.registry.record_decision(
                    decision_id="task_selected",
                    context=context,
                    parameters_used=params,
                    outcome_snapshot={}
                )
                logger.debug(f"Recorded task selection decision: {decision_record_id}")

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
            # Route task based on type
            is_code_task = task.type in (
                TaskType.CODE_READ,
                TaskType.CODE_MODIFY,
                TaskType.CODE_TEST,
                TaskType.CODE_ANALYZE,
            )

            if is_code_task:
                # Execute code access task
                response_text, task_metadata = await self._execute_code_task(task, trace_id)
                completed_at = datetime.now(timezone.utc)
                retrieval_metadata = {
                    "memory_count": 0,
                    "query": task.prompt[:100] if task.prompt else "",
                    "filters": {},
                    "latency_ms": 0,
                    "source": ["code_access"],
                }
                reconciliation = None
            else:
                # Execute cognitive task by generating response with persona
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
                task_metadata = {}

            # Create result
            result_metadata = {
                "reconciliation": reconciliation,
                "task_type": task.type,
                "trace_id": trace_id,
                "span_id": span_id,
                "decision_record_id": decision_record_id,  # For linking to DecisionFramework
            }
            # Merge task-specific metadata
            result_metadata.update(task_metadata)

            result = TaskResult(
                task_id=task.id,
                task_name=task.name,
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
                success=True,
                response=response_text,
                metadata=result_metadata
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

            # Increment execution counter and trigger adaptation if needed
            self.executions_since_adaptation += 1
            self.trigger_parameter_adaptation()

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
                    "decision_record_id": decision_record_id,  # For linking to DecisionFramework
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

    async def execute_task_with_retry(
        self,
        task_id: str,
        persona_service,
        max_retries: Optional[int] = None,
        timeout_ms: Optional[int] = None
    ) -> TaskResult:
        """
        Execute task with retry logic using TaskExecutor (Phase 2).

        This is a wrapper around the original execute_task that adds:
        - Automatic retry on transient failures
        - Exponential backoff with jitter
        - Timeout enforcement
        - Circuit breaker integration

        Args:
            task_id: ID of task to execute
            persona_service: PersonaService for generating responses
            max_retries: Override default retry limit
            timeout_ms: Override default timeout

        Returns:
            TaskResult with execution details
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task not found: {task_id}")

        task_def = self.tasks[task_id]

        # Create TaskNode from TaskDefinition
        node = TaskNode(
            task_id=task_def.id,
            action_name=task_def.type.value,
            normalized_args={"prompt": task_def.prompt, "task_name": task_def.name},
            resource_ids=[],
            version="1.0",
            max_retries=max_retries or 3,
            task_timeout_ms=timeout_ms
        )

        # Define task callable
        async def task_callable(args):
            # Call original execute_task
            result = await self.execute_task(task_id, persona_service)
            return {
                "success": result.success,
                "error": result.error,
                "result": result
            }

        # Execute with TaskExecutor
        exec_result = await self.task_executor.execute(
            node=node,
            task_callable=task_callable,
            timeout_ms=timeout_ms
        )

        # Convert executor result to TaskResult
        if exec_result.status == ExecutorTaskStatus.DUPLICATE:
            # Return cached result if available
            if exec_result.metadata and "result" in exec_result.metadata:
                return exec_result.metadata["result"]

        # Extract original TaskResult from metadata
        if exec_result.success and exec_result.metadata and "result" in exec_result.metadata:
            return exec_result.metadata["result"]

        # Failed - create error TaskResult
        now = datetime.now(timezone.utc)
        return TaskResult(
            task_id=task_def.id,
            task_name=task_def.name,
            started_at=now.isoformat(),
            completed_at=now.isoformat(),
            success=False,
            error=exec_result.error,
            metadata={
                "executor_status": exec_result.status.value,
                "error_class": exec_result.error_class.value if exec_result.error_class else None,
                "execution_time_ms": exec_result.execution_time_ms
            }
        )

    async def execute_graph(
        self,
        graph: TaskGraph,
        persona_service,
        max_parallel: int = 5,
        per_action_caps: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Execute task graph with parallel execution and dependency tracking (Phase 2).

        Args:
            graph: TaskGraph with tasks and dependencies
            persona_service: PersonaService for generating responses
            max_parallel: Maximum parallel tasks
            per_action_caps: Per-action concurrency limits

        Returns:
            Dict with execution statistics and results
        """
        logger.info(f"Executing task graph: {graph.graph_id} ({len(graph.nodes)} tasks)")

        results = {}
        start_time = datetime.now(timezone.utc)

        # Save original max_parallel and update if specified
        original_max_parallel = graph.max_parallel
        if max_parallel is not None:
            graph.max_parallel = max_parallel

        # Execute until all tasks complete
        while not graph.is_complete():
            # Get ready tasks
            ready_task_ids = graph.get_ready_tasks(per_action_caps=per_action_caps)

            if not ready_task_ids:
                # No ready tasks - either waiting for dependencies or at concurrency limit
                # Wait a bit for running tasks to complete
                await asyncio.sleep(0.05)
                continue

            # Limit to max_parallel
            ready_task_ids = ready_task_ids[:max_parallel]

            # Execute ready tasks in parallel
            tasks = []
            for task_id in ready_task_ids:
                graph.mark_running(task_id)
                node = graph.nodes[task_id]

                # Create async task for execution
                tasks.append(self._execute_graph_node(node, persona_service, graph))

            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(batch_results):
                task_id = ready_task_ids[i]
                node = graph.nodes[task_id]

                if isinstance(result, Exception):
                    # Execution failed
                    logger.error(f"Task {task_id} failed: {result}")
                    graph.mark_completed(task_id, success=False, error=str(result))
                    results[task_id] = {"success": False, "error": str(result)}
                else:
                    # Execution succeeded
                    graph.mark_completed(task_id, success=result.get("success", True))
                    results[task_id] = result

        end_time = datetime.now(timezone.utc)
        elapsed_ms = (end_time - start_time).total_seconds() * 1000

        # Restore original max_parallel
        graph.max_parallel = original_max_parallel

        # Get final statistics
        stats = graph.get_stats()

        return {
            "graph_id": graph.graph_id,
            "total_tasks": len(graph.nodes),
            "results": results,
            "statistics": stats,
            "elapsed_ms": elapsed_ms,
            "completed_at": end_time.isoformat()
        }

    async def _execute_graph_node(
        self,
        node: TaskNode,
        persona_service,
        graph: TaskGraph
    ) -> Dict[str, Any]:
        """Execute a single node in the task graph."""
        # Define task callable
        async def task_callable(args):
            # For graph execution, args contain the prompt
            prompt = args.get("prompt", "")

            # Generate response using persona service
            # Handle both sync and async generate_response
            result = persona_service.generate_response(
                user_message=prompt,
                retrieve_memories=True,
                top_k=10
            )

            # If result is a coroutine, await it
            if asyncio.iscoroutine(result):
                response, reconciliation = await result
            else:
                response, reconciliation = result

            return {
                "success": True,
                "response": response,
                "reconciliation": reconciliation
            }

        # Execute with TaskExecutor
        exec_result = await self.task_executor.execute(
            node=node,
            task_callable=task_callable
        )

        return {
            "success": exec_result.success,
            "error": exec_result.error,
            "execution_time_ms": exec_result.execution_time_ms,
            "metadata": exec_result.metadata
        }

    def trigger_parameter_adaptation(self, force: bool = False) -> Optional[Dict]:
        """
        Trigger parameter adaptation from task outcomes.

        Args:
            force: If True, adapt regardless of interval

        Returns:
            Adaptation results dict, or None if skipped
        """
        if not self.parameter_adapter or not self.decision_framework:
            return None

        # Check if adaptation should run
        if not force and self.executions_since_adaptation < self.adaptation_interval:
            logger.debug(
                f"Skipping adaptation: {self.executions_since_adaptation}/{self.adaptation_interval}"
            )
            return None

        logger.info("Triggering parameter adaptation from task outcomes")

        # Adapt task_selected decision parameters
        result = self.parameter_adapter.adapt_from_evaluated_decisions(
            decision_id="task_selected",
            since_hours=48,  # Look at last 48 hours of task executions
            dry_run=False
        )

        # Reset counter
        self.executions_since_adaptation = 0

        logger.info(f"Parameter adaptation result: {result}")
        return result

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

    # === Goal-Driven Task Selection (Phase 1 Integration) ===

    def _goal_to_task(self, goal) -> TaskDefinition:
        """Convert a GoalDefinition to a TaskDefinition for execution.

        Args:
            goal: GoalDefinition from GoalStore

        Returns:
            TaskDefinition that can be executed by TaskScheduler
        """
        from src.services.goal_store import GoalCategory

        # Map goal category to task type
        category_to_type = {
            GoalCategory.INTROSPECTION: TaskType.SELF_REFLECTION,
            GoalCategory.EXPLORATION: TaskType.CAPABILITY_EXPLORATION,
            GoalCategory.MAINTENANCE: TaskType.CUSTOM,
            GoalCategory.USER_REQUESTED: TaskType.CUSTOM
        }

        task_type = category_to_type.get(goal.category, TaskType.CUSTOM)

        return TaskDefinition(
            id=goal.id,
            name=goal.text[:100],  # Truncate long goal text for name
            type=task_type,
            schedule=TaskSchedule.MANUAL,  # Goals are executed on-demand
            prompt=goal.text,
            enabled=True,
            metadata={
                "goal_id": goal.id,
                "goal_value": goal.value,
                "goal_effort": goal.effort,
                "goal_risk": goal.risk,
                "goal_category": goal.category.value,
                "aligns_with": goal.aligns_with,
                "success_metrics": goal.success_metrics,
                "source": "goal_store"
            }
        )

    def get_next_goal(self, active_belief_ids: Optional[List[str]] = None) -> Optional[Any]:
        """Get the highest priority goal ready for execution.

        This method:
        1. Queries GoalStore for ADOPTED goals
        2. Scores them using current adaptive weights
        3. Returns the highest priority goal

        Args:
            active_belief_ids: List of currently active belief IDs for scoring

        Returns:
            GoalDefinition if available, None otherwise
        """
        if not self.goal_store:
            return None

        from src.services.goal_store import GoalState

        try:
            # Get current weights from DecisionRegistry if available
            weights = {}
            if self.decision_framework and hasattr(self.decision_framework, 'registry'):
                try:
                    weights = self.decision_framework.registry.get_all_parameters("goal_selected") or {}
                except:
                    # Use defaults if registry lookup fails
                    weights = {
                        "value_weight": 0.5,
                        "effort_weight": 0.25,
                        "risk_weight": 0.15,
                        "urgency_weight": 0.05,
                        "alignment_weight": 0.05
                    }

            # Get prioritized ADOPTED goals
            prioritized = self.goal_store.prioritized(
                state=GoalState.ADOPTED,
                limit=10,
                weights=weights,
                active_beliefs=active_belief_ids or []
            )

            if not prioritized:
                return None

            # Return highest priority goal
            goal, score = prioritized[0]
            logger.info(f"Selected goal '{goal.id}' with priority score {score:.3f}")
            return goal

        except Exception as e:
            logger.error(f"Failed to get next goal: {e}", exc_info=True)
            return None

    async def execute_goal(
        self,
        goal_id: str,
        persona_service,
        active_belief_ids: Optional[List[str]] = None
    ) -> TaskResult:
        """Execute a specific goal by ID.

        This is a convenience method that:
        1. Fetches the goal from GoalStore
        2. Converts it to a TaskDefinition
        3. Executes it via execute_task()

        Args:
            goal_id: ID of the goal to execute
            persona_service: Service to generate responses
            active_belief_ids: Currently active belief IDs

        Returns:
            TaskResult
        """
        if not self.goal_store:
            raise ValueError("GoalStore not configured")

        # Fetch goal
        goal = self.goal_store.get_goal(goal_id)
        if not goal:
            raise ValueError(f"Goal not found: {goal_id}")

        # Convert to task
        task = self._goal_to_task(goal)

        # Add task to tasks dictionary temporarily (required for execute_task)
        self.tasks[task.id] = task

        try:
            # Execute via standard task execution
            result = await self.execute_task(task.id, persona_service)
        finally:
            # Remove temporary task (goals aren't persisted in tasks.json)
            if task.id in self.tasks:
                del self.tasks[task.id]

        # Update goal state based on result
        try:
            from src.services.goal_store import GoalState

            if result.success:
                # Mark goal as SATISFIED
                self.goal_store.update_goal(
                    goal_id,
                    {"state": GoalState.SATISFIED},
                    expected_version=goal.version
                )
                logger.info(f"Goal {goal_id} marked as SATISFIED")
            else:
                # Could implement retry logic or mark as failed
                logger.warning(f"Goal {goal_id} execution failed: {result.error}")

        except Exception as e:
            logger.error(f"Failed to update goal state: {e}")

        return result


def create_task_scheduler(
    persona_space_path: str = "persona_space",
    raw_store: Optional[RawStore] = None,
    abort_monitor: Optional[Any] = None,
    decision_framework: Optional[Any] = None,
    parameter_adapter: Optional[Any] = None,
    task_executor: Optional[TaskExecutor] = None,
    goal_store: Optional[Any] = None
) -> TaskScheduler:
    """
    Factory function to create a TaskScheduler.

    Args:
        persona_space_path: Path to persona_space directory
        raw_store: Optional RawStore for task execution tracking
        abort_monitor: Optional AbortConditionMonitor for safety checks
        decision_framework: Optional DecisionFramework for adaptive task selection
        parameter_adapter: Optional ParameterAdapter for learning from outcomes
        task_executor: Optional TaskExecutor for robust task execution with retry logic
        goal_store: Optional GoalStore for goal-driven task selection (Phase 1)

    Returns:
        TaskScheduler instance
    """
    return TaskScheduler(
        persona_space_path,
        raw_store=raw_store,
        abort_monitor=abort_monitor,
        decision_framework=decision_framework,
        parameter_adapter=parameter_adapter,
        task_executor=task_executor,
        goal_store=goal_store
    )
