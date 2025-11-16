"""Goal Execution Service - Autonomous coding task execution.

Connects HTN Planning → TaskGraph → Execution Engine for end-to-end goal execution.

This service enables Astra to autonomously execute coding tasks by:
1. Decomposing high-level goals using HTN planner
2. Creating task dependency graphs
3. Executing tasks with safety hooks and retry logic
4. Returning structured results with full observability

Example:
    >>> service = GoalExecutionService(code_access, identity_ledger)
    >>> result = await service.execute_goal(
    ...     goal_text="Implement user authentication feature",
    ...     context={"codebase_path": "/src"}
    ... )
    >>> print(f"Success: {result.success}, Tasks: {len(result.completed_tasks)}")
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4

from src.services.htn_planner import HTNPlanner, Method, Plan
from src.services.task_graph import TaskGraph, TaskNode, TaskState
from src.services.task_execution_engine import TaskExecutionEngine
from src.services.project_manager import ProjectManager, ProjectMetadata
from src.services.task_executors.base import RunContext, TaskExecutor
from src.services.task_executors.code_modification import CodeModificationExecutor
from src.services.task_executors.test_runner import TestExecutor
from src.services.task_executors.shell_command import ShellCommandExecutor
from src.services.code_generator import CodeGenerator, GenRequest

logger = logging.getLogger(__name__)


# === Result Types ===

@dataclass
class TaskResult:
    """Result from a single task execution."""
    task_id: str
    task_name: str
    action_name: str
    state: str
    stdout: str = ""
    stderr: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_class: Optional[str] = None
    retry_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class GoalExecutionResult:
    """Result from executing a goal."""
    goal_id: str
    goal_text: str
    success: bool

    # Execution details
    completed_tasks: List[TaskResult] = field(default_factory=list)
    failed_tasks: List[TaskResult] = field(default_factory=list)
    total_tasks: int = 0

    # Metrics
    execution_time_ms: float = 0.0
    retry_count: int = 0

    # HTN Planning
    plan_id: Optional[str] = None
    methods_used: List[str] = field(default_factory=list)
    planning_cost: float = 0.0

    # Artifacts and errors
    artifacts: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    # Timestamps
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


# === Default HTN Methods for Coding Tasks ===

DEFAULT_CODING_METHODS = [
    Method(
        name="implement_feature_full",
        task="implement_feature",
        preconditions=[],
        subtasks=["create_file", "create_file", "run_tests"],  # Create implementation + test file
        cost=0.6,
        metadata={"description": "Implement new feature with tests"}
    ),
    Method(
        name="fix_bug_simple",
        task="fix_bug",
        preconditions=[],
        subtasks=["modify_code", "run_tests"],
        cost=0.4,
        metadata={"description": "Fix bug and verify with tests"}
    ),
    Method(
        name="refactor_code_safe",
        task="refactor_code",
        preconditions=[],
        subtasks=["modify_code", "run_tests"],
        cost=0.5,
        metadata={"description": "Refactor code while maintaining tests"}
    ),
    Method(
        name="add_tests_only",
        task="add_tests",
        preconditions=[],
        subtasks=["create_file", "run_tests"],
        cost=0.3,
        metadata={"description": "Add test coverage"}
    ),
]

# Primitive task names (executable by executors)
DEFAULT_PRIMITIVE_TASKS = {
    "create_file",
    "modify_code",
    "delete_file",
    "run_tests",
    "pytest",
    "shell_command",
    "bash"
}


# === Goal Execution Service ===

class GoalExecutionService:
    """Service for autonomous goal execution.

    Orchestrates HTN planning, task graph creation, and execution.

    Attributes:
        code_access: CodeAccessService for file operations
        identity_ledger: IdentityLedger for event tracking
        planner: HTNPlanner for goal decomposition
        executors: List of available task executors
        workdir: Working directory for task execution
    """

    def __init__(
        self,
        code_access: Any,
        code_generator: Optional[CodeGenerator] = None,
        identity_ledger: Optional[Any] = None,
        workdir: str = "/home/d/git/ai-exp",
        methods: Optional[List[Method]] = None,
        primitive_tasks: Optional[Set[str]] = None,
        max_concurrent: int = 3,
        executors: Optional[List[TaskExecutor]] = None,
        project_manager: Optional[ProjectManager] = None
    ):
        """Initialize goal execution service.

        Args:
            code_access: CodeAccessService instance
            code_generator: Optional CodeGenerator for LLM-based code generation
            identity_ledger: Optional IdentityLedger for event tracking
            workdir: Working directory for execution
            methods: HTN decomposition methods (uses defaults if None)
            primitive_tasks: Set of primitive task names (uses defaults if None)
            max_concurrent: Maximum concurrent task execution
            executors: Custom executors (uses defaults if None)
            project_manager: Optional ProjectManager for workspace projects
        """
        self.code_access = code_access
        self.code_generator = code_generator
        self.identity_ledger = identity_ledger
        self.workdir = workdir
        self.max_concurrent = max_concurrent
        self.project_manager = project_manager or ProjectManager()

        # Current execution context (set during execute_goal)
        self.current_goal_text: Optional[str] = None
        self.current_context: Dict[str, Any] = {}
        self.current_goal_id: Optional[str] = None
        self.current_project: Optional[ProjectMetadata] = None

        # Initialize HTN planner
        self.planner = HTNPlanner(
            belief_kernel=None,  # TODO: Wire belief kernel in Phase 3
            method_library=methods or DEFAULT_CODING_METHODS,
            primitive_tasks=primitive_tasks or DEFAULT_PRIMITIVE_TASKS
        )

        # Initialize executors (use provided or create defaults)
        if executors is not None:
            self.executors = executors
        else:
            self.executors = [
                CodeModificationExecutor(code_access),
                TestExecutor(),
                ShellCommandExecutor()
            ]

        logger.info(
            f"GoalExecutionService initialized: "
            f"{len(self.planner.methods)} methods, "
            f"{len(self.executors)} executors, "
            f"max_concurrent={max_concurrent}"
        )

    async def execute_goal(
        self,
        goal_text: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Any]] = None,
        timeout_ms: int = 600000  # 10 minutes default
    ) -> GoalExecutionResult:
        """Execute a high-level goal autonomously.

        Full pipeline:
        1. HTN planning: Decompose goal into primitive tasks
        2. TaskGraph creation: Build dependency graph
        3. Task execution: Run tasks with retry logic
        4. Result aggregation: Collect outcomes and artifacts

        Args:
            goal_text: High-level goal description (e.g., "implement user auth")
            context: World state for HTN precondition checking
            constraints: Additional constraint functions
            timeout_ms: Maximum execution time in milliseconds

        Returns:
            GoalExecutionResult with task outcomes, metrics, and artifacts

        Example:
            >>> result = await service.execute_goal(
            ...     goal_text="implement_feature",
            ...     context={"has_codebase": True}
            ... )
            >>> assert result.success
            >>> print(f"Completed {len(result.completed_tasks)} tasks")
        """
        goal_id = str(uuid4())
        start_time = time.monotonic()

        logger.info(f"Starting goal execution: {goal_id} - {goal_text}")

        # Create workspace project for this goal
        try:
            project = self.project_manager.create_project(
                goal_text=goal_text,
                template="python-module",
                description=f"Autonomous implementation of: {goal_text}"
            )
            self.current_project = project
            logger.info(f"Created workspace project: {project.project_id} at {project.project_dir}")
        except Exception as e:
            logger.error(f"Failed to create workspace project: {e}")
            # Fall back to tests/generated if project creation fails
            self.current_project = None

        # Store execution context for code generation
        self.current_goal_id = goal_id
        self.current_goal_text = goal_text
        self.current_context = context or {}

        result = GoalExecutionResult(
            goal_id=goal_id,
            goal_text=goal_text,
            success=False
        )

        try:
            # Step 1: HTN Planning
            plan = await self._plan_goal(goal_id, goal_text, context, constraints)

            if not plan:
                result.errors.append("HTN planning failed: no valid plan found")
                logger.error(f"Goal {goal_id}: Planning failed")
                return result

            result.plan_id = plan.plan_id
            result.methods_used = plan.methods_used
            result.planning_cost = plan.total_cost
            result.total_tasks = len(plan.tasks)

            logger.info(
                f"Goal {goal_id}: Plan created with {len(plan.tasks)} tasks "
                f"using methods {plan.methods_used}"
            )

            # Step 2: Create TaskGraph
            graph = await self._plan_to_taskgraph(plan, goal_id)

            logger.info(f"Goal {goal_id}: TaskGraph created with {len(graph.nodes)} nodes")

            # Step 3: Execute tasks
            execution_result = await self._execute_graph(graph, timeout_ms)

            # Step 4: Aggregate results
            self._populate_result(result, graph, execution_result)

            # Calculate execution time
            end_time = time.monotonic()
            result.execution_time_ms = (end_time - start_time) * 1000
            result.completed_at = datetime.now(timezone.utc)

            # Determine overall success
            result.success = (len(result.failed_tasks) == 0 and len(result.completed_tasks) > 0)

            # Update workspace project status
            if self.current_project:
                try:
                    self.project_manager.update_project(
                        project_id=self.current_project.project_id,
                        status="completed" if result.success else "failed",
                        test_results={
                            "status": "passed" if result.success else "failed",
                            "passed": len(result.completed_tasks),
                            "failed": len(result.failed_tasks)
                        },
                        implementation_notes=f"Executed {result.total_tasks} tasks",
                        next_steps="Review generated code and tests" if result.success else "Fix errors and retry"
                    )
                    logger.info(f"Updated project {self.current_project.project_id}: status={'completed' if result.success else 'failed'}")
                except Exception as e:
                    logger.warning(f"Failed to update project status: {e}")

            logger.info(
                f"Goal {goal_id} completed: "
                f"success={result.success}, "
                f"tasks={result.total_tasks}, "
                f"failed={len(result.failed_tasks)}, "
                f"time={result.execution_time_ms:.1f}ms"
            )

            return result

        except Exception as e:
            logger.exception(f"Goal {goal_id} execution failed with exception")
            result.errors.append(f"Execution exception: {str(e)}")
            result.success = False
            result.completed_at = datetime.now(timezone.utc)

            # Update workspace project as failed
            if self.current_project:
                try:
                    self.project_manager.update_project(
                        project_id=self.current_project.project_id,
                        status="failed",
                        error_message=str(e),
                        next_steps="Review error and retry"
                    )
                except Exception as update_error:
                    logger.warning(f"Failed to update project status: {update_error}")

            return result

    async def _materialize_task_content(
        self,
        action_name: str,
        params: Dict[str, Any],
        action_num: int,
        goal_id: str
    ) -> str:
        """Generate code content using LLM.

        Args:
            action_name: Name of the action (create_file, modify_code)
            params: Task parameters including file_path
            action_num: Action sequence number (1=impl, 2=test)
            goal_id: Current goal ID

        Returns:
            Generated code as string
        """
        # Determine role based on action number
        role = "implementation" if action_num == 1 else "test"

        # Build generation request
        req = GenRequest(
            goal_text=self.current_goal_text or "implement feature",
            context={
                **self.current_context,
                "goal_id": goal_id,
                "action_name": action_name,
            },
            file_path=params.get("file_path", "generated.py"),
            role=role,
            language=params.get("language", "python")
        )

        logger.info(f"Generating {role} code for {req.file_path}")

        # Generate code
        result = await self.code_generator.generate(req)

        logger.info(
            f"Generated {len(result.code)} bytes "
            f"({'cache hit' if result.cache_hit else 'new generation'})"
        )

        return result.code

    async def _plan_goal(
        self,
        goal_id: str,
        goal_text: str,
        context: Optional[Dict[str, Any]],
        constraints: Optional[List[Any]]
    ) -> Optional[Plan]:
        """Run HTN planner to decompose goal.

        This is async to support future async belief kernel queries.
        """
        # HTN planner is synchronous for now - just call it directly
        plan = self.planner.plan(
            goal_id=goal_id,
            goal_text=goal_text,
            world_state=context or {},
            constraints=constraints or []
        )

        return plan

    async def _plan_to_taskgraph(
        self,
        plan: Plan,
        goal_id: str,
        sequential: bool = True
    ) -> TaskGraph:
        """Convert HTN Plan to TaskGraph.

        Args:
            plan: HTN Plan with ordered primitive tasks
            goal_id: Goal identifier for graph naming
            sequential: If True, creates dependency chain; else parallel

        Returns:
            TaskGraph ready for execution
        """
        graph = TaskGraph(
            graph_id=f"goal-{goal_id}",
            max_retry_tokens=10,
            graph_timeout_ms=600000  # 10 minutes
        )

        prev_task_id = None
        task_counter = {}  # Track task types for unique file naming

        for task in plan.tasks:
            task_id = task.task_id
            action_name = task.task_name

            # Enrich parameters with defaults based on action type
            params = await self._enrich_task_parameters(
                action_name=action_name,
                original_params=task.parameters or {},
                task_id=task_id,
                goal_id=goal_id,
                task_counter=task_counter
            )

            # Build dependencies
            deps = [prev_task_id] if (prev_task_id and sequential) else []

            # Add task to graph
            graph.add_task(
                task_id=task_id,
                action_name=action_name,
                normalized_args=params,
                resource_ids=[],
                version="1.0",
                dependencies=deps,
                priority=0.5,
                max_retries=3
            )

            prev_task_id = task_id

        return graph

    async def _enrich_task_parameters(
        self,
        action_name: str,
        original_params: Dict[str, Any],
        task_id: str,
        goal_id: str,
        task_counter: Dict[str, int]
    ) -> Dict[str, Any]:
        """Enrich task parameters with reasonable defaults.

        For demo/testing purposes, adds default file paths and content.
        In production, Astra would provide these from goal context.

        Args:
            action_name: Type of action (create_file, modify_code, etc.)
            original_params: Parameters from HTN plan
            task_id: Task identifier
            goal_id: Goal identifier
            task_counter: Counter for unique file naming

        Returns:
            Enriched parameters dict
        """
        params = original_params.copy()

        # Track action type counts for unique naming
        task_counter[action_name] = task_counter.get(action_name, 0) + 1
        action_num = task_counter[action_name]

        # Add default parameters based on action type
        if action_name in ("create_file", "modify_code", "delete_file"):
            if "file_path" not in params:
                # Use workspace project paths if available
                if self.current_project:
                    # Generate paths in workspace project
                    if action_num == 1:
                        # First file = implementation in src/
                        params["file_path"] = f"{self.current_project.project_dir}/src/feature_{goal_id[:8]}.py"
                    elif action_num == 2:
                        # Second file = tests in tests/
                        params["file_path"] = f"{self.current_project.project_dir}/tests/test_{goal_id[:8]}.py"
                    else:
                        # Additional files in appropriate directory
                        params["file_path"] = f"{self.current_project.project_dir}/src/module_{action_num}_{goal_id[:8]}.py"
                else:
                    # Fallback to tests/generated/ if no project
                    if action_num == 1:
                        params["file_path"] = f"tests/generated/feature_{goal_id[:8]}.py"
                    elif action_num == 2:
                        params["file_path"] = f"tests/generated/test_{goal_id[:8]}.py"
                    else:
                        params["file_path"] = f"tests/generated/file_{action_num}_{goal_id[:8]}.py"

            # Generate code content if not provided
            if "content" not in params and action_name in ("create_file", "modify_code"):
                if self.code_generator:
                    # Use LLM to generate real code
                    try:
                        params["content"] = await self._materialize_task_content(
                            action_name=action_name,
                            params=params,
                            action_num=action_num,
                            goal_id=goal_id
                        )
                    except Exception as e:
                        logger.warning(f"Code generation failed: {e}, using placeholder")
                        params["content"] = f"# Auto-generated for {action_name}\ndef placeholder():\n    pass\n"
                else:
                    # Fallback to placeholder if no code generator
                    params["content"] = f"# Auto-generated for {action_name}\ndef placeholder():\n    pass\n"

            if "reason" not in params:
                params["reason"] = f"Generated by goal {goal_id}"

        elif action_name in ("run_tests", "pytest"):
            if "cmd" not in params:
                # Use python -m pytest to ensure it works without activating venv
                params["cmd"] = ["python3", "-m", "pytest", "-v", "--tb=short"]

        elif action_name in ("shell_command", "bash"):
            if "cmd" not in params:
                params["cmd"] = "echo 'placeholder command'"

        return params

    async def _execute_graph(
        self,
        graph: TaskGraph,
        timeout_ms: int
    ) -> Dict[str, Any]:
        """Execute TaskGraph using TaskExecutionEngine.

        Args:
            graph: TaskGraph to execute
            timeout_ms: Execution timeout

        Returns:
            Execution metadata (stats, timing, etc.)
        """
        # Create execution engine
        engine = TaskExecutionEngine(
            graph=graph,
            executors=self.executors,
            max_concurrent=self.max_concurrent
        )

        # Create RunContext factory
        def make_context(task: TaskNode) -> RunContext:
            return RunContext(
                trace_id=f"goal-{graph.graph_id}",
                span_id=f"task-{task.task_id}",
                workdir=self.workdir,
                timeout_ms=task.task_timeout_ms or timeout_ms,
                env={},
                monotonic=time.monotonic,
                ledger=self.identity_ledger,
                breaker=graph,
                caps={}
            )

        # Run execution
        await engine.run(make_context)

        # Return execution metadata
        return {
            "stats": graph.get_stats(),
            "completed": graph.is_complete()
        }

    def _populate_result(
        self,
        result: GoalExecutionResult,
        graph: TaskGraph,
        execution_meta: Dict[str, Any]
    ) -> None:
        """Populate GoalExecutionResult from graph state.

        Args:
            result: GoalExecutionResult to populate (modified in place)
            graph: Executed TaskGraph
            execution_meta: Execution metadata from engine
        """
        total_retries = 0
        all_artifacts = {}

        for task_id, node in graph.nodes.items():
            task_result = TaskResult(
                task_id=task_id,
                task_name=node.action_name,
                action_name=node.action_name,
                state=node.state.value,
                error=node.last_error,
                error_class=node.error_class,
                retry_count=node.retry_count,
                started_at=node.started_at,
                completed_at=node.completed_at,
                artifacts=node.normalized_args  # Include task args as artifacts
            )

            total_retries += node.retry_count

            if node.state == TaskState.SUCCEEDED:
                result.completed_tasks.append(task_result)
            elif node.state == TaskState.FAILED:
                result.failed_tasks.append(task_result)
                if node.last_error:
                    result.errors.append(f"{task_id}: {node.last_error}")

            # Collect artifacts
            if node.normalized_args:
                all_artifacts[task_id] = node.normalized_args

        result.retry_count = total_retries
        result.artifacts = all_artifacts
