"""HTN (Hierarchical Task Network) Planner.

Decomposes high-level goals into executable primitive tasks using
hierarchical decomposition methods. Integrates with BeliefKernel for
precondition checking and DecisionFramework for adaptive method selection.

Key Features:
- Hierarchical task decomposition
- Belief-based precondition checking
- Cost-based method selection
- Constraint satisfaction
- Adaptive learning through DecisionFramework

Example:
    >>> from src.services.htn_planner import HTNPlanner, Method, Task
    >>>
    >>> # Define decomposition methods
    >>> methods = [
    >>>     Method(
    >>>         name="improve_code_quality",
    >>>         task="improve_code_quality",
    >>>         preconditions=["belief_has_codebase"],
    >>>         subtasks=["run_linter", "fix_issues", "run_tests"],
    >>>         constraints=[],
    >>>         cost=0.7
    >>>     )
    >>> ]
    >>>
    >>> planner = HTNPlanner(belief_kernel, methods)
    >>> plan = planner.plan(goal, world_state, constraints)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from datetime import datetime, timezone
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


# === Data Structures ===

@dataclass
class Method:
    """HTN decomposition method.

    Defines how to decompose a compound task into subtasks.
    Methods are selected based on precondition satisfaction and cost.

    Attributes:
        name: Unique identifier for this method
        task: High-level task name this method decomposes
        preconditions: List of belief IDs that must be satisfied
        subtasks: Ordered list of subtask names
        constraints: Additional constraint functions that must be satisfied
        cost: Expected effort/cost (0.0-1.0, lower is better)
        metadata: Additional method-specific data
    """
    name: str
    task: str
    preconditions: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    constraints: List[Callable] = field(default_factory=list)
    cost: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate method definition."""
        if not self.name:
            raise ValueError("Method name cannot be empty")
        if not self.task:
            raise ValueError("Method task cannot be empty")
        if not 0.0 <= self.cost <= 1.0:
            raise ValueError(f"Method cost must be in [0.0, 1.0], got {self.cost}")
        if not self.subtasks:
            raise ValueError("Method must have at least one subtask")


@dataclass
class Task:
    """Task in the HTN hierarchy.

    Tasks can be either compound (decomposable) or primitive (executable).

    Attributes:
        task_id: Unique identifier
        task_name: Human-readable task name
        primitive: True if task can be executed directly
        parameters: Task-specific parameters
        metadata: Additional task data
    """
    task_id: str
    task_name: str
    primitive: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate task definition."""
        if not self.task_id:
            raise ValueError("Task ID cannot be empty")
        if not self.task_name:
            raise ValueError("Task name cannot be empty")


@dataclass
class Plan:
    """Ordered sequence of primitive tasks.

    Result of HTN planning: a total-order plan of primitive tasks
    that achieves the goal while satisfying all constraints.

    Attributes:
        plan_id: Unique plan identifier
        goal_id: ID of the goal this plan achieves
        goal_text: Text description of the goal
        tasks: Ordered list of primitive tasks
        total_cost: Sum of method costs used
        constraints_satisfied: Names of satisfied constraints
        methods_used: Names of decomposition methods applied
        created_at: Plan creation timestamp
        metadata: Additional plan data
    """
    plan_id: str
    goal_id: str
    goal_text: str
    tasks: List[Task]
    total_cost: float
    constraints_satisfied: List[str] = field(default_factory=list)
    methods_used: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# === HTN Planner ===

class HTNPlanner:
    """Hierarchical Task Network planner with belief-based preconditions.

    Decomposes high-level goals into primitive tasks using methods that
    specify how to break down compound tasks. Method selection is based on:
    - Precondition satisfaction (beliefs must hold)
    - Cost minimization (prefer cheaper methods)
    - Constraint satisfaction (all constraints must pass)

    Example:
        >>> planner = HTNPlanner(belief_kernel, method_library)
        >>> plan = planner.plan(
        ...     goal_id="improve_quality",
        ...     goal_text="Improve code quality",
        ...     world_state={"has_codebase": True},
        ...     constraints=[]
        ... )
    """

    def __init__(
        self,
        belief_kernel: Optional[Any] = None,
        method_library: Optional[List[Method]] = None,
        primitive_tasks: Optional[Set[str]] = None
    ):
        """Initialize HTN planner.

        Args:
            belief_kernel: BeliefKernel for precondition checking (optional)
            method_library: List of decomposition methods
            primitive_tasks: Set of task names that are primitive (executable)
        """
        self.belief_kernel = belief_kernel
        self.methods = method_library or []
        self.primitive_tasks = primitive_tasks or set()

        # Build index for fast method lookup
        self._method_index: Dict[str, List[Method]] = {}
        for method in self.methods:
            if method.task not in self._method_index:
                self._method_index[method.task] = []
            self._method_index[method.task].append(method)

        logger.info(
            f"HTNPlanner initialized with {len(self.methods)} methods, "
            f"{len(self.primitive_tasks)} primitive tasks"
        )

    def add_method(self, method: Method) -> None:
        """Add a decomposition method to the library.

        Args:
            method: Method to add
        """
        self.methods.append(method)

        if method.task not in self._method_index:
            self._method_index[method.task] = []
        self._method_index[method.task].append(method)

        logger.debug(f"Added method '{method.name}' for task '{method.task}'")

    def add_primitive_task(self, task_name: str) -> None:
        """Mark a task as primitive (executable).

        Args:
            task_name: Name of primitive task
        """
        self.primitive_tasks.add(task_name)
        logger.debug(f"Added primitive task '{task_name}'")

    def is_primitive(self, task_name: str) -> bool:
        """Check if a task is primitive (executable directly).

        Args:
            task_name: Task name to check

        Returns:
            True if task is primitive, False if compound
        """
        return task_name in self.primitive_tasks

    def plan(
        self,
        goal_id: str,
        goal_text: str,
        world_state: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Callable]] = None,
        max_depth: int = 100
    ) -> Optional[Plan]:
        """Generate plan using HTN decomposition.

        Algorithm:
        1. Start with goal as single compound task
        2. While task_network is not empty:
           - Pop current task
           - If primitive: add to plan
           - Else: find applicable decomposition methods
           - Pick lowest-cost method
           - Replace current task with subtasks
        3. Check all constraints
        4. Return plan or None if no valid plan exists

        Args:
            goal_id: Unique goal identifier
            goal_text: Goal description (becomes initial compound task)
            world_state: Current world state for precondition checking
            constraints: Additional constraint functions
            max_depth: Maximum decomposition depth (safety limit)

        Returns:
            Plan if successful, None if no valid decomposition exists
        """
        world_state = world_state or {}
        constraints = constraints or []

        logger.info(f"Planning for goal '{goal_id}': {goal_text}")

        # Start with goal as single compound task
        initial_task = Task(
            task_id=goal_id,
            task_name=goal_text,
            primitive=self.is_primitive(goal_text),
            parameters={}
        )

        task_network = [initial_task]
        plan_tasks: List[Task] = []
        methods_used: List[str] = []
        total_cost = 0.0
        depth = 0

        while task_network and depth < max_depth:
            current_task = task_network.pop(0)
            depth += 1

            if current_task.primitive:
                # Task is executable - add to plan
                plan_tasks.append(current_task)
                logger.debug(f"Added primitive task to plan: {current_task.task_name}")
            else:
                # Task is compound - decompose it
                logger.debug(f"Decomposing compound task: {current_task.task_name}")

                # Find applicable decomposition methods
                applicable = self._find_applicable_methods(
                    current_task.task_name,
                    world_state
                )

                if not applicable:
                    logger.warning(
                        f"No applicable methods for task '{current_task.task_name}' "
                        f"with world_state={world_state}"
                    )
                    return None  # No way to decompose

                # Pick lowest-cost method
                method = min(applicable, key=lambda m: m.cost)
                logger.debug(
                    f"Selected method '{method.name}' (cost={method.cost:.2f}) "
                    f"for task '{current_task.task_name}'"
                )

                # Decompose: replace current_task with subtasks
                subtasks = [
                    Task(
                        task_id=f"{current_task.task_id}.{i}",
                        task_name=st,
                        primitive=self.is_primitive(st),
                        parameters=current_task.parameters.copy(),
                        metadata={"parent_task": current_task.task_id, "method": method.name}
                    )
                    for i, st in enumerate(method.subtasks)
                ]

                # Insert subtasks at front of network (total-order planning)
                task_network = subtasks + task_network

                # Track method usage and cost
                methods_used.append(method.name)
                total_cost += method.cost

        if depth >= max_depth:
            logger.error(f"Planning exceeded max depth {max_depth} - possible infinite recursion")
            return None

        # Check all constraints
        constraints_satisfied = []
        for constraint in constraints:
            if not constraint(plan_tasks):
                logger.warning(
                    f"Plan failed constraint '{constraint.__name__}' "
                    f"with {len(plan_tasks)} tasks"
                )
                return None
            constraints_satisfied.append(constraint.__name__)

        plan = Plan(
            plan_id=str(uuid4()),
            goal_id=goal_id,
            goal_text=goal_text,
            tasks=plan_tasks,
            total_cost=total_cost,
            constraints_satisfied=constraints_satisfied,
            methods_used=methods_used,
            created_at=datetime.now(timezone.utc)
        )

        logger.info(
            f"Generated plan '{plan.plan_id}' with {len(plan_tasks)} tasks, "
            f"cost={total_cost:.2f}, methods={methods_used}"
        )

        return plan

    def _find_applicable_methods(
        self,
        task_name: str,
        world_state: Dict[str, Any]
    ) -> List[Method]:
        """Find all methods applicable to a task given world state.

        A method is applicable if:
        1. It decomposes the given task
        2. All preconditions are satisfied

        Args:
            task_name: Name of compound task to decompose
            world_state: Current world state

        Returns:
            List of applicable methods, sorted by cost (ascending)
        """
        # Get methods for this task
        methods = self._method_index.get(task_name, [])

        # Filter by precondition satisfaction
        applicable = [
            m for m in methods
            if self._preconditions_satisfied(m.preconditions, world_state)
        ]

        # Sort by cost (prefer cheaper methods)
        applicable.sort(key=lambda m: m.cost)

        return applicable

    def _preconditions_satisfied(
        self,
        preconditions: List[str],
        world_state: Dict[str, Any]
    ) -> bool:
        """Check if all preconditions are satisfied.

        Preconditions can be either:
        1. Belief IDs (checked via BeliefKernel if available)
        2. World state keys (checked directly)

        Args:
            preconditions: List of belief IDs or world state keys
            world_state: Current world state

        Returns:
            True if all preconditions satisfied, False otherwise
        """
        for precondition in preconditions:
            # Try checking via BeliefKernel first
            if self.belief_kernel:
                try:
                    belief = self.belief_kernel.get_belief(precondition)
                    if belief and belief.confidence >= 0.5:
                        continue  # Precondition satisfied via belief
                except:
                    pass  # Fall through to world_state check

            # Check world_state
            if precondition in world_state:
                value = world_state[precondition]
                # Treat as boolean if possible
                if isinstance(value, bool):
                    if value:
                        continue
                    else:
                        return False
                elif value:  # Truthy check
                    continue

            # Precondition not satisfied
            logger.debug(f"Precondition '{precondition}' not satisfied")
            return False

        return True


# === TaskGraph Integration ===

def plan_to_task_graph(plan: Plan, task_graph=None):
    """Convert an HTN Plan to a TaskGraph for execution.

    Creates task nodes from plan tasks and adds them to a TaskGraph
    with sequential dependencies (each task depends on the previous).

    Args:
        plan: HTN Plan to convert
        task_graph: Optional existing TaskGraph to add tasks to.
                   If None, creates a new TaskGraph.

    Returns:
        TaskGraph with plan tasks added

    Example:
        >>> plan = planner.plan(goal_id="g1", goal_text="improve_quality")
        >>> task_graph = plan_to_task_graph(plan)
        >>> ready_tasks = task_graph.get_ready_tasks()
    """
    from src.services.task_graph import TaskGraph

    if task_graph is None:
        task_graph = TaskGraph(graph_id=plan.plan_id)

    if not plan.tasks:
        logger.warning(f"Plan {plan.plan_id} has no tasks")
        return task_graph

    # Add tasks with sequential dependencies
    prev_task_id = None

    for task in plan.tasks:
        # Store metadata in normalized_args
        args_with_metadata = {
            **task.parameters,
            "_plan_id": plan.plan_id,
            "_goal_id": plan.goal_id,
            "_goal_text": plan.goal_text,
            "_htn_metadata": task.metadata
        }

        # Add task using add_task method (expects individual parameters, not TaskNode)
        task_graph.add_task(
            task_id=task.task_id,
            action_name=task.task_name,
            normalized_args=args_with_metadata,
            resource_ids=[],
            version="1.0",
            dependencies=[prev_task_id] if prev_task_id else [],
            priority=1.0,
            deadline=None
        )

        prev_task_id = task.task_id

    logger.info(
        f"Converted plan {plan.plan_id} to TaskGraph: "
        f"{len(plan.tasks)} tasks with sequential dependencies"
    )

    # Persist TaskGraph for querying
    _persist_task_graph(task_graph)

    return task_graph


def _persist_task_graph(task_graph: "TaskGraph") -> None:
    """Persist TaskGraph to disk for query API."""
    import json
    from pathlib import Path

    graph_dir = Path("persona_space/taskgraphs")
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph_file = graph_dir / f"{task_graph.graph_id}.json"

    try:
        data = task_graph.to_dict()
        with open(graph_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Persisted TaskGraph: {graph_file}")
    except Exception as e:
        logger.error(f"Failed to persist TaskGraph {task_graph.graph_id}: {e}")


# === Helper Functions ===

def register_htn_decision(decision_registry) -> None:
    """Register plan_generated as an adaptive decision point.

    Allows DecisionFramework to learn optimal method selection parameters.

    Args:
        decision_registry: DecisionRegistry instance
    """
    from src.services.decision_framework import Parameter

    parameters = {
        "cost_weight": Parameter(
            name="cost_weight",
            current_value=0.7,
            min_value=0.0,
            max_value=1.0,
            step_size=0.05,
            adaptation_rate=0.1
        ),
        "precondition_threshold": Parameter(
            name="precondition_threshold",
            current_value=0.5,
            min_value=0.3,
            max_value=0.9,
            step_size=0.05,
            adaptation_rate=0.1
        ),
    }

    decision_registry.register_decision(
        decision_id="plan_generated",
        subsystem="htn_planner",
        description="HTN method selection and plan generation",
        parameters=parameters,
        success_metrics=["plan_success", "task_completion", "cost_accuracy"],
        context_features=["goal_category", "world_state_size", "method_count"]
    )

    logger.info("Registered plan_generated decision point with adaptive parameters")
