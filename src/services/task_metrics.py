"""
Task Execution Metrics - Track success rates and patterns for scheduled tasks.

Provides aggregated metrics on task execution outcomes to support:
- Success rate tracking per task type
- Failure pattern detection
- Duration statistics
- Outcome correlation analysis
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.services.outcome_evaluator import TaskOutcome

logger = logging.getLogger(__name__)


@dataclass
class TaskMetrics:
    """Aggregated metrics for a specific task."""
    task_id: str
    task_type: str

    # Execution counts
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    partial_executions: int = 0

    # Duration statistics (ms)
    avg_duration_ms: float = 0.0
    min_duration_ms: int = 0
    max_duration_ms: int = 0

    # Outcome statistics
    avg_composite_score: float = 0.0
    avg_coherence_delta: float = 0.0
    avg_dissonance_delta: float = 0.0
    avg_satisfaction: float = 0.0

    # Recent outcomes (last 10)
    recent_outcomes: List[TaskOutcome] = field(default_factory=list)

    # Last execution
    last_execution_at: Optional[float] = None
    last_status: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Compute success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def failure_rate(self) -> float:
        """Compute failure rate."""
        if self.total_executions == 0:
            return 0.0
        return self.failed_executions / self.total_executions


class TaskMetricsTracker:
    """
    Tracks and aggregates task execution metrics.

    Maintains running statistics for each task and provides
    methods to query success rates, failure patterns, and
    outcome correlations.
    """

    def __init__(self, max_history: int = 100):
        """
        Initialize metrics tracker.

        Args:
            max_history: Maximum number of outcomes to keep in history
        """
        self.max_history = max_history

        # Task metrics by task_id
        self.metrics: Dict[str, TaskMetrics] = {}

        # Task type aggregations
        self.type_metrics: Dict[str, TaskMetrics] = {}

        # All outcomes (for global analysis)
        self.all_outcomes: List[TaskOutcome] = []

        logger.info(f"TaskMetricsTracker initialized (max_history={max_history})")

    def record_outcome(self, outcome: TaskOutcome, task_type: str) -> None:
        """
        Record a task outcome and update metrics.

        Args:
            outcome: TaskOutcome to record
            task_type: Type of task (for type-level aggregation)
        """
        # Initialize task metrics if needed
        if outcome.task_id not in self.metrics:
            self.metrics[outcome.task_id] = TaskMetrics(
                task_id=outcome.task_id,
                task_type=task_type
            )

        # Initialize type metrics if needed
        if task_type not in self.type_metrics:
            self.type_metrics[task_type] = TaskMetrics(
                task_id=f"TYPE:{task_type}",
                task_type=task_type
            )

        # Update task-level metrics
        self._update_metrics(self.metrics[outcome.task_id], outcome)

        # Update type-level metrics
        self._update_metrics(self.type_metrics[task_type], outcome)

        # Add to global history
        self.all_outcomes.append(outcome)
        if len(self.all_outcomes) > self.max_history:
            self.all_outcomes.pop(0)

        logger.debug(
            f"Recorded outcome for {outcome.task_id}: "
            f"status={outcome.status}, score={outcome.composite_score:.3f}"
        )

    def _update_metrics(self, metrics: TaskMetrics, outcome: TaskOutcome) -> None:
        """Update metrics with new outcome."""
        # Update execution counts
        metrics.total_executions += 1

        if outcome.status == "success":
            metrics.successful_executions += 1
        elif outcome.status == "failed":
            metrics.failed_executions += 1
        elif outcome.status == "partial":
            metrics.partial_executions += 1

        # Update duration statistics
        if metrics.total_executions == 1:
            # First execution
            metrics.avg_duration_ms = outcome.duration_ms
            metrics.min_duration_ms = outcome.duration_ms
            metrics.max_duration_ms = outcome.duration_ms
        else:
            # Running average
            n = metrics.total_executions
            metrics.avg_duration_ms = (
                (metrics.avg_duration_ms * (n - 1) + outcome.duration_ms) / n
            )
            metrics.min_duration_ms = min(metrics.min_duration_ms, outcome.duration_ms)
            metrics.max_duration_ms = max(metrics.max_duration_ms, outcome.duration_ms)

        # Update outcome statistics (running averages)
        n = metrics.total_executions
        metrics.avg_composite_score = (
            (metrics.avg_composite_score * (n - 1) + outcome.composite_score) / n
        )
        metrics.avg_coherence_delta = (
            (metrics.avg_coherence_delta * (n - 1) + outcome.coherence_delta) / n
        )
        metrics.avg_dissonance_delta = (
            (metrics.avg_dissonance_delta * (n - 1) + outcome.dissonance_delta) / n
        )
        metrics.avg_satisfaction = (
            (metrics.avg_satisfaction * (n - 1) + outcome.satisfaction_score) / n
        )

        # Update recent outcomes
        metrics.recent_outcomes.append(outcome)
        if len(metrics.recent_outcomes) > 10:
            metrics.recent_outcomes.pop(0)

        # Update last execution
        metrics.last_execution_at = outcome.evaluated_at
        metrics.last_status = outcome.status

    def get_task_metrics(self, task_id: str) -> Optional[TaskMetrics]:
        """Get metrics for specific task."""
        return self.metrics.get(task_id)

    def get_type_metrics(self, task_type: str) -> Optional[TaskMetrics]:
        """Get aggregated metrics for task type."""
        return self.type_metrics.get(task_type)

    def get_all_task_metrics(self) -> List[TaskMetrics]:
        """Get metrics for all tasks."""
        return list(self.metrics.values())

    def get_failing_tasks(self, threshold: float = 0.3) -> List[TaskMetrics]:
        """
        Get tasks with high failure rates.

        Args:
            threshold: Failure rate threshold (default 0.3 = 30%)

        Returns:
            List of TaskMetrics with failure rate >= threshold
        """
        return [
            m for m in self.metrics.values()
            if m.failure_rate >= threshold and m.total_executions >= 3
        ]

    def get_low_performing_tasks(self, threshold: float = -0.2) -> List[TaskMetrics]:
        """
        Get tasks with low average composite scores.

        Args:
            threshold: Composite score threshold (default -0.2)

        Returns:
            List of TaskMetrics with avg_composite_score <= threshold
        """
        return [
            m for m in self.metrics.values()
            if m.avg_composite_score <= threshold and m.total_executions >= 3
        ]

    def detect_degradation(self, task_id: str, window: int = 5) -> bool:
        """
        Detect if task performance is degrading.

        Compares recent outcomes to historical average.

        Args:
            task_id: Task to check
            window: Number of recent outcomes to compare

        Returns:
            True if recent performance is significantly worse
        """
        metrics = self.metrics.get(task_id)
        if not metrics or len(metrics.recent_outcomes) < window:
            return False

        # Get recent scores
        recent = metrics.recent_outcomes[-window:]
        recent_avg = sum(o.composite_score for o in recent) / len(recent)

        # Compare to historical average
        # Degradation if recent avg is 0.3 or more below historical
        return recent_avg < (metrics.avg_composite_score - 0.3)

    def get_correlation_analysis(self) -> Dict[str, float]:
        """
        Analyze correlations between duration and outcomes.

        Returns:
            Dict with correlation coefficients
        """
        if len(self.all_outcomes) < 10:
            return {
                "duration_vs_success": 0.0,
                "coherence_vs_satisfaction": 0.0,
            }

        # Extract arrays
        durations = np.array([o.duration_ms for o in self.all_outcomes])
        scores = np.array([o.composite_score for o in self.all_outcomes])
        coherence = np.array([o.coherence_delta for o in self.all_outcomes])
        satisfaction = np.array([o.satisfaction_score for o in self.all_outcomes])

        # Compute correlations
        try:
            duration_vs_success = np.corrcoef(durations, scores)[0, 1]
            coherence_vs_satisfaction = np.corrcoef(coherence, satisfaction)[0, 1]
        except:
            # Handle edge cases (e.g., all zeros)
            duration_vs_success = 0.0
            coherence_vs_satisfaction = 0.0

        return {
            "duration_vs_success": float(duration_vs_success),
            "coherence_vs_satisfaction": float(coherence_vs_satisfaction),
        }

    def get_telemetry(self) -> Dict:
        """Get telemetry for status endpoint."""
        return {
            "total_tasks_tracked": len(self.metrics),
            "total_task_types": len(self.type_metrics),
            "total_outcomes_recorded": len(self.all_outcomes),
            "failing_tasks": len(self.get_failing_tasks()),
            "low_performing_tasks": len(self.get_low_performing_tasks()),
            "correlations": self.get_correlation_analysis(),
        }


def create_task_metrics_tracker(max_history: int = 100) -> TaskMetricsTracker:
    """Factory function to create task metrics tracker."""
    return TaskMetricsTracker(max_history=max_history)
