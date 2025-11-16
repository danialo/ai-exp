"""Task Failure Pattern Detector.

Monitors task execution failures and proposes goals to fix recurring issues.
"""

import logging
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from src.services.goal_generator import PatternDetector, GoalProposal
from src.services.goal_store import GoalCategory
from src.services.task_scheduler import TaskScheduler

logger = logging.getLogger(__name__)


class TaskFailureDetector(PatternDetector):
    """Detects recurring task failures and proposes fix goals.

    Monitors task execution history and generates proposals when:
    - Same task fails multiple times
    - Failure rate exceeds threshold
    - Critical tasks are failing

    Example:
        >>> detector = TaskFailureDetector(
        ...     task_scheduler=scheduler,
        ...     failure_threshold=3,
        ...     lookback_hours=24
        ... )
        >>> proposals = await detector.detect()
    """

    def __init__(
        self,
        task_scheduler: Optional[TaskScheduler] = None,
        failure_threshold: int = 3,
        lookback_hours: int = 24,
        min_confidence: float = 0.75,
        scan_interval: int = 60,  # minutes
        detector_enabled: bool = True,
    ):
        """Initialize task failure detector.

        Args:
            task_scheduler: TaskScheduler for accessing task history
            failure_threshold: Number of failures to trigger proposal
            lookback_hours: How far back to look for failures
            min_confidence: Minimum confidence for proposals
            scan_interval: How often to scan (minutes)
            detector_enabled: Whether detector is active
        """
        self._task_scheduler = task_scheduler
        self._failure_threshold = failure_threshold
        self._lookback_hours = lookback_hours
        self._min_confidence = min_confidence
        self._scan_interval = scan_interval
        self._detector_enabled = detector_enabled

    async def detect(self) -> List[GoalProposal]:
        """Scan for recurring task failures.

        Returns:
            List of goal proposals for fixing failures
        """
        if not self._task_scheduler:
            logger.warning("TaskFailureDetector: No task scheduler configured")
            return []

        proposals = []

        # Get task execution history
        failures = self._get_recent_failures()

        if not failures:
            logger.debug("TaskFailureDetector: No recent failures found")
            return []

        # Count failures by task type
        failure_counts = Counter(f["task_type"] for f in failures)

        # Generate proposals for recurring failures
        for task_type, count in failure_counts.items():
            if count >= self._failure_threshold:
                proposal = self._create_proposal(
                    task_type=task_type,
                    failure_count=count,
                    recent_failures=failures
                )

                if proposal:
                    proposals.append(proposal)

        if proposals:
            logger.info(
                f"TaskFailureDetector: Generated {len(proposals)} proposals "
                f"from {len(failures)} failures"
            )

        return proposals

    def name(self) -> str:
        """Detector identifier."""
        return "task_failure_detector"

    def scan_interval_minutes(self) -> int:
        """Scan interval in minutes."""
        return self._scan_interval

    def enabled(self) -> bool:
        """Whether detector is enabled."""
        return self._detector_enabled

    def _get_recent_failures(self) -> List[dict]:
        """Get recent task failures from scheduler.

        Returns:
            List of failure records
        """
        # This is a placeholder - actual implementation would query
        # task execution history from TaskScheduler or database

        # For now, return empty list until we wire in actual history
        # TODO: Wire into TaskScheduler.get_execution_history()

        logger.debug(
            f"TaskFailureDetector: Checking failures "
            f"(lookback={self._lookback_hours}h)"
        )

        return []

    def _create_proposal(
        self,
        task_type: str,
        failure_count: int,
        recent_failures: List[dict]
    ) -> Optional[GoalProposal]:
        """Create goal proposal from failure pattern.

        Args:
            task_type: Type of failing task
            failure_count: Number of recent failures
            recent_failures: List of failure records

        Returns:
            Goal proposal or None
        """
        # Calculate confidence based on failure frequency
        # More failures = higher confidence this needs fixing
        confidence = min(
            self._min_confidence + (failure_count - self._failure_threshold) * 0.05,
            0.95
        )

        # Get most recent error message
        recent_error = None
        for failure in reversed(recent_failures):
            if failure.get("task_type") == task_type:
                recent_error = failure.get("error_message", "Unknown error")
                break

        # Create proposal
        proposal = GoalProposal(
            text=f"Fix recurring task failure: {task_type}",
            category=GoalCategory.MAINTENANCE,
            pattern_detected="task_failure_recurring",
            evidence={
                "task_type": task_type,
                "failure_count": failure_count,
                "lookback_hours": self._lookback_hours,
                "recent_error": recent_error,
            },
            confidence=confidence,
            estimated_value=0.7,  # Fixing failures has good value
            estimated_effort=0.5,  # Medium effort (depends on issue)
            estimated_risk=0.3,   # Some risk in making changes
            aligns_with=[],  # Could check for "reliability" belief
            contradicts=[],
            detector_name=self.name(),
        )

        logger.info(
            f"Created proposal for {task_type}: "
            f"{failure_count} failures, confidence={confidence:.2f}"
        )

        return proposal
