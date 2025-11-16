"""
Task Outcome Linker - Links task execution outcomes to DecisionFramework records.

Provides utilities to evaluate task outcomes and update decision records
for learning and parameter adaptation.
"""

import logging
from typing import Optional
from datetime import datetime, timezone

from src.services.outcome_evaluator import OutcomeEvaluator, TaskOutcome
from src.services.decision_framework import DecisionOutcome

logger = logging.getLogger(__name__)


class TaskOutcomeLinker:
    """
    Links task execution outcomes to decision framework records.

    Bridges the gap between task execution tracking and adaptive
    decision learning by converting TaskOutcomes into DecisionOutcomes.
    """

    def __init__(
        self,
        outcome_evaluator: OutcomeEvaluator,
        decision_framework
    ):
        """
        Initialize outcome linker.

        Args:
            outcome_evaluator: OutcomeEvaluator for task outcome computation
            decision_framework: DecisionFramework for decision tracking
        """
        self.outcome_evaluator = outcome_evaluator
        self.decision_framework = decision_framework

        logger.info("TaskOutcomeLinker initialized")

    async def link_task_outcome(
        self,
        task_id: str,
        execution_id: str,
        decision_record_id: Optional[str],
        status: str,
        started_at: float,
        ended_at: float,
        horizon: str = "short"
    ) -> Optional[TaskOutcome]:
        """
        Evaluate task outcome and link to decision record.

        Args:
            task_id: Task identifier
            execution_id: Experience ID from TASK_EXECUTION
            decision_record_id: Decision record ID to link to (may be None)
            status: Execution status ("success", "failed", "partial")
            started_at: Task start timestamp
            ended_at: Task end timestamp
            horizon: Evaluation horizon ("short" or "long")

        Returns:
            TaskOutcome if evaluation succeeded, None otherwise
        """
        try:
            # Evaluate task outcome
            task_outcome = await self.outcome_evaluator.evaluate_task_outcome(
                task_id=task_id,
                execution_id=execution_id,
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                horizon=horizon
            )

            # Link to decision record if available
            if decision_record_id:
                decision_outcome = self._convert_to_decision_outcome(
                    task_outcome,
                    decision_record_id
                )
                self.decision_framework.registry.update_decision_outcome(
                    record_id=decision_record_id,
                    outcome=decision_outcome
                )
                logger.info(
                    f"Linked task outcome to decision record: {decision_record_id} "
                    f"(score={task_outcome.composite_score:.3f})"
                )
            else:
                logger.debug(f"No decision record to link for task {task_id}")

            return task_outcome

        except Exception as e:
            logger.error(f"Failed to link task outcome: {e}", exc_info=True)
            return None

    def _convert_to_decision_outcome(
        self,
        task_outcome: TaskOutcome,
        decision_record_id: str
    ) -> DecisionOutcome:
        """
        Convert TaskOutcome to DecisionOutcome.

        Args:
            task_outcome: TaskOutcome from evaluation
            decision_record_id: Decision record ID

        Returns:
            DecisionOutcome for decision framework
        """
        return DecisionOutcome(
            decision_record_id=decision_record_id,
            success_score=task_outcome.composite_score,
            coherence_delta=task_outcome.coherence_delta,
            dissonance_delta=task_outcome.dissonance_delta,
            satisfaction_delta=task_outcome.satisfaction_score,
            aborted=task_outcome.status == "failed",
            abort_reason=None,  # Could be enhanced to track abort reasons
            evaluation_timestamp=datetime.fromtimestamp(
                task_outcome.evaluated_at,
                tz=timezone.utc
            )
        )

    async def batch_link_outcomes(
        self,
        task_executions: list,
        horizon: str = "short"
    ) -> list:
        """
        Link multiple task outcomes in batch.

        Args:
            task_executions: List of task execution dicts with required fields
            horizon: Evaluation horizon

        Returns:
            List of TaskOutcomes (None for failed evaluations)
        """
        outcomes = []
        for execution in task_executions:
            outcome = await self.link_task_outcome(
                task_id=execution["task_id"],
                execution_id=execution["execution_id"],
                decision_record_id=execution.get("decision_record_id"),
                status=execution["status"],
                started_at=execution["started_at"],
                ended_at=execution["ended_at"],
                horizon=horizon
            )
            outcomes.append(outcome)

        successful_links = sum(1 for o in outcomes if o is not None)
        logger.info(
            f"Batch linked {successful_links}/{len(task_executions)} task outcomes"
        )

        return outcomes


def create_task_outcome_linker(
    outcome_evaluator: OutcomeEvaluator,
    decision_framework
) -> TaskOutcomeLinker:
    """Factory function to create task outcome linker."""
    return TaskOutcomeLinker(
        outcome_evaluator=outcome_evaluator,
        decision_framework=decision_framework
    )
