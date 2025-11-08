"""
Outcome Evaluation Background Task.

Periodically evaluates decision outcomes and triggers parameter adaptation.
Runs as an asyncio background task.
"""

import asyncio
import logging
from datetime import datetime, timezone

from src.services.parameter_adapter import ParameterAdapter
from src.services.decision_framework import get_decision_registry
from src.services.success_signal_evaluator import SuccessSignalEvaluator

logger = logging.getLogger(__name__)


class OutcomeEvaluationTask:
    """Background task for evaluating decision outcomes and adapting parameters."""

    def __init__(
        self,
        parameter_adapter: ParameterAdapter,
        interval_minutes: int = 30,
        adaptation_interval_hours: int = 168,  # Weekly by default
        enabled: bool = True
    ):
        """
        Args:
            parameter_adapter: Parameter adapter instance
            interval_minutes: How often to check for unevaluated decisions
            adaptation_interval_hours: How often to trigger parameter adaptation
            enabled: Whether to start the task
        """
        self.adapter = parameter_adapter
        self.registry = get_decision_registry()
        self.interval_minutes = interval_minutes
        self.adaptation_interval_hours = adaptation_interval_hours
        self.enabled = enabled

        self.task = None
        self.running = False
        self.last_adaptation_time = None
        self.evaluation_count = 0
        self.adaptation_count = 0

    async def start(self):
        """Start the background task."""
        if not self.enabled:
            logger.info("Outcome evaluation task disabled")
            return

        if self.task and not self.task.done():
            logger.warning("Outcome evaluation task already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._run())
        logger.info(
            f"Started outcome evaluation task: "
            f"check every {self.interval_minutes}m, "
            f"adapt every {self.adaptation_interval_hours}h"
        )

    async def stop(self):
        """Stop the background task."""
        if not self.task:
            return

        self.running = False
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass

        logger.info("Stopped outcome evaluation task")

    async def _run(self):
        """Main task loop."""
        while self.running:
            try:
                # Check if it's time to adapt
                should_adapt = self._should_adapt()

                if should_adapt:
                    logger.info("Triggering parameter adaptation")
                    await self._adapt_parameters()
                else:
                    # Just evaluate outcomes
                    await self._evaluate_outcomes()

                # Wait for next cycle
                await asyncio.sleep(self.interval_minutes * 60)

            except asyncio.CancelledError:
                logger.info("Outcome evaluation task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in outcome evaluation task: {e}", exc_info=True)
                # Continue running despite errors
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _evaluate_outcomes(self):
        """Evaluate pending decision outcomes without adapting."""
        # Get unevaluated decisions
        unevaluated = self.registry.get_unevaluated_decisions(older_than_hours=24)

        if not unevaluated:
            logger.debug("No decisions ready for evaluation")
            return

        logger.info(f"Evaluating {len(unevaluated)} decision outcomes")

        # This would trigger actual evaluation
        # For now, just count
        self.evaluation_count += len(unevaluated)

        # TODO: Actually call evaluator to compute outcomes
        # The adapter.adapt_all_decisions() does this, but we may want
        # evaluation separate from adaptation

    async def _adapt_parameters(self):
        """Evaluate outcomes and adapt parameters."""
        try:
            # Run adaptation
            results = self.adapter.adapt_all_decisions(
                evaluation_window_hours=24,
                dry_run=False
            )

            # Log results
            adapted_count = sum(1 for r in results.values() if r.get("adapted"))
            total_samples = sum(r.get("sample_count", 0) for r in results.values())

            logger.info(
                f"Parameter adaptation complete: "
                f"{adapted_count}/{len(results)} decision types adapted, "
                f"{total_samples} total decisions evaluated"
            )

            for decision_id, result in results.items():
                if result.get("adapted"):
                    adjustments = result.get("adjustments", {})
                    avg_score = result.get("avg_success_score", 0)
                    logger.info(
                        f"  {decision_id}: {len(adjustments)} params adjusted, "
                        f"avg_score={avg_score:.2f}"
                    )

            self.adaptation_count += 1
            self.last_adaptation_time = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Failed to adapt parameters: {e}", exc_info=True)

    def _should_adapt(self) -> bool:
        """Check if it's time to run parameter adaptation."""
        if not self.last_adaptation_time:
            # First run
            return True

        elapsed = datetime.now(timezone.utc) - self.last_adaptation_time
        elapsed_hours = elapsed.total_seconds() / 3600

        return elapsed_hours >= self.adaptation_interval_hours

    def get_telemetry(self) -> dict:
        """Get telemetry data for monitoring."""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "interval_minutes": self.interval_minutes,
            "adaptation_interval_hours": self.adaptation_interval_hours,
            "evaluation_count": self.evaluation_count,
            "adaptation_count": self.adaptation_count,
            "last_adaptation_time": (
                self.last_adaptation_time.isoformat()
                if self.last_adaptation_time
                else None
            )
        }
