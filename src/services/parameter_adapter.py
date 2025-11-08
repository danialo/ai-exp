"""
Parameter Adapter - Learns optimal parameter values from decision outcomes.

Uses gradient-free optimization to adjust decision parameters based on
observed success scores. Implements epsilon-greedy exploration to avoid
local optima.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timezone

from src.services.decision_framework import DecisionRegistry, DecisionOutcome, get_decision_registry
from src.services.success_signal_evaluator import SuccessSignalEvaluator
from src.services.identity_ledger import parameter_adapted_event

logger = logging.getLogger(__name__)


class ParameterAdapter:
    """
    Adapts decision parameters based on observed outcomes.

    Uses a simple but effective gradient-free approach:
    1. Track recent outcomes for each parameter configuration
    2. If outcomes consistently positive → adjust in same direction
    3. If outcomes consistently negative → adjust in opposite direction
    4. Add epsilon-greedy exploration to avoid local optima
    """

    def __init__(
        self,
        decision_registry: Optional[DecisionRegistry] = None,
        success_evaluator: Optional[SuccessSignalEvaluator] = None,
        min_samples: int = 20,
        exploration_rate: float = 0.10,
        adaptation_rate: float = 0.15
    ):
        """
        Args:
            decision_registry: Decision registry for parameter access
            success_evaluator: Evaluator for computing outcomes
            min_samples: Minimum decisions before adapting
            exploration_rate: Probability of random exploration (0-1)
            adaptation_rate: How aggressively to adapt parameters (0-1)
        """
        self.registry = decision_registry or get_decision_registry()
        self.evaluator = success_evaluator
        self.min_samples = min_samples
        self.exploration_rate = exploration_rate
        self.adaptation_rate = adaptation_rate

        # Track adaptation history
        self.adaptation_history: List[Dict] = []

    def adapt_all_decisions(
        self,
        evaluation_window_hours: int = 24,
        dry_run: bool = False
    ) -> Dict[str, Dict]:
        """
        Adapt parameters for all decision types.

        Args:
            evaluation_window_hours: How long to wait before evaluating
            dry_run: If True, compute adaptations but don't apply

        Returns:
            Dictionary mapping decision_id to adaptation results
        """
        results = {}

        # Get all unevaluated decisions
        unevaluated = self.registry.get_unevaluated_decisions(
            older_than_hours=evaluation_window_hours
        )

        # Group by decision type
        by_decision = defaultdict(list)
        for decision in unevaluated:
            by_decision[decision["decision_id"]].append(decision)

        logger.info(
            f"Adapting parameters: {len(by_decision)} decision types, "
            f"{len(unevaluated)} total decisions to evaluate"
        )

        # Adapt each decision type
        for decision_id, decisions in by_decision.items():
            result = self.adapt_decision(
                decision_id=decision_id,
                unevaluated_decisions=decisions,
                dry_run=dry_run
            )
            results[decision_id] = result

        return results

    def adapt_decision(
        self,
        decision_id: str,
        unevaluated_decisions: Optional[List[Dict]] = None,
        dry_run: bool = False
    ) -> Dict:
        """
        Adapt parameters for a specific decision type.

        Args:
            decision_id: Decision type to adapt
            unevaluated_decisions: Pre-fetched unevaluated decisions (optional)
            dry_run: If True, compute adaptations but don't apply

        Returns:
            Dictionary with adaptation results
        """
        # Get unevaluated decisions if not provided
        if unevaluated_decisions is None:
            unevaluated_decisions = self.registry.get_unevaluated_decisions(
                decision_id=decision_id,
                older_than_hours=24
            )

        # Check if we have enough samples
        if len(unevaluated_decisions) < self.min_samples:
            logger.debug(
                f"Not enough samples for {decision_id}: {len(unevaluated_decisions)} < {self.min_samples}"
            )
            return {
                "adapted": False,
                "reason": "insufficient_samples",
                "sample_count": len(unevaluated_decisions),
                "min_required": self.min_samples
            }

        # Evaluate outcomes
        outcomes = self._evaluate_decisions(unevaluated_decisions)

        if not outcomes:
            logger.warning(f"Failed to evaluate outcomes for {decision_id}")
            return {
                "adapted": False,
                "reason": "evaluation_failed",
                "sample_count": len(unevaluated_decisions)
            }

        # Compute parameter adjustments
        adjustments = self._compute_adjustments(decision_id, outcomes)

        if not adjustments:
            logger.debug(f"No adjustments needed for {decision_id}")
            return {
                "adapted": False,
                "reason": "no_adjustments",
                "sample_count": len(outcomes),
                "avg_success_score": sum(o.success_score for o in outcomes) / len(outcomes)
            }

        # Apply adjustments
        applied = {}
        if not dry_run:
            for param_name, new_value in adjustments.items():
                success = self.registry.update_parameter(
                    decision_id=decision_id,
                    param_name=param_name,
                    new_value=new_value,
                    reason="adaptive_learning",
                    based_on_records=[o.decision_record_id for o in outcomes]
                )
                applied[param_name] = success

            # Log adaptation
            avg_success_score = sum(o.success_score for o in outcomes) / len(outcomes)
            self.adaptation_history.append({
                "decision_id": decision_id,
                "adjustments": adjustments,
                "applied": applied,
                "sample_count": len(outcomes),
                "avg_success_score": avg_success_score
            })

            # Log to identity ledger
            # Get old parameter values for comparison
            old_params = self.registry.get_all_parameters(decision_id)
            parameters_updated = {}
            for param_name, new_value in adjustments.items():
                old_value = old_params.get(param_name, 0.0)
                parameters_updated[param_name] = {
                    "old": old_value,
                    "new": new_value
                }

            parameter_adapted_event(
                decision_id=decision_id,
                parameters_updated=parameters_updated,
                success_score=avg_success_score,
                sample_count=len(outcomes),
                meta={
                    "exploration_rate": self.exploration_rate,
                    "adaptation_rate": self.adaptation_rate,
                    "method": "epsilon_greedy"
                }
            )

        logger.info(
            f"Adapted {decision_id}: {len(adjustments)} parameters, "
            f"avg_score={sum(o.success_score for o in outcomes) / len(outcomes):.2f}"
        )

        return {
            "adapted": True,
            "decision_id": decision_id,
            "adjustments": adjustments,
            "applied": applied if not dry_run else None,
            "sample_count": len(outcomes),
            "avg_success_score": sum(o.success_score for o in outcomes) / len(outcomes),
            "dry_run": dry_run
        }

    def adapt_from_evaluated_decisions(
        self,
        decision_id: str,
        since_hours: int = 24,
        dry_run: bool = False
    ) -> Dict:
        """
        Adapt parameters from already-evaluated decisions.

        This is used for tasks where TaskOutcomeLinker has already
        evaluated and updated decision outcomes.

        Args:
            decision_id: Decision type to adapt
            since_hours: Look at decisions from last N hours
            dry_run: If True, compute adaptations but don't apply

        Returns:
            Dictionary with adaptation results
        """
        # Get evaluated decisions
        evaluated_decisions = self.registry.get_evaluated_decisions(
            decision_id=decision_id,
            since_hours=since_hours
        )

        # Check if we have enough samples
        if len(evaluated_decisions) < self.min_samples:
            logger.debug(
                f"Not enough evaluated samples for {decision_id}: "
                f"{len(evaluated_decisions)} < {self.min_samples}"
            )
            return {
                "adapted": False,
                "reason": "insufficient_samples",
                "sample_count": len(evaluated_decisions),
                "min_required": self.min_samples
            }

        # Convert to DecisionOutcome objects
        outcomes = []
        for dec in evaluated_decisions:
            outcome_details = dec.get("outcome_details", {})
            if outcome_details and "decision_record_id" in outcome_details:
                # Reconstruct DecisionOutcome from stored details
                outcome = DecisionOutcome(
                    decision_record_id=outcome_details["decision_record_id"],
                    success_score=dec["success_score"],
                    coherence_delta=outcome_details.get("coherence_delta", 0.0),
                    dissonance_delta=outcome_details.get("dissonance_delta", 0.0),
                    satisfaction_delta=outcome_details.get("satisfaction_delta", 0.0),
                    aborted=outcome_details.get("aborted", False),
                    abort_reason=outcome_details.get("abort_reason"),
                    evaluation_timestamp=datetime.fromisoformat(
                        outcome_details["evaluation_timestamp"]
                    )
                )
                outcomes.append(outcome)

        if not outcomes:
            logger.warning(f"No valid outcomes found for {decision_id}")
            return {
                "adapted": False,
                "reason": "no_valid_outcomes",
                "sample_count": len(evaluated_decisions)
            }

        # Compute parameter adjustments
        adjustments = self._compute_adjustments(decision_id, outcomes)

        if not adjustments:
            logger.debug(f"No adjustments needed for {decision_id}")
            return {
                "adapted": False,
                "reason": "no_adjustments",
                "sample_count": len(outcomes),
                "avg_success_score": sum(o.success_score for o in outcomes) / len(outcomes)
            }

        # Apply adjustments
        applied = {}
        if not dry_run:
            for param_name, new_value in adjustments.items():
                success = self.registry.update_parameter(
                    decision_id=decision_id,
                    param_name=param_name,
                    new_value=new_value,
                    reason="adaptive_learning_from_tasks",
                    based_on_records=[o.decision_record_id for o in outcomes]
                )
                applied[param_name] = success

            # Log adaptation
            avg_success_score = sum(o.success_score for o in outcomes) / len(outcomes)
            self.adaptation_history.append({
                "decision_id": decision_id,
                "adjustments": adjustments,
                "applied": applied,
                "sample_count": len(outcomes),
                "avg_success_score": avg_success_score,
                "method": "from_evaluated_decisions"
            })

            # Log to identity ledger
            old_params = self.registry.get_all_parameters(decision_id)
            parameters_updated = {}
            for param_name, new_value in adjustments.items():
                old_value = old_params.get(param_name, 0.0)
                parameters_updated[param_name] = {
                    "old": old_value,
                    "new": new_value
                }

            parameter_adapted_event(
                decision_id=decision_id,
                parameters_updated=parameters_updated,
                success_score=avg_success_score,
                sample_count=len(outcomes),
                meta={
                    "exploration_rate": self.exploration_rate,
                    "adaptation_rate": self.adaptation_rate,
                    "method": "epsilon_greedy_from_tasks",
                    "since_hours": since_hours
                }
            )

        logger.info(
            f"Adapted {decision_id} from evaluated decisions: "
            f"{len(adjustments)} parameters, avg_score={sum(o.success_score for o in outcomes) / len(outcomes):.2f}"
        )

        return {
            "adapted": True,
            "decision_id": decision_id,
            "adjustments": adjustments,
            "applied": applied if not dry_run else None,
            "sample_count": len(outcomes),
            "avg_success_score": sum(o.success_score for o in outcomes) / len(outcomes),
            "dry_run": dry_run,
            "method": "from_evaluated_decisions"
        }

    def _evaluate_decisions(self, decisions: List[Dict]) -> List[DecisionOutcome]:
        """Evaluate outcomes for a list of decisions."""
        if not self.evaluator:
            logger.warning("No success evaluator configured, cannot evaluate decisions")
            return []

        outcomes = []
        for decision in decisions:
            try:
                outcome = self.evaluator.evaluate_decision_outcome(
                    decision_record_id=decision["record_id"],
                    pre_decision_metrics=decision.get("outcome_snapshot", {})
                )

                # Update decision record with outcome
                self.registry.update_decision_outcome(decision["record_id"], outcome)

                outcomes.append(outcome)

            except Exception as e:
                logger.error(f"Failed to evaluate decision {decision['record_id']}: {e}")

        return outcomes

    def _compute_adjustments(
        self,
        decision_id: str,
        outcomes: List[DecisionOutcome]
    ) -> Dict[str, float]:
        """
        Compute parameter adjustments based on outcomes.

        Strategy:
        1. Compute average success score
        2. If score > 0: good direction, continue (small adjustment)
        3. If score < 0: bad direction, reverse (larger adjustment)
        4. Add exploration noise
        """
        # Get current parameters
        current_params = self.registry.get_all_parameters(decision_id)
        if not current_params:
            logger.warning(f"No parameters found for {decision_id}")
            return {}

        # Compute average success score
        avg_score = sum(o.success_score for o in outcomes) / len(outcomes)

        adjustments = {}

        # Get parameter definitions from registry
        # TODO: Need to fetch full parameter objects with min/max/step
        # For now, use simple heuristics

        for param_name, current_value in current_params.items():
            # Epsilon-greedy exploration
            if random.random() < self.exploration_rate:
                # Random exploration
                adjustment = self._explore_parameter(param_name, current_value)
                logger.debug(f"Exploring {param_name}: {current_value:.3f} → {adjustment:.3f}")
            else:
                # Exploitation: adapt based on success
                adjustment = self._exploit_parameter(
                    param_name, current_value, avg_score
                )
                logger.debug(f"Exploiting {param_name}: {current_value:.3f} → {adjustment:.3f} (score={avg_score:.2f})")

            # Only include if changed
            if abs(adjustment - current_value) > 0.001:
                adjustments[param_name] = adjustment

        return adjustments

    def _explore_parameter(self, param_name: str, current_value: float) -> float:
        """
        Randomly explore parameter space.

        Uses small random perturbations to avoid local optima.
        """
        # TODO: Get actual bounds from registry
        # For now, use reasonable defaults based on parameter name

        if "threshold" in param_name:
            min_val, max_val = 0.0, 1.0
            step = 0.05
        elif "evidence" in param_name:
            min_val, max_val = 2.0, 15.0
            step = 1.0
        elif "confidence" in param_name or "boost" in param_name:
            min_val, max_val = 0.01, 0.20
            step = 0.01
        else:
            # Generic bounds
            min_val = max(0.0, current_value * 0.5)
            max_val = current_value * 1.5
            step = (max_val - min_val) / 20

        # Random step in either direction
        direction = random.choice([-1, 1])
        new_value = current_value + (direction * step * random.uniform(0.5, 1.5))

        # Clamp to bounds
        return max(min_val, min(max_val, new_value))

    def _exploit_parameter(
        self,
        param_name: str,
        current_value: float,
        avg_success_score: float
    ) -> float:
        """
        Adjust parameter based on success score.

        Logic:
        - If score > 0.2: good direction, small increase
        - If score > 0: slight positive, tiny increase
        - If score < -0.2: bad direction, decrease
        - If score < 0: slight negative, tiny decrease
        - If score ≈ 0: no change
        """
        # TODO: Get actual bounds from registry
        if "threshold" in param_name:
            min_val, max_val = 0.0, 1.0
            step = 0.05
        elif "evidence" in param_name:
            min_val, max_val = 2.0, 15.0
            step = 1.0
        elif "confidence" in param_name or "boost" in param_name:
            min_val, max_val = 0.01, 0.20
            step = 0.01
        else:
            min_val = max(0.0, current_value * 0.5)
            max_val = current_value * 1.5
            step = (max_val - min_val) / 20

        # Determine adjustment direction and magnitude
        if abs(avg_success_score) < 0.05:
            # Neutral, no change
            return current_value

        if avg_success_score > 0:
            # Positive outcomes, continue in same direction
            # For thresholds, "same direction" depends on semantic meaning
            # promotion_threshold: higher is more conservative
            # deprecation_threshold: higher is more aggressive
            # For now, use generic increase
            adjustment = step * self.adaptation_rate * min(avg_success_score / 0.5, 1.0)
            new_value = current_value + adjustment
        else:
            # Negative outcomes, reverse direction
            adjustment = step * self.adaptation_rate * min(abs(avg_success_score) / 0.5, 1.0)
            new_value = current_value - adjustment

        # Clamp to bounds
        return max(min_val, min(max_val, new_value))

    def get_telemetry(self) -> Dict:
        """Get telemetry data for monitoring."""
        return {
            "min_samples": self.min_samples,
            "exploration_rate": self.exploration_rate,
            "adaptation_rate": self.adaptation_rate,
            "adaptation_count": len(self.adaptation_history),
            "recent_adaptations": self.adaptation_history[-10:] if self.adaptation_history else []
        }
