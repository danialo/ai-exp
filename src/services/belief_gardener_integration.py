"""
Belief Gardener Integration with Adaptive Decision Framework.

This module wires the belief gardener into the decision framework,
registering decision points and recording decisions for outcome evaluation.
"""

import logging
from datetime import datetime
from typing import Optional

from src.services.belief_gardener import BeliefLifecycleManager, GardenerConfig, BeliefGardener
from src.services.belief_store import BeliefStore, DeltaOp
from src.memory.raw_store import RawStore
from src.services.feedback_aggregator import FeedbackAggregator
from src.services.decision_framework import (
    get_decision_registry,
    Parameter,
    DecisionRegistry
)
from src.services.abort_condition_monitor import AbortConditionMonitor
from src.services.success_signal_evaluator import SuccessSignalEvaluator
from src.services.identity_ledger import (
    decision_made_event,
    decision_aborted_event,
    parameter_adapted_event
)

logger = logging.getLogger(__name__)


class AdaptiveBeliefLifecycleManager(BeliefLifecycleManager):
    """
    Enhanced belief lifecycle manager that integrates with decision framework.

    Registers decision points:
    - belief_formation
    - belief_promotion
    - belief_deprecation

    Records all decisions for outcome evaluation and parameter adaptation.
    Respects abort conditions from the abort monitor.
    """

    def __init__(
        self,
        belief_store: BeliefStore,
        raw_store: RawStore,
        config: GardenerConfig,
        feedback_aggregator: Optional[FeedbackAggregator] = None,
        decision_registry: Optional[DecisionRegistry] = None,
        abort_monitor: Optional[AbortConditionMonitor] = None,
        success_evaluator: Optional[SuccessSignalEvaluator] = None
    ):
        super().__init__(belief_store, raw_store, config, feedback_aggregator)

        self.decision_registry = decision_registry or get_decision_registry()
        self.abort_monitor = abort_monitor
        self.success_evaluator = success_evaluator

        # Register decision points
        self._register_decision_points()

    def _register_decision_points(self):
        """Register belief lifecycle decision points."""
        # Decision 1: Belief Formation
        self.decision_registry.register_decision(
            decision_id="belief_formation",
            subsystem="belief_gardener",
            description="Form new tentative belief from detected pattern",
            parameters={
                "min_evidence": Parameter(
                    name="min_evidence",
                    current_value=float(self.config.min_evidence_for_tentative),
                    min_value=2.0,
                    max_value=10.0,
                    step_size=1.0,
                    adaptation_rate=0.1
                ),
                "confidence_boost": Parameter(
                    name="confidence_boost",
                    current_value=self.config.evidence_confidence_boost,
                    min_value=0.01,
                    max_value=0.15,
                    step_size=0.01,
                    adaptation_rate=0.1
                )
            },
            success_metrics=["coherence_delta", "dissonance_delta"],
            context_features=["category", "evidence_count"]
        )

        # Decision 2: Belief Promotion
        self.decision_registry.register_decision(
            decision_id="belief_promotion",
            subsystem="belief_gardener",
            description="Promote tentative belief to asserted",
            parameters={
                "promotion_threshold": Parameter(
                    name="promotion_threshold",
                    current_value=0.2,  # From consider_promotion logic
                    min_value=0.1,
                    max_value=0.5,
                    step_size=0.05,
                    adaptation_rate=0.15
                ),
                "min_evidence_asserted": Parameter(
                    name="min_evidence_asserted",
                    current_value=float(self.config.min_evidence_for_asserted),
                    min_value=3.0,
                    max_value=15.0,
                    step_size=1.0,
                    adaptation_rate=0.1
                )
            },
            success_metrics=["coherence_delta", "user_validation", "stability"],
            context_features=["belief_confidence", "feedback_quality"]
        )

        # Decision 3: Belief Deprecation
        self.decision_registry.register_decision(
            decision_id="belief_deprecation",
            subsystem="belief_gardener",
            description="Deprecate low-confidence belief",
            parameters={
                "deprecation_threshold": Parameter(
                    name="deprecation_threshold",
                    current_value=self.config.deprecation_threshold,
                    min_value=0.1,
                    max_value=0.5,
                    step_size=0.05,
                    adaptation_rate=0.15
                )
            },
            success_metrics=["coherence_delta", "dissonance_reduction"],
            context_features=["belief_age", "contradiction_count"]
        )

        logger.info("Registered 3 belief lifecycle decision points")

    def form_belief_from_pattern(self, pattern) -> tuple[Optional[str], Optional[str]]:
        """
        Form belief from pattern with decision tracking.

        Overrides parent method to add decision framework integration.
        """
        # Check abort conditions
        if self.abort_monitor:
            aborted, reason = self.abort_monitor.check_abort_conditions()
            if aborted:
                logger.warning(f"Belief formation aborted: {reason}")
                return None, f"aborted_{reason}"

        # Get current parameters from decision registry
        params = self.decision_registry.get_all_parameters("belief_formation")
        if params:
            min_evidence = int(params.get("min_evidence", self.config.min_evidence_for_tentative))
        else:
            min_evidence = self.config.min_evidence_for_tentative

        # Check evidence threshold (adaptive)
        if pattern.evidence_count() < min_evidence:
            return None, "insufficient_evidence"

        # Capture pre-decision metrics
        pre_metrics = {}
        if self.success_evaluator:
            pre_metrics = self.success_evaluator.get_current_metrics()

        # Record decision context
        context = {
            "category": pattern.category,
            "evidence_count": pattern.evidence_count(),
            "confidence": pattern.confidence
        }

        # Call parent implementation
        belief_id, error = super().form_belief_from_pattern(pattern)

        # Record decision for evaluation
        if belief_id:
            record_id = self.decision_registry.record_decision(
                decision_id="belief_formation",
                context=context,
                parameters_used=params or {},
                outcome_snapshot=pre_metrics
            )

            logger.debug(f"Recorded belief formation decision: {record_id}")

            # Log to identity ledger
            decision_made_event(
                decision_id="belief_formation",
                decision_record_id=record_id,
                parameters_used=params or {},
                beliefs_touched=[belief_id],
                meta={
                    "category": pattern.category,
                    "evidence_count": pattern.evidence_count(),
                    "confidence": pattern.confidence
                }
            )

            # Notify abort monitor of belief formation
            if self.abort_monitor:
                self.abort_monitor.record_belief_formation()

        return belief_id, error

    def consider_promotion(self, belief_id: str, new_evidence: int) -> bool:
        """
        Consider promotion with decision tracking.

        Overrides parent method to add decision framework integration.
        """
        # Check abort conditions
        if self.abort_monitor:
            aborted, reason = self.abort_monitor.check_abort_conditions()
            if aborted:
                logger.warning(f"Belief promotion aborted: {reason}")
                return False

        # Get current parameters from decision registry
        params = self.decision_registry.get_all_parameters("belief_promotion")
        if params:
            promotion_threshold = params.get("promotion_threshold", 0.2)
            min_evidence = int(params.get("min_evidence_asserted", self.config.min_evidence_for_asserted))
        else:
            promotion_threshold = 0.2
            min_evidence = self.config.min_evidence_for_asserted

        # Check evidence threshold (adaptive)
        if new_evidence < min_evidence:
            return False

        # Get feedback score
        if self.feedback_aggregator:
            feedback_score, _, _ = self.feedback_aggregator.score(belief_id)
        else:
            feedback_score = 0.0

        # Check feedback threshold (adaptive)
        if feedback_score < promotion_threshold:
            logger.debug(f"Skip promotion {belief_id}: feedback {feedback_score:.2f} < {promotion_threshold:.2f}")
            return False

        # Capture pre-decision metrics
        pre_metrics = {}
        if self.success_evaluator:
            pre_metrics = self.success_evaluator.get_current_metrics()

        # Get belief for context
        beliefs = self.belief_store.get_current()
        belief = beliefs.get(belief_id)
        if not belief:
            return False

        # Record decision context
        context = {
            "belief_confidence": belief.confidence,
            "feedback_score": feedback_score,
            "evidence_count": new_evidence
        }

        # Attempt promotion via parent
        promoted = super().consider_promotion(belief_id, new_evidence)

        # Record decision for evaluation
        if promoted:
            record_id = self.decision_registry.record_decision(
                decision_id="belief_promotion",
                context=context,
                parameters_used=params or {},
                outcome_snapshot=pre_metrics
            )

            logger.debug(f"Recorded belief promotion decision: {record_id}")

            # Log to identity ledger
            decision_made_event(
                decision_id="belief_promotion",
                decision_record_id=record_id,
                parameters_used=params or {},
                beliefs_touched=[belief_id],
                meta={
                    "belief_confidence": belief.confidence,
                    "feedback_score": feedback_score,
                    "evidence_count": new_evidence
                }
            )

        return promoted

    def consider_deprecation(self, belief_id: str, decay_evidence: int = 0) -> bool:
        """
        Consider deprecation with decision tracking.

        Extends parent method (if it exists) to add decision framework integration.

        Args:
            belief_id: ID of belief to consider for deprecation
            decay_evidence: Evidence count for decay (ignored, for compatibility)
        """
        # Check abort conditions
        if self.abort_monitor:
            aborted, reason = self.abort_monitor.check_abort_conditions()
            if aborted:
                logger.warning(f"Belief deprecation aborted: {reason}")
                return False

        # Get current parameters from decision registry
        params = self.decision_registry.get_all_parameters("belief_deprecation")
        if params:
            deprecation_threshold = params.get("deprecation_threshold", self.config.deprecation_threshold)
        else:
            deprecation_threshold = self.config.deprecation_threshold

        # Get belief
        beliefs = self.belief_store.get_current()
        belief = beliefs.get(belief_id)
        if not belief:
            return False

        # Check if below threshold (adaptive)
        if belief.confidence >= deprecation_threshold:
            return False

        # Capture pre-decision metrics
        pre_metrics = {}
        if self.success_evaluator:
            pre_metrics = self.success_evaluator.get_current_metrics()

        # Record decision context
        context = {
            "belief_confidence": belief.confidence,
            "belief_age_days": (datetime.now() - belief.created_at).days if hasattr(belief, "created_at") else 0
        }

        # Perform deprecation
        try:
            self.belief_store.deprecate_belief(
                belief_id=belief_id,
                from_ver=belief.ver,
                updated_by="gardener_adaptive"
            )

            # Record decision for evaluation
            record_id = self.decision_registry.record_decision(
                decision_id="belief_deprecation",
                context=context,
                parameters_used=params or {},
                outcome_snapshot=pre_metrics
            )

            logger.info(f"Deprecated belief {belief_id}, recorded decision: {record_id}")

            # Log to identity ledger
            decision_made_event(
                decision_id="belief_deprecation",
                decision_record_id=record_id,
                parameters_used=params or {},
                beliefs_touched=[belief_id],
                meta={
                    "belief_confidence": belief.confidence,
                    "belief_age_days": (datetime.now() - belief.created_at).days if hasattr(belief, "created_at") else 0
                }
            )

            return True

        except Exception as e:
            logger.error(f"Failed to deprecate belief {belief_id}: {e}")
            return False


def create_adaptive_belief_lifecycle_manager(
    belief_store: BeliefStore,
    raw_store: RawStore,
    config: GardenerConfig,
    feedback_aggregator: Optional[FeedbackAggregator] = None,
    awareness_loop=None,
    belief_consistency_checker=None
) -> AdaptiveBeliefLifecycleManager:
    """
    Factory function to create fully-wired adaptive belief lifecycle manager.

    Args:
        belief_store: Belief version control store
        raw_store: Experience raw store
        config: Gardener configuration
        feedback_aggregator: Feedback scoring system
        awareness_loop: For coherence monitoring
        belief_consistency_checker: For dissonance monitoring

    Returns:
        Fully-wired adaptive belief lifecycle manager
    """
    # Create success signal evaluator
    success_evaluator = SuccessSignalEvaluator(
        awareness_loop=awareness_loop,
        belief_consistency_checker=belief_consistency_checker,
        feedback_aggregator=feedback_aggregator
    )

    # Create abort condition monitor
    abort_monitor = AbortConditionMonitor(
        awareness_loop=awareness_loop,
        belief_consistency_checker=belief_consistency_checker,
        feedback_aggregator=feedback_aggregator,
        belief_store=belief_store,
        success_evaluator=success_evaluator
    )

    # Create adaptive lifecycle manager
    manager = AdaptiveBeliefLifecycleManager(
        belief_store=belief_store,
        raw_store=raw_store,
        config=config,
        feedback_aggregator=feedback_aggregator,
        decision_registry=get_decision_registry(),
        abort_monitor=abort_monitor,
        success_evaluator=success_evaluator
    )

    # Create BeliefGardener with standard lifecycle manager first
    gardener = BeliefGardener(
        belief_store=belief_store,
        raw_store=raw_store,
        config=config,
        feedback_aggregator=feedback_aggregator
    )

    # Replace the lifecycle_manager with our adaptive version
    gardener.lifecycle_manager = manager

    logger.info("Created adaptive belief lifecycle manager with decision framework integration")

    return gardener
