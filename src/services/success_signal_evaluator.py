"""
Success Signal Evaluator - Defines and measures what constitutes successful decisions.

Tracks:
- Coherence delta (from awareness loop)
- Dissonance delta (from belief consistency checker)
- Satisfaction delta (from feedback aggregator)

Computes overall success scores to guide parameter adaptation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from src.services.decision_framework import DecisionOutcome

logger = logging.getLogger(__name__)


@dataclass
class SuccessSignalBaselines:
    """Baseline metrics measured during cold start."""
    coherence: float = 0.70
    dissonance: float = 0.20
    satisfaction: float = 0.60


@dataclass
class SuccessSignalTargets:
    """Target metrics we're optimizing toward."""
    coherence: float = 0.85
    dissonance: float = 0.10
    satisfaction: float = 0.80


class SuccessSignalEvaluator:
    """Evaluates whether decisions led to successful outcomes."""

    def __init__(
        self,
        awareness_loop=None,
        belief_consistency_checker=None,
        feedback_aggregator=None
    ):
        """
        Args:
            awareness_loop: For coherence metrics
            belief_consistency_checker: For dissonance metrics
            feedback_aggregator: For satisfaction metrics
        """
        self.awareness = awareness_loop
        self.consistency = belief_consistency_checker
        self.feedback = feedback_aggregator

        # Baselines (learned during cold start or configured)
        self.baselines = SuccessSignalBaselines()

        # Targets (what we're optimizing for)
        self.targets = SuccessSignalTargets()

        # Metrics history for baseline computation
        self.coherence_history = []
        self.dissonance_history = []
        self.satisfaction_history = []

    def set_baselines(
        self,
        coherence: Optional[float] = None,
        dissonance: Optional[float] = None,
        satisfaction: Optional[float] = None
    ):
        """Set baseline metrics."""
        if coherence is not None:
            self.baselines.coherence = coherence
        if dissonance is not None:
            self.baselines.dissonance = dissonance
        if satisfaction is not None:
            self.baselines.satisfaction = satisfaction

        logger.info(
            f"Baselines set: coherence={self.baselines.coherence:.2f}, "
            f"dissonance={self.baselines.dissonance:.2f}, "
            f"satisfaction={self.baselines.satisfaction:.2f}"
        )

    def set_targets(
        self,
        coherence: Optional[float] = None,
        dissonance: Optional[float] = None,
        satisfaction: Optional[float] = None
    ):
        """Set target metrics."""
        if coherence is not None:
            self.targets.coherence = coherence
        if dissonance is not None:
            self.targets.dissonance = dissonance
        if satisfaction is not None:
            self.targets.satisfaction = satisfaction

        logger.info(
            f"Targets set: coherence={self.targets.coherence:.2f}, "
            f"dissonance={self.targets.dissonance:.2f}, "
            f"satisfaction={self.targets.satisfaction:.2f}"
        )

    def evaluate_decision_outcome(
        self,
        decision_record_id: str,
        pre_decision_metrics: dict,
        evaluation_window_hours: int = 24
    ) -> DecisionOutcome:
        """
        Evaluate a decision's outcome after a time window.

        Args:
            decision_record_id: ID of the decision to evaluate
            pre_decision_metrics: Metrics captured before decision
            evaluation_window_hours: How long to wait before evaluating

        Returns:
            DecisionOutcome with success score and component metrics
        """
        # Get current metrics
        coherence_now = self._get_current_coherence()
        dissonance_now = self._get_current_dissonance()
        satisfaction_now = self._get_current_satisfaction()

        # Compute deltas from pre-decision snapshot
        coherence_pre = pre_decision_metrics.get("coherence", self.baselines.coherence)
        dissonance_pre = pre_decision_metrics.get("dissonance", self.baselines.dissonance)
        satisfaction_pre = pre_decision_metrics.get("satisfaction", self.baselines.satisfaction)

        coherence_delta = coherence_now - coherence_pre
        dissonance_delta = dissonance_now - dissonance_pre
        satisfaction_delta = satisfaction_now - satisfaction_pre

        # Compute overall success score
        success_score = self.compute_success_score(
            coherence_delta, dissonance_delta, satisfaction_delta
        )

        outcome = DecisionOutcome(
            decision_record_id=decision_record_id,
            success_score=success_score,
            coherence_delta=coherence_delta,
            dissonance_delta=dissonance_delta,
            satisfaction_delta=satisfaction_delta,
            aborted=False
        )

        logger.info(
            f"Evaluated decision {decision_record_id}: score={success_score:.2f} "
            f"(Δcoh={coherence_delta:+.2f}, Δdis={dissonance_delta:+.2f}, Δsat={satisfaction_delta:+.2f})"
        )

        return outcome

    def compute_success_score(
        self,
        coherence_delta: float,
        dissonance_delta: float,
        satisfaction_delta: float,
        weights: Optional[Tuple[float, float, float]] = None
    ) -> float:
        """
        Compute overall success score from component deltas.

        Args:
            coherence_delta: Change in coherence
            dissonance_delta: Change in dissonance (negative is good)
            satisfaction_delta: Change in satisfaction
            weights: Optional (w_coh, w_dis, w_sat) weights (default: 0.4, 0.3, 0.3)

        Returns:
            Score ∈ [-1, 1]:
            - +1: Perfect improvement on all metrics
            - 0: No change from baseline
            - -1: Degradation on all metrics
        """
        if weights is None:
            weights = (0.4, 0.3, 0.3)  # Coherence weighted slightly higher

        w_coh, w_dis, w_sat = weights

        # Normalize deltas to [-1, 1] range
        # Coherence: positive delta is good
        coherence_range = self.targets.coherence - self.baselines.coherence
        coh_norm = coherence_delta / coherence_range if coherence_range > 0 else 0

        # Dissonance: negative delta is good (less dissonance)
        dissonance_range = self.baselines.dissonance - self.targets.dissonance
        dis_norm = -dissonance_delta / dissonance_range if dissonance_range > 0 else 0

        # Satisfaction: positive delta is good
        satisfaction_range = self.targets.satisfaction - self.baselines.satisfaction
        sat_norm = satisfaction_delta / satisfaction_range if satisfaction_range > 0 else 0

        # Clamp normalized values to [-1, 1]
        coh_norm = max(-1, min(1, coh_norm))
        dis_norm = max(-1, min(1, dis_norm))
        sat_norm = max(-1, min(1, sat_norm))

        # Weighted sum
        score = w_coh * coh_norm + w_dis * dis_norm + w_sat * sat_norm

        return float(max(-1, min(1, score)))

    def get_current_metrics(self) -> dict:
        """Get current values of all success metrics."""
        return {
            "coherence": self._get_current_coherence(),
            "dissonance": self._get_current_dissonance(),
            "satisfaction": self._get_current_satisfaction(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _get_current_coherence(self) -> float:
        """Get current coherence from awareness loop."""
        if self.awareness and hasattr(self.awareness, "last_sim_live"):
            return float(self.awareness.last_sim_live)
        return self.baselines.coherence

    def _get_current_dissonance(self) -> float:
        """Get current dissonance from belief consistency checker."""
        if self.consistency:
            # TODO: Wire into consistency checker for contradiction rate
            # For now, return baseline
            pass
        return self.baselines.dissonance

    def _get_current_satisfaction(self) -> float:
        """Get current satisfaction from feedback aggregator."""
        if self.feedback:
            try:
                # Use global feedback score as satisfaction proxy
                feedback_score, negative_score = self.feedback.global_score()
                # Map feedback [-1,1] to satisfaction [0,1]
                satisfaction = (feedback_score + 1) / 2
                return float(satisfaction)
            except Exception as e:
                logger.debug(f"Failed to get satisfaction from feedback: {e}")
                pass
        return self.baselines.satisfaction

    def record_metric_snapshot(self):
        """Record current metrics for baseline computation."""
        coherence = self._get_current_coherence()
        dissonance = self._get_current_dissonance()
        satisfaction = self._get_current_satisfaction()

        self.coherence_history.append((datetime.now(timezone.utc), coherence))
        self.dissonance_history.append((datetime.now(timezone.utc), dissonance))
        self.satisfaction_history.append((datetime.now(timezone.utc), satisfaction))

        # Keep only last 1000 samples
        if len(self.coherence_history) > 1000:
            self.coherence_history = self.coherence_history[-1000:]
        if len(self.dissonance_history) > 1000:
            self.dissonance_history = self.dissonance_history[-1000:]
        if len(self.satisfaction_history) > 1000:
            self.satisfaction_history = self.satisfaction_history[-1000:]

    def compute_baselines_from_history(self, window_days: int = 7):
        """Compute baselines from historical metrics."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

        # Filter recent history
        recent_coherence = [v for ts, v in self.coherence_history if ts > cutoff]
        recent_dissonance = [v for ts, v in self.dissonance_history if ts > cutoff]
        recent_satisfaction = [v for ts, v in self.satisfaction_history if ts > cutoff]

        # Compute means
        if recent_coherence:
            self.baselines.coherence = sum(recent_coherence) / len(recent_coherence)
        if recent_dissonance:
            self.baselines.dissonance = sum(recent_dissonance) / len(recent_dissonance)
        if recent_satisfaction:
            self.baselines.satisfaction = sum(recent_satisfaction) / len(recent_satisfaction)

        logger.info(
            f"Computed baselines from {window_days}d history: "
            f"coherence={self.baselines.coherence:.3f}, "
            f"dissonance={self.baselines.dissonance:.3f}, "
            f"satisfaction={self.baselines.satisfaction:.3f}"
        )

    def get_telemetry(self) -> dict:
        """Get telemetry data for monitoring."""
        return {
            "baselines": {
                "coherence": self.baselines.coherence,
                "dissonance": self.baselines.dissonance,
                "satisfaction": self.baselines.satisfaction
            },
            "targets": {
                "coherence": self.targets.coherence,
                "dissonance": self.targets.dissonance,
                "satisfaction": self.targets.satisfaction
            },
            "current": self.get_current_metrics(),
            "history_samples": {
                "coherence": len(self.coherence_history),
                "dissonance": len(self.dissonance_history),
                "satisfaction": len(self.satisfaction_history)
            }
        }
