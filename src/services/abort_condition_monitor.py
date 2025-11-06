"""
Abort Condition Monitor - Watches for dangerous degradation patterns and halts autonomous decisions.

Monitors:
- Rising dissonance (contradiction spikes)
- Coherence drops (self-similarity degradation)
- Satisfaction collapse (negative user feedback)
- Runaway belief formation (too many beliefs too fast)

When abort conditions trigger, autonomous decision-making is paused and parameters are reset.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List
from collections import deque
import statistics

from src.services.identity_ledger import decision_aborted_event

logger = logging.getLogger(__name__)


@dataclass
class AbortThresholds:
    """Thresholds for abort conditions."""
    dissonance_sigma: float = 3.0  # Std devs above baseline
    coherence_sigma: float = 2.0  # Std devs below baseline
    negative_tag_threshold: float = 0.70  # 70% negative tags
    belief_rate_limit: int = 10  # Max beliefs/hour
    coherence_drop_window: int = 10  # Ticks to check
    dissonance_spike_window: int = 10  # Ticks to check


class AbortConditionMonitor:
    """Monitors for conditions that should halt autonomous decisions."""

    def __init__(
        self,
        awareness_loop=None,
        belief_consistency_checker=None,
        feedback_aggregator=None,
        belief_store=None,
        success_evaluator=None,
        thresholds: Optional[AbortThresholds] = None
    ):
        """
        Args:
            awareness_loop: For coherence monitoring
            belief_consistency_checker: For dissonance monitoring
            feedback_aggregator: For satisfaction monitoring
            belief_store: For belief formation rate monitoring
            success_evaluator: For baseline metrics
            thresholds: Custom abort thresholds
        """
        self.awareness = awareness_loop
        self.consistency = belief_consistency_checker
        self.feedback = feedback_aggregator
        self.belief_store = belief_store
        self.evaluator = success_evaluator

        self.thresholds = thresholds or AbortThresholds()

        # State tracking
        self.aborted = False
        self.abort_reason = None
        self.abort_timestamp = None

        # History buffers
        self.coherence_history = deque(maxlen=100)
        self.dissonance_history = deque(maxlen=100)
        self.belief_formation_times = deque(maxlen=100)

        logger.info("Abort condition monitor initialized")

    def check_abort_conditions(self) -> Tuple[bool, Optional[str]]:
        """
        Check if any abort conditions are triggered.

        Returns:
            (should_abort, reason)
        """
        # If already aborted, check if recovery time has passed
        if self.aborted:
            if self._check_recovery():
                logger.info("Abort condition recovered, resuming decisions")
                self.aborted = False
                self.abort_reason = None
                self.abort_timestamp = None
            else:
                return True, self.abort_reason

        # Check each abort condition
        checks = [
            self.check_dissonance_spike,
            self.check_coherence_drop,
            self.check_satisfaction_collapse,
            self.check_belief_runaway
        ]

        for check in checks:
            triggered, reason = check()
            if triggered:
                self._trigger_abort(reason)
                return True, reason

        return False, None

    def check_dissonance_spike(self) -> Tuple[bool, Optional[str]]:
        """
        Abort if contradiction count rising rapidly.

        Condition: dissonance > baseline + 3σ over N ticks
        """
        if not self.evaluator or not self.consistency:
            return False, None

        # Get current dissonance
        current_dissonance = self._get_current_dissonance()
        if current_dissonance is None:
            return False, None

        # Record in history
        self.dissonance_history.append(current_dissonance)

        # Need enough history to compute stats
        if len(self.dissonance_history) < self.thresholds.dissonance_spike_window:
            return False, None

        # Get baseline and compute threshold
        baseline = self.evaluator.baselines.dissonance
        recent = list(self.dissonance_history)[-self.thresholds.dissonance_spike_window:]

        # Compute mean and std dev of recent window
        if len(recent) < 2:
            return False, None

        recent_mean = statistics.mean(recent)
        recent_std = statistics.stdev(recent) if len(recent) > 1 else 0.1

        # Check if recent mean is significantly above baseline
        threshold = baseline + (self.thresholds.dissonance_sigma * recent_std)

        if recent_mean > threshold:
            reason = (
                f"Dissonance spike: {recent_mean:.3f} > threshold {threshold:.3f} "
                f"(baseline={baseline:.3f}, σ={recent_std:.3f})"
            )
            logger.warning(reason)
            return True, reason

        return False, None

    def check_coherence_drop(self) -> Tuple[bool, Optional[str]]:
        """
        Abort if coherence dropping significantly.

        Condition: coherence < baseline - 2σ over N ticks
        """
        if not self.evaluator or not self.awareness:
            return False, None

        # Get current coherence
        current_coherence = self._get_current_coherence()
        if current_coherence is None:
            return False, None

        # Record in history
        self.coherence_history.append(current_coherence)

        # Need enough history
        if len(self.coherence_history) < self.thresholds.coherence_drop_window:
            return False, None

        # Get baseline and compute threshold
        baseline = self.evaluator.baselines.coherence
        recent = list(self.coherence_history)[-self.thresholds.coherence_drop_window:]

        # Compute mean and std dev of recent window
        if len(recent) < 2:
            return False, None

        recent_mean = statistics.mean(recent)
        recent_std = statistics.stdev(recent) if len(recent) > 1 else 0.1

        # Check if recent mean is significantly below baseline
        threshold = baseline - (self.thresholds.coherence_sigma * recent_std)

        if recent_mean < threshold:
            reason = (
                f"Coherence drop: {recent_mean:.3f} < threshold {threshold:.3f} "
                f"(baseline={baseline:.3f}, σ={recent_std:.3f})"
            )
            logger.warning(reason)
            return True, reason

        return False, None

    def check_satisfaction_collapse(self) -> Tuple[bool, Optional[str]]:
        """
        Abort if user satisfaction plummeting.

        Condition: explicit negative tags > 70% over 24h
        """
        if not self.feedback:
            return False, None

        try:
            # Get global feedback score
            feedback_score, negative_score = self.feedback.global_score()

            # Check if negative tags dominate
            if negative_score > self.thresholds.negative_tag_threshold:
                reason = (
                    f"Satisfaction collapse: negative_score={negative_score:.2f} > "
                    f"threshold {self.thresholds.negative_tag_threshold}"
                )
                logger.warning(reason)
                return True, reason

        except Exception as e:
            logger.debug(f"Failed to check satisfaction: {e}")

        return False, None

    def check_belief_runaway(self) -> Tuple[bool, Optional[str]]:
        """
        Abort if beliefs forming too rapidly.

        Condition: >10 beliefs formed in 1 hour
        """
        # Check recent belief formation times
        one_hour_ago = time.time() - 3600

        # Count formations in last hour
        recent_formations = sum(1 for ts in self.belief_formation_times if ts > one_hour_ago)

        if recent_formations > self.thresholds.belief_rate_limit:
            reason = (
                f"Belief runaway: {recent_formations} beliefs formed in last hour "
                f"(limit={self.thresholds.belief_rate_limit})"
            )
            logger.warning(reason)
            return True, reason

        return False, None

    def record_belief_formation(self):
        """Record a belief formation event for rate limiting."""
        self.belief_formation_times.append(time.time())

    def _get_current_coherence(self) -> Optional[float]:
        """Get current coherence from awareness loop."""
        if self.awareness and hasattr(self.awareness, "last_sim_live"):
            return float(self.awareness.last_sim_live)
        return None

    def _get_current_dissonance(self) -> Optional[float]:
        """Get current dissonance from consistency checker."""
        if self.consistency:
            # TODO: Wire into consistency checker for actual contradiction count
            # For now, return None to skip this check
            pass
        return None

    def _trigger_abort(self, reason: str):
        """Trigger abort state."""
        self.aborted = True
        self.abort_reason = reason
        self.abort_timestamp = datetime.now(timezone.utc)

        logger.error(f"🚨 ABORT TRIGGERED: {reason}")

        # Log to identity ledger
        coherence_drop = None
        if self.evaluator:
            current_coh = self._get_current_coherence()
            if current_coh is not None:
                coherence_drop = self.evaluator.baselines.coherence - current_coh

        decision_aborted_event(
            abort_reason=reason,
            decision_id=None,  # Could be enhanced to track which decision was blocked
            coherence_drop=coherence_drop,
            meta={
                "timestamp": self.abort_timestamp.isoformat(),
                "aborted": True
            }
        )

    def _check_recovery(self) -> bool:
        """
        Check if system has recovered from abort condition.

        Recovery requires:
        - 1 hour since abort
        - Coherence back above baseline
        - Dissonance back below baseline + 1σ
        """
        if not self.abort_timestamp:
            return True

        # Check time elapsed
        elapsed = datetime.now(timezone.utc) - self.abort_timestamp
        if elapsed < timedelta(hours=1):
            return False

        # Check if metrics recovered
        coherence = self._get_current_coherence()
        dissonance = self._get_current_dissonance()

        if self.evaluator:
            if coherence and coherence < self.evaluator.baselines.coherence:
                logger.debug("Coherence still below baseline, not recovered")
                return False

            if dissonance and dissonance > self.evaluator.baselines.dissonance * 1.5:
                logger.debug("Dissonance still elevated, not recovered")
                return False

        return True

    def get_telemetry(self) -> dict:
        """Get telemetry data for monitoring."""
        return {
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "abort_timestamp": self.abort_timestamp.isoformat() if self.abort_timestamp else None,
            "thresholds": {
                "dissonance_sigma": self.thresholds.dissonance_sigma,
                "coherence_sigma": self.thresholds.coherence_sigma,
                "negative_tag_threshold": self.thresholds.negative_tag_threshold,
                "belief_rate_limit": self.thresholds.belief_rate_limit
            },
            "current_metrics": {
                "coherence": self._get_current_coherence(),
                "dissonance": self._get_current_dissonance(),
                "coherence_history_len": len(self.coherence_history),
                "dissonance_history_len": len(self.dissonance_history),
                "recent_belief_formations": sum(
                    1 for ts in self.belief_formation_times
                    if ts > time.time() - 3600
                )
            }
        }

    def reset(self):
        """Reset abort state (admin override)."""
        logger.warning("Abort condition manually reset")
        self.aborted = False
        self.abort_reason = None
        self.abort_timestamp = None
