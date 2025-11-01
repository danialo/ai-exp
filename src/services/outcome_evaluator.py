"""
Outcome Evaluator - Delayed credit assignment for belief decisions.

Computes multi-component rewards from coherence, conflict, stability,
and user validation signals. Manages eligibility traces to apportion
credit among provenance actors.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EligibilityTrace:
    """Eligibility trace for a single actor's contribution to a decision."""
    actor: str
    belief_id: str
    tag_weight: float  # Total weighted contribution
    alignment: float  # Semantic alignment of tags
    ts: float  # Decision timestamp
    decay_gamma: float = 0.9  # Decay factor per 30s


@dataclass
class OutcomeConfig:
    """Configuration for outcome evaluation."""
    enabled: bool = True

    # Outcome component weights
    w_coherence: float = 0.4  # Coherence improvement
    w_conflict: float = 0.2  # Conflict reduction
    w_stability: float = 0.2  # Temporal stability
    w_validation: float = 0.2  # User validation

    # Evaluation horizons
    horizon_short_hours: float = 2.0  # Short-term evaluation
    horizon_long_hours: float = 24.0  # Long-term evaluation

    # Trust update weights
    alpha_short: float = 0.15  # Weight for short-horizon updates
    alpha_long: float = 1.0  # Weight for long-horizon updates

    # Stability tracking
    stability_window_days: int = 7  # Window for flip detection

    # Validation tracking
    validation_window_hours: int = 24  # Window for user validation

    # Minimum deltas
    min_confidence_delta: float = 0.05  # Minimum change to count
    min_reward_magnitude: float = 0.1  # Minimum |r| for update


@dataclass
class PendingEvaluation:
    """Scheduled evaluation for a belief decision."""
    belief_id: str
    decision_ts: float
    eligibility: Dict[str, float]  # actor → contribution
    horizon: str  # "short" or "long"
    scheduled_ts: float  # When to evaluate


class OutcomeEvaluator:
    """
    Evaluates belief decisions and updates provenance trust.

    Tracks eligibility traces, computes multi-component rewards,
    and schedules delayed evaluations at multiple horizons.
    """

    def __init__(
        self,
        provenance_trust: Any,  # ProvenanceTrust instance
        awareness_loop: Optional[Any] = None,  # For coherence signals
        belief_store: Optional[Any] = None,  # For stability tracking
        raw_store: Optional[Any] = None,  # For validation signals
        config: Optional[OutcomeConfig] = None
    ):
        """
        Initialize outcome evaluator.

        Args:
            provenance_trust: ProvenanceTrust manager for trust updates
            awareness_loop: AwarenessLoop for coherence metrics
            belief_store: BeliefStore for stability tracking
            raw_store: RawStore for user validation signals
            config: Configuration
        """
        self.provenance_trust = provenance_trust
        self.awareness_loop = awareness_loop
        self.belief_store = belief_store
        self.raw_store = raw_store
        self.config = config or OutcomeConfig()

        # Eligibility traces: (belief_id, actor) → EligibilityTrace
        self.eligibility_traces: Dict[Tuple[str, str], EligibilityTrace] = {}

        # Pending evaluations
        self.pending_evals: List[PendingEvaluation] = []

        # Baseline coherence for drift subtraction
        self.baseline_coherence: Optional[float] = None
        self._coherence_history: List[Tuple[float, float]] = []  # (ts, coherence)

        # Last evaluation run
        self._last_eval_ts = time.time()

        logger.info(f"OutcomeEvaluator initialized (horizons: {self.config.horizon_short_hours}h / {self.config.horizon_long_hours}h)")

    def record_decision(
        self,
        belief_id: str,
        actor_contributions: Dict[str, float],
        now: Optional[float] = None
    ) -> None:
        """
        Record a belief decision with eligibility traces.

        Args:
            belief_id: Belief that was modified
            actor_contributions: Dict of actor → weighted contribution
            now: Current timestamp
        """
        if not self.config.enabled:
            return

        if now is None:
            now = time.time()

        # Normalize contributions
        total = sum(actor_contributions.values())
        if total == 0:
            logger.debug(f"No contributions for {belief_id}, skipping")
            return

        normalized = {
            actor: contrib / total
            for actor, contrib in actor_contributions.items()
        }

        # Create eligibility traces
        for actor, contrib in normalized.items():
            key = (belief_id, actor)
            self.eligibility_traces[key] = EligibilityTrace(
                actor=actor,
                belief_id=belief_id,
                tag_weight=contrib,
                alignment=1.0,  # TODO: Include actual alignment
                ts=now
            )

        # Schedule evaluations
        short_eval = PendingEvaluation(
            belief_id=belief_id,
            decision_ts=now,
            eligibility=normalized,
            horizon="short",
            scheduled_ts=now + self.config.horizon_short_hours * 3600
        )
        long_eval = PendingEvaluation(
            belief_id=belief_id,
            decision_ts=now,
            eligibility=normalized,
            horizon="long",
            scheduled_ts=now + self.config.horizon_long_hours * 3600
        )

        self.pending_evals.append(short_eval)
        self.pending_evals.append(long_eval)

        logger.info(
            f"Recorded decision for {belief_id}: {len(normalized)} actors, "
            f"evals at {self.config.horizon_short_hours}h and {self.config.horizon_long_hours}h"
        )

    async def run_pending_evaluations(self, now: Optional[float] = None) -> int:
        """
        Run pending evaluations that are due.

        Args:
            now: Current timestamp

        Returns:
            Number of evaluations completed
        """
        if now is None:
            now = time.time()

        # Find due evaluations
        due = [e for e in self.pending_evals if e.scheduled_ts <= now]
        if not due:
            return 0

        # Run each evaluation
        completed = 0
        for eval_task in due:
            try:
                await self._evaluate_outcome(eval_task, now)
                self.pending_evals.remove(eval_task)
                completed += 1
            except Exception as e:
                logger.error(f"Evaluation failed for {eval_task.belief_id}: {e}")

        if completed > 0:
            logger.info(f"Completed {completed} pending evaluations")

        return completed

    async def _evaluate_outcome(
        self,
        eval_task: PendingEvaluation,
        now: float
    ) -> None:
        """
        Evaluate outcome for a belief decision and update trust.

        Args:
            eval_task: Pending evaluation to process
            now: Current timestamp
        """
        belief_id = eval_task.belief_id

        # Compute outcome components
        coherence_score = await self._compute_coherence_outcome(belief_id, eval_task.decision_ts, now)
        conflict_score = await self._compute_conflict_outcome(belief_id, eval_task.decision_ts, now)
        stability_score = await self._compute_stability_outcome(belief_id, eval_task.decision_ts, now)
        validation_score = await self._compute_validation_outcome(belief_id, eval_task.decision_ts, now)

        # Composite reward
        r_raw = (
            self.config.w_coherence * coherence_score +
            self.config.w_conflict * conflict_score +
            self.config.w_stability * stability_score +
            self.config.w_validation * validation_score
        )
        r = np.clip(r_raw, -1.0, 1.0)

        # Update trust for each actor
        alpha_multiplier = self.config.alpha_short if eval_task.horizon == "short" else self.config.alpha_long

        for actor, share in eval_task.eligibility.items():
            r_actor = share * r

            # Gate on minimum reward
            if abs(r_actor) >= self.config.min_reward_magnitude:
                self.provenance_trust.update_trust(actor, r_actor, now)

        logger.info(
            f"Outcome evaluation ({eval_task.horizon}): {belief_id} → r={r:.3f} "
            f"(coh={coherence_score:.2f}, conf={conflict_score:.2f}, "
            f"stab={stability_score:.2f}, val={validation_score:.2f})"
        )

    async def _compute_coherence_outcome(
        self,
        belief_id: str,
        decision_ts: float,
        now: float
    ) -> float:
        """
        Compute coherence improvement score ∈ [-1, 1].

        Args:
            belief_id: Belief ID
            decision_ts: When decision was made
            now: Current timestamp

        Returns:
            Coherence delta, normalized to [-1, 1]
        """
        if not self.awareness_loop:
            return 0.0

        try:
            # Get coherence before/after decision
            # Use sim_live as coherence proxy
            before_coherence = self._get_historical_coherence(decision_ts - 60)
            after_coherence = self.awareness_loop.last_sim_live

            if before_coherence is None:
                return 0.0

            delta_coh = after_coherence - before_coherence

            # Subtract baseline drift
            if self.baseline_coherence is not None:
                baseline_delta = after_coherence - self.baseline_coherence
                delta_coh -= baseline_delta * 0.5  # Partial correction

            # Normalize to [-1, 1] (assume max delta is ~0.5)
            normalized = np.clip(delta_coh / 0.5, -1.0, 1.0)

            return normalized

        except Exception as e:
            logger.error(f"Coherence outcome computation failed: {e}")
            return 0.0

    async def _compute_conflict_outcome(
        self,
        belief_id: str,
        decision_ts: float,
        now: float
    ) -> float:
        """
        Compute conflict reduction score ∈ [-1, 1].

        Returns:
            Conflict reduction delta, normalized
        """
        # TODO: Integrate with belief consistency checker
        # For now, return neutral
        return 0.0

    async def _compute_stability_outcome(
        self,
        belief_id: str,
        decision_ts: float,
        now: float
    ) -> float:
        """
        Compute stability score ∈ [-1, 1].

        Args:
            belief_id: Belief ID
            decision_ts: Decision timestamp
            now: Current timestamp

        Returns:
            Stability score: +1 if stable, -1 if flipped, 0 if inconclusive
        """
        if not self.belief_store:
            return 0.0

        try:
            # Check if belief still exists and hasn't been reversed
            current_beliefs = self.belief_store.get_current()
            if belief_id not in current_beliefs:
                # Belief was removed/deprecated
                return -1.0

            belief = current_beliefs[belief_id]

            # Check version history for flips
            history = self.belief_store.get_history(belief_id)
            if not history:
                return 0.0

            # Look for confidence reversals in window
            window_secs = self.config.stability_window_days * 86400
            window_start = decision_ts

            recent_versions = [
                v for v in history
                if v.get("timestamp", 0) >= window_start
            ]

            if len(recent_versions) < 2:
                # Not enough history, assume stable
                return 0.5

            # Check for ping-pong (confidence drops then rises, or vice versa)
            confidences = [v.get("confidence", 0.5) for v in recent_versions]
            if self._detect_ping_pong(confidences):
                return -1.0

            # Stable
            return 1.0

        except Exception as e:
            logger.error(f"Stability outcome computation failed: {e}")
            return 0.0

    async def _compute_validation_outcome(
        self,
        belief_id: str,
        decision_ts: float,
        now: float
    ) -> float:
        """
        Compute user validation score ∈ [-1, 1].

        Args:
            belief_id: Belief ID
            decision_ts: Decision timestamp
            now: Current timestamp

        Returns:
            Validation score based on user tags
        """
        # TODO: Query raw_store for user tags mentioning belief
        # For now, return neutral
        return 0.0

    def _get_historical_coherence(self, ts: float) -> Optional[float]:
        """Get coherence value at specific timestamp."""
        # Linear interpolation from history
        if not self._coherence_history:
            return None

        # Find closest samples
        before = [c for c in self._coherence_history if c[0] <= ts]
        after = [c for c in self._coherence_history if c[0] > ts]

        if before and after:
            # Interpolate
            t1, c1 = before[-1]
            t2, c2 = after[0]
            if t2 - t1 > 0:
                alpha = (ts - t1) / (t2 - t1)
                return c1 + alpha * (c2 - c1)

        if before:
            return before[-1][1]
        if after:
            return after[0][1]

        return None

    def _detect_ping_pong(self, confidences: List[float]) -> bool:
        """Detect ping-pong pattern in confidence values."""
        if len(confidences) < 3:
            return False

        # Check for direction reversals
        reversals = 0
        for i in range(1, len(confidences) - 1):
            delta_prev = confidences[i] - confidences[i-1]
            delta_next = confidences[i+1] - confidences[i]

            # Reversal if signs differ
            if delta_prev * delta_next < -0.01:  # Threshold to avoid noise
                reversals += 1

        # Two or more reversals = ping-pong
        return reversals >= 2

    def update_coherence_history(self, ts: float, coherence: float) -> None:
        """Update coherence history for baseline tracking."""
        self._coherence_history.append((ts, coherence))

        # Keep last 1000 samples
        if len(self._coherence_history) > 1000:
            self._coherence_history.pop(0)

        # Update baseline (rolling average)
        recent = [c for t, c in self._coherence_history if t > ts - 3600]
        if recent:
            self.baseline_coherence = sum(recent) / len(recent)

    def get_telemetry(self) -> Dict:
        """Get telemetry for status endpoint."""
        return {
            "pending_evaluations": len(self.pending_evals),
            "eligibility_traces": len(self.eligibility_traces),
            "baseline_coherence": self.baseline_coherence,
            "config": {
                "w_coherence": self.config.w_coherence,
                "w_conflict": self.config.w_conflict,
                "w_stability": self.config.w_stability,
                "w_validation": self.config.w_validation,
                "horizon_short_hours": self.config.horizon_short_hours,
                "horizon_long_hours": self.config.horizon_long_hours,
            }
        }


def create_outcome_evaluator(
    provenance_trust: Any,
    awareness_loop: Optional[Any] = None,
    belief_store: Optional[Any] = None,
    raw_store: Optional[Any] = None,
    config: Optional[OutcomeConfig] = None
) -> OutcomeEvaluator:
    """Factory function to create outcome evaluator."""
    return OutcomeEvaluator(
        provenance_trust=provenance_trust,
        awareness_loop=awareness_loop,
        belief_store=belief_store,
        raw_store=raw_store,
        config=config
    )
