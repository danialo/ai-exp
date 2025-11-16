"""Autonomous Goal Generation Service.

Detects patterns in system behavior and generates goal proposals that
can be reviewed and adopted. Enables collaborative goal-driven system
where both user and system contribute goals.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from src.services.goal_store import (
    GoalStore,
    GoalDefinition,
    GoalCategory,
    GoalState,
    GoalSource,
)
from src.services.belief_store import BeliefStore

logger = logging.getLogger(__name__)


# ===== Data Structures =====

@dataclass
class GoalProposal:
    """A proposed goal from pattern detection.

    Proposals are evaluated and may become actual goals if they pass
    safety checks and have sufficient confidence.
    """

    # What (required)
    text: str  # "Improve test coverage for auth module"
    category: GoalCategory
    pattern_detected: str  # "test_coverage_drop"
    evidence: Dict[str, Any]  # {"old": 0.85, "new": 0.62, "module": "auth"}
    confidence: float  # How confident detector is (0-1)
    estimated_value: float  # How valuable is this? (0-1)
    estimated_effort: float  # How much work? (0-1)
    estimated_risk: float  # How risky? (0-1)
    detector_name: str

    # Identification (with defaults)
    proposal_id: str = field(default_factory=lambda: f"prop_{uuid4().hex[:8]}")

    # Alignment (with defaults)
    aligns_with: List[str] = field(default_factory=list)  # Belief IDs
    contradicts: List[str] = field(default_factory=list)  # Belief IDs

    # Metadata (with defaults)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None  # Proposal expiry

    def __post_init__(self):
        """Set default expiration."""
        if self.expires_at is None:
            self.expires_at = self.detected_at + timedelta(days=7)


# ===== Pattern Detector Interface =====

class PatternDetector(ABC):
    """Base class for pattern detectors.

    Each detector observes a specific aspect of the system and proposes
    goals when it detects opportunities for improvement.
    """

    @abstractmethod
    async def detect(self) -> List[GoalProposal]:
        """Scan for patterns and return goal proposals.

        Returns:
            List of goal proposals (may be empty)
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Detector identifier (e.g., 'test_coverage_detector')."""
        pass

    @abstractmethod
    def scan_interval_minutes(self) -> int:
        """How often to run this detector (in minutes)."""
        pass

    @abstractmethod
    def enabled(self) -> bool:
        """Whether this detector is currently enabled."""
        pass


# ===== Goal Generator =====

class GoalGenerator:
    """Autonomous goal generation from observed patterns.

    Coordinates pattern detectors, evaluates proposals, and creates
    system-generated goals that enter the unified prioritization queue.

    Example:
        >>> generator = GoalGenerator(
        ...     goal_store=goal_store,
        ...     belief_store=belief_store,
        ...     detectors=[TestCoverageDetector(), TaskFailureDetector()]
        ... )
        >>> proposals = await generator.scan_for_opportunities()
        >>> for proposal in proposals:
        ...     goal = generator.create_system_goal(proposal)
    """

    def __init__(
        self,
        goal_store: GoalStore,
        belief_store: Optional[BeliefStore] = None,
        detectors: Optional[List[PatternDetector]] = None,
        min_confidence: float = 0.7,
        max_system_goals_per_day: int = 10,
        max_goals_per_detector_per_day: int = 3,
        belief_alignment_threshold: float = 0.5,
        auto_approve_threshold: float = 0.9,
    ):
        """Initialize goal generator.

        Args:
            goal_store: GoalStore for creating goals
            belief_store: BeliefStore for alignment checking
            detectors: List of pattern detectors
            min_confidence: Minimum confidence to create proposal (0-1)
            max_system_goals_per_day: Max system goals allowed per day
            max_goals_per_detector_per_day: Max goals per detector per day
            belief_alignment_threshold: Min alignment score with beliefs
            auto_approve_threshold: Confidence threshold for auto-approval
        """
        self.goal_store = goal_store
        self.belief_store = belief_store
        self.detectors = detectors or []
        self.min_confidence = min_confidence
        self.max_system_goals_per_day = max_system_goals_per_day
        self.max_goals_per_detector_per_day = max_goals_per_detector_per_day
        self.belief_alignment_threshold = belief_alignment_threshold
        self.auto_approve_threshold = auto_approve_threshold

        # Track recent proposals/goals for rate limiting
        self._recent_proposals: List[GoalProposal] = []
        self._detector_counts: Dict[str, int] = {}  # detector -> count today

        logger.info(
            f"GoalGenerator initialized with {len(self.detectors)} detectors, "
            f"min_confidence={min_confidence:.2f}"
        )

    def add_detector(self, detector: PatternDetector) -> None:
        """Add a pattern detector.

        Args:
            detector: Pattern detector to add
        """
        self.detectors.append(detector)
        logger.info(f"Added detector: {detector.name()}")

    async def scan_for_opportunities(self) -> List[GoalProposal]:
        """Scan all pattern detectors for goal opportunities.

        Returns:
            List of goal proposals that passed initial filtering
        """
        all_proposals = []

        for detector in self.detectors:
            if not detector.enabled():
                continue

            try:
                proposals = await detector.detect()

                # Filter by confidence
                valid_proposals = [
                    p for p in proposals
                    if p.confidence >= self.min_confidence
                ]

                if valid_proposals:
                    logger.info(
                        f"Detector '{detector.name()}' generated "
                        f"{len(valid_proposals)} proposals"
                    )

                all_proposals.extend(valid_proposals)

            except Exception as e:
                logger.error(f"Detector '{detector.name()}' failed: {e}", exc_info=True)

        return all_proposals

    def evaluate_proposal(self, proposal: GoalProposal) -> tuple[bool, Optional[str]]:
        """Evaluate if proposal should become a goal.

        Checks:
        1. Confidence threshold
        2. Rate limiting
        3. Duplicate detection
        4. Belief alignment
        5. Expiration

        Args:
            proposal: Goal proposal to evaluate

        Returns:
            (approved, reason) tuple
        """
        # Check expiration
        if proposal.expires_at and datetime.now(timezone.utc) > proposal.expires_at:
            return False, "expired"

        # Check confidence
        if proposal.confidence < self.min_confidence:
            return False, f"confidence_too_low_{proposal.confidence:.2f}"

        # Check rate limits
        if not self._check_rate_limits(proposal):
            return False, "rate_limit_exceeded"

        # Check for duplicates
        if self._is_duplicate(proposal):
            return False, "duplicate_goal_exists"

        # Check belief alignment
        if self.belief_store:
            alignment_ok, reason = self._check_belief_alignment(proposal)
            if not alignment_ok:
                return False, reason

        return True, None

    def create_system_goal(
        self,
        proposal: GoalProposal,
        state: Optional[GoalState] = None
    ) -> Optional[GoalDefinition]:
        """Create a system-generated goal from proposal.

        Args:
            proposal: Goal proposal to convert
            state: Goal state (defaults based on confidence)

        Returns:
            Created GoalDefinition or None if creation failed
        """
        # Determine state
        if state is None:
            # High confidence goals can be auto-approved
            if proposal.confidence >= self.auto_approve_threshold:
                state = GoalState.ADOPTED
                auto_approved = True
            else:
                state = GoalState.PROPOSED
                auto_approved = False
        else:
            auto_approved = (state == GoalState.ADOPTED)

        # Create goal
        goal = GoalDefinition(
            id=f"goal_auto_{uuid4().hex[:12]}",
            text=proposal.text,
            category=proposal.category,
            value=proposal.estimated_value,
            effort=proposal.estimated_effort,
            risk=proposal.estimated_risk,
            horizon_min_min=0,  # Can start immediately
            horizon_max_min=None,  # No hard deadline
            aligns_with=proposal.aligns_with,
            contradicts=proposal.contradicts,
            state=state,
            source=GoalSource.SYSTEM,
            created_by=proposal.detector_name,
            proposal_id=proposal.proposal_id,
            auto_approved=auto_approved,
            metadata={
                "pattern_detected": proposal.pattern_detected,
                "evidence": proposal.evidence,
                "confidence": proposal.confidence,
                "detected_at": proposal.detected_at.isoformat(),
            }
        )

        try:
            created = self.goal_store.create_goal(goal)

            # Track for rate limiting
            self._track_goal_creation(proposal)

            logger.info(
                f"Created system goal: {created.id} "
                f"(detector={proposal.detector_name}, "
                f"confidence={proposal.confidence:.2f}, "
                f"state={state.value})"
            )

            return created

        except Exception as e:
            logger.error(f"Failed to create system goal: {e}", exc_info=True)
            return None

    async def generate_and_create_goals(self) -> List[GoalDefinition]:
        """Full workflow: scan → evaluate → create.

        Returns:
            List of created goals
        """
        # Scan for opportunities
        proposals = await self.scan_for_opportunities()

        if not proposals:
            logger.debug("No proposals generated")
            return []

        created_goals = []

        for proposal in proposals:
            # Evaluate proposal
            approved, reason = self.evaluate_proposal(proposal)

            if not approved:
                logger.debug(
                    f"Proposal rejected: {proposal.text} "
                    f"(reason={reason}, detector={proposal.detector_name})"
                )
                continue

            # Create goal
            goal = self.create_system_goal(proposal)
            if goal:
                created_goals.append(goal)

        if created_goals:
            logger.info(
                f"Generated {len(created_goals)} system goals "
                f"from {len(proposals)} proposals"
            )

        return created_goals

    # ===== Internal Helpers =====

    def _check_rate_limits(self, proposal: GoalProposal) -> bool:
        """Check if rate limits allow creating this goal."""
        # Clean up old counts (daily reset)
        self._reset_daily_counts_if_needed()

        # Check global limit
        total_today = sum(self._detector_counts.values())
        if total_today >= self.max_system_goals_per_day:
            logger.warning(
                f"Global rate limit reached: {total_today}/{self.max_system_goals_per_day}"
            )
            return False

        # Check per-detector limit
        detector_count = self._detector_counts.get(proposal.detector_name, 0)
        if detector_count >= self.max_goals_per_detector_per_day:
            logger.warning(
                f"Detector rate limit reached for '{proposal.detector_name}': "
                f"{detector_count}/{self.max_goals_per_detector_per_day}"
            )
            return False

        return True

    def _is_duplicate(self, proposal: GoalProposal) -> bool:
        """Check if similar goal already exists."""
        # Get recent goals from same detector
        try:
            existing_goals = self.goal_store.list_goals(
                state=GoalState.PROPOSED,
                limit=100
            )

            # Check for similar text from same detector
            for goal in existing_goals:
                if (goal.source == GoalSource.SYSTEM and
                    goal.created_by == proposal.detector_name):
                    # Simple similarity check: same text or very similar
                    if goal.text == proposal.text:
                        return True

                    # Check pattern type
                    if (goal.metadata.get("pattern_detected") ==
                        proposal.pattern_detected):
                        return True

            return False

        except Exception as e:
            logger.error(f"Duplicate check failed: {e}")
            return False  # Fail open

    def _check_belief_alignment(
        self,
        proposal: GoalProposal
    ) -> tuple[bool, Optional[str]]:
        """Check if proposal aligns with active beliefs.

        Returns:
            (aligned, reason) tuple
        """
        if not self.belief_store:
            return True, None  # Skip check if no belief store

        try:
            beliefs = self.belief_store.get_current()

            # Check contradictions
            for belief_id in proposal.contradicts:
                if belief_id in beliefs:
                    belief = beliefs[belief_id]
                    if belief.confidence >= 0.5:  # Active belief
                        return False, f"contradicts_belief_{belief_id}"

            # Check alignments
            alignment_score = 0.0
            if proposal.aligns_with:
                aligned_count = 0
                for belief_id in proposal.aligns_with:
                    if belief_id in beliefs:
                        belief = beliefs[belief_id]
                        if belief.confidence >= 0.5:
                            aligned_count += 1

                alignment_score = aligned_count / len(proposal.aligns_with)

            # Require minimum alignment
            if alignment_score < self.belief_alignment_threshold:
                return False, f"insufficient_alignment_{alignment_score:.2f}"

            return True, None

        except Exception as e:
            logger.error(f"Belief alignment check failed: {e}")
            return True, None  # Fail open

    def _track_goal_creation(self, proposal: GoalProposal) -> None:
        """Track goal creation for rate limiting."""
        self._recent_proposals.append(proposal)

        # Increment detector count
        detector_name = proposal.detector_name
        self._detector_counts[detector_name] = \
            self._detector_counts.get(detector_name, 0) + 1

    def _reset_daily_counts_if_needed(self) -> None:
        """Reset counts if it's a new day."""
        # Simple daily reset: clear if oldest proposal is >24h old
        if self._recent_proposals:
            oldest = self._recent_proposals[0].detected_at
            if datetime.now(timezone.utc) - oldest > timedelta(days=1):
                self._recent_proposals.clear()
                self._detector_counts.clear()
                logger.debug("Reset daily goal generation counts")

    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data for monitoring.

        Returns:
            Dictionary with generation metrics
        """
        return {
            "detectors_enabled": len([d for d in self.detectors if d.enabled()]),
            "detectors_total": len(self.detectors),
            "min_confidence": self.min_confidence,
            "max_system_goals_per_day": self.max_system_goals_per_day,
            "goals_created_today": sum(self._detector_counts.values()),
            "detector_counts": dict(self._detector_counts),
            "recent_proposals": len(self._recent_proposals),
        }
