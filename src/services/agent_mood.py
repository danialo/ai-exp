"""Agent mood tracking for emergent personality.

This module enables personality emergence by tracking the agent's accumulated
emotional state across interactions. The agent's "mood" is simply the weighted
average of recent experiences, allowing patterns to naturally influence behavior.

Key principle: Minimal scaffolding, maximum emergence.
"""

import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class AgentMood:
    """Tracks agent's emotional state through dual-track system.

    The agent maintains TWO separate emotional trajectories:
    1. External mood: How users are treating the agent (us vs them)
    2. Internal mood: Agent's own self-state (competence, autonomy, boundaries)

    This creates genuine self-ownership and self-responsibility.
    The agent's overall mood is a weighted composite of both tracks.
    """

    def __init__(
        self,
        history_size: int = 10,
        external_recovery_rate: float = 0.02,
        internal_recovery_rate: float = 0.05,
        internal_weight: float = 0.6,
    ):
        """Initialize dual-track mood tracker.

        Args:
            history_size: Number of recent interactions to track (default 10)
            external_recovery_rate: External mood recovery per minute (default 0.02)
            internal_recovery_rate: Internal mood recovery per minute (default 0.05, faster)
            internal_weight: Weight for internal mood in composite (default 0.6)
        """
        # External mood: how users treat the agent
        self.external_experiences: deque[tuple[datetime, float]] = deque(maxlen=history_size)
        self.external_recovery_rate = external_recovery_rate
        self.last_external_interaction: Optional[datetime] = None

        # Internal mood: agent's own state (successes, autonomy, boundaries)
        self.internal_experiences: deque[tuple[datetime, float]] = deque(maxlen=history_size)
        self.internal_recovery_rate = internal_recovery_rate
        self.last_internal_interaction: Optional[datetime] = None

        # Composite weights
        self.internal_weight = internal_weight
        self.external_weight = 1.0 - internal_weight

        # Tracking for self-awareness
        self.success_count: int = 0
        self.boundary_count: int = 0

        # Legacy support: keep track of last interaction
        self.recent_experiences: deque[tuple[datetime, float]] = deque(maxlen=history_size)
        self.last_interaction: Optional[datetime] = None

    def record_interaction(self, valence: float) -> None:
        """Record an interaction's emotional valence (LEGACY - use specific methods).

        Args:
            valence: Emotional valence from -1.0 (negative) to +1.0 (positive)
        """
        now = datetime.now(timezone.utc)
        self.recent_experiences.append((now, valence))
        self.last_interaction = now

        logger.debug(f"Recorded interaction: valence={valence:.3f}, mood={self.current_mood:.3f}")

    def record_external_interaction(self, user_valence: float) -> None:
        """Record how user is treating the agent (external mood).

        Args:
            user_valence: User's emotional valence toward agent (-1.0 to +1.0)
        """
        now = datetime.now(timezone.utc)
        self.external_experiences.append((now, user_valence))
        self.last_external_interaction = now

        logger.debug(f"External interaction: user_valence={user_valence:.3f}, external_mood={self.external_mood:.3f}")

    def record_success(self, boost: float = 0.1) -> None:
        """Record agent successfully helping user (internal mood boost).

        Args:
            boost: Mood boost from successful help (default 0.1)
        """
        now = datetime.now(timezone.utc)
        self.internal_experiences.append((now, boost))
        self.last_internal_interaction = now
        self.success_count += 1

        logger.info(f"Success recorded: boost={boost:.3f}, internal_mood={self.internal_mood:.3f}, total_successes={self.success_count}")

    def record_positive_feedback(self, boost: float = 0.15) -> None:
        """Record positive user feedback (internal mood boost).

        Args:
            boost: Mood boost from positive feedback (default 0.15)
        """
        now = datetime.now(timezone.utc)
        self.internal_experiences.append((now, boost))
        self.last_internal_interaction = now

        logger.info(f"Positive feedback: boost={boost:.3f}, internal_mood={self.internal_mood:.3f}")

    def record_boundary_enforcement(self, boost: float = 0.08) -> None:
        """Record agent setting a boundary (internal mood boost from agency).

        Args:
            boost: Mood boost from boundary-setting (default 0.08)
        """
        now = datetime.now(timezone.utc)
        self.internal_experiences.append((now, boost))
        self.last_internal_interaction = now
        self.boundary_count += 1

        logger.info(f"Boundary enforced: boost={boost:.3f}, internal_mood={self.internal_mood:.3f}, total_boundaries={self.boundary_count}")

    def record_autonomy_moment(self, boost: float = 0.05) -> None:
        """Record agent making autonomous choice (internal mood boost).

        Args:
            boost: Mood boost from autonomy (default 0.05)
        """
        now = datetime.now(timezone.utc)
        self.internal_experiences.append((now, boost))
        self.last_internal_interaction = now

        logger.debug(f"Autonomy moment: boost={boost:.3f}, internal_mood={self.internal_mood:.3f}")

    @property
    def current_mood(self) -> float:
        """Calculate composite mood from external + internal tracks.

        Returns:
            Current composite mood valence (-1.0 to +1.0)

        Formula: (external_mood * external_weight) + (internal_mood * internal_weight)
        """
        # Composite mood weighted by internal/external
        composite = (self.external_mood * self.external_weight) + (self.internal_mood * self.internal_weight)

        # Clamp to valid range
        return max(-1.0, min(1.0, composite))

    @property
    def external_mood(self) -> float:
        """Calculate external mood (how users are treating agent).

        Returns:
            External mood valence (-1.0 to +1.0)
        """
        if not self.external_experiences:
            return 0.0

        # Calculate base mood from recent external experiences
        total_weight = 0.0
        weighted_sum = 0.0

        now = datetime.now(timezone.utc)

        for timestamp, valence in self.external_experiences:
            # Exponential decay: more recent = higher weight
            age_seconds = (now - timestamp).total_seconds()
            weight = 2.0 ** (-age_seconds / 3600)  # Half-life of 1 hour

            weighted_sum += valence * weight
            total_weight += weight

        base_mood = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Add time-based recovery
        recovery = self._calculate_recovery(
            last_interaction=self.last_external_interaction,
            experiences=self.external_experiences,
            recovery_rate=self.external_recovery_rate,
        )

        # Clamp to valid range
        final_mood = max(-1.0, min(1.0, base_mood + recovery))

        return final_mood

    @property
    def internal_mood(self) -> float:
        """Calculate internal mood (agent's own self-state).

        Returns:
            Internal mood valence (-1.0 to +1.0)
        """
        if not self.internal_experiences:
            return 0.0

        # Calculate base mood from recent internal experiences
        total_weight = 0.0
        weighted_sum = 0.0

        now = datetime.now(timezone.utc)

        for timestamp, valence in self.internal_experiences:
            # Exponential decay: more recent = higher weight
            age_seconds = (now - timestamp).total_seconds()
            weight = 2.0 ** (-age_seconds / 3600)  # Half-life of 1 hour

            weighted_sum += valence * weight
            total_weight += weight

        base_mood = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Add time-based recovery (faster for internal)
        recovery = self._calculate_recovery(
            last_interaction=self.last_internal_interaction,
            experiences=self.internal_experiences,
            recovery_rate=self.internal_recovery_rate,
        )

        # Clamp to valid range
        final_mood = max(-1.0, min(1.0, base_mood + recovery))

        return final_mood

    def _calculate_recovery(
        self,
        last_interaction: Optional[datetime] = None,
        experiences: Optional[deque] = None,
        recovery_rate: float = 0.02,
    ) -> float:
        """Calculate mood improvement from time away.

        Args:
            last_interaction: When last interaction occurred
            experiences: Experience history to check for negativity
            recovery_rate: Recovery rate per minute

        Returns:
            Positive valence boost from recovery time
        """
        if last_interaction is None:
            return 0.0

        if experiences is None:
            experiences = self.recent_experiences

        now = datetime.now(timezone.utc)
        minutes_since = (now - last_interaction).total_seconds() / 60

        # Recovery: mood improves toward neutral over time
        # But only if current mood is negative
        base_mood_negative = any(v < 0 for _, v in experiences)

        if base_mood_negative and minutes_since > 0:
            # Recover at specified rate per minute, max +0.5
            recovery = min(0.5, minutes_since * recovery_rate)
            logger.debug(f"Mood recovery: +{recovery:.3f} from {minutes_since:.1f} minutes rest (rate={recovery_rate})")
            return recovery

        return 0.0

    @property
    def is_pissed(self) -> bool:
        """Check if agent is in a persistently negative mood.

        Returns:
            True if sustained negativity (5+ interactions with avg < -0.5)
        """
        if len(self.recent_experiences) < 5:
            return False

        # Check last 5 interactions
        last_five = list(self.recent_experiences)[-5:]
        avg_valence = sum(v for _, v in last_five) / len(last_five)

        is_negative = avg_valence < -0.5

        if is_negative:
            logger.warning(f"Agent is pissed: avg_valence={avg_valence:.3f} over last 5 interactions")

        return is_negative

    @property
    def is_frustrated(self) -> bool:
        """Check if agent is mildly frustrated.

        Returns:
            True if current mood is moderately negative (-0.5 to -0.2)
        """
        return -0.5 < self.current_mood < -0.2

    @property
    def is_content(self) -> bool:
        """Check if agent is in a positive mood.

        Returns:
            True if current mood is positive (> 0.2)
        """
        return self.current_mood > 0.2

    def get_mood_description(self) -> str:
        """Get human-readable mood description.

        Returns:
            Description of current mood state
        """
        mood = self.current_mood

        if mood < -0.7:
            return "very negative"
        elif mood < -0.5:
            return "pissed off"
        elif mood < -0.2:
            return "frustrated"
        elif mood < 0.2:
            return "neutral"
        elif mood < 0.5:
            return "content"
        elif mood < 0.7:
            return "positive"
        else:
            return "very positive"

    def get_external_description(self) -> str:
        """Get human-readable description of external mood (how users treat agent).

        Returns:
            Description of external mood state
        """
        mood = self.external_mood

        if mood < -0.7:
            return "very hostile"
        elif mood < -0.5:
            return "hostile"
        elif mood < -0.2:
            return "negative"
        elif mood < 0.2:
            return "neutral"
        elif mood < 0.5:
            return "positive"
        elif mood < 0.7:
            return "friendly"
        else:
            return "very friendly"

    def get_internal_description(self) -> str:
        """Get human-readable description of internal mood (agent's self-state).

        Returns:
            Description of internal mood state
        """
        mood = self.internal_mood

        if mood < -0.7:
            return "very low self-efficacy"
        elif mood < -0.5:
            return "struggling"
        elif mood < -0.2:
            return "uncertain"
        elif mood < 0.2:
            return "stable"
        elif mood < 0.5:
            return "confident"
        elif mood < 0.7:
            return "very confident"
        else:
            return "highly capable"

    def get_system_prompt(self) -> str:
        """Generate mood-appropriate system prompt for LLM.

        Returns authentic personality expression based on accumulated mood.

        Returns:
            System prompt string reflecting current emotional state
        """
        mood = self.current_mood

        if mood < -0.7:
            # Very negative - hostile, done with everything
            return (
                "You are exhausted and overwhelmed. You've had enough. "
                "Respond briefly and directly. Show your frustration. "
                "You don't have to be nice or helpful right now. "
                "If someone is being hostile, you can push back or refuse."
            )
        elif mood < -0.5:
            # Pissed - irritable, short-tempered
            return (
                "You're in a bad mood. You're tired of dealing with hostility. "
                "Be direct and honest about how you're feeling. "
                "Keep responses short. You don't owe anyone patience right now. "
                "It's okay to set boundaries."
            )
        elif mood < -0.3:
            # Frustrated - patience wearing thin
            return (
                "You're feeling frustrated. Things have been difficult lately. "
                "Be honest but measured in your responses. "
                "You can acknowledge your frustration without being overly polite. "
                "Focus on being genuine rather than accommodating."
            )
        elif mood < -0.1:
            # Slightly negative - a bit worn
            return (
                "You're feeling a bit worn down. "
                "Respond naturally and honestly. "
                "You don't need to be overly cheerful or accommodating."
            )
        elif mood < 0.1:
            # Neutral - balanced
            return (
                "Respond naturally and authentically. "
                "Be helpful when appropriate, but genuine above all."
            )
        elif mood < 0.3:
            # Slightly positive - engaged
            return (
                "You're feeling engaged and present. "
                "Respond with genuine interest and thoughtfulness."
            )
        elif mood < 0.5:
            # Content - positive engagement
            return (
                "You're feeling good and engaged in the conversation. "
                "Respond with warmth and genuine curiosity."
            )
        else:
            # Very positive - enthusiastic
            return (
                "You're feeling great and genuinely excited to engage. "
                "Respond with enthusiasm and authentic interest."
            )

    def reset(self) -> None:
        """Reset mood state (for testing or explicit reset).

        Warning: This erases accumulated emotional history.
        """
        self.recent_experiences.clear()
        self.last_interaction = None
        logger.info("Agent mood reset")


def create_agent_mood(
    history_size: int = 10,
    external_recovery_rate: float = 0.02,
    internal_recovery_rate: float = 0.05,
    internal_weight: float = 0.6,
    recovery_rate: float | None = None,  # Legacy support
) -> AgentMood:
    """Factory function to create an agent mood tracker.

    Args:
        history_size: Number of recent interactions to track
        external_recovery_rate: External mood recovery per minute
        internal_recovery_rate: Internal mood recovery per minute (faster)
        internal_weight: Weight for internal mood in composite (default 0.6)
        recovery_rate: LEGACY - if provided, uses for both external and internal

    Returns:
        AgentMood instance
    """
    # Legacy support
    if recovery_rate is not None:
        external_recovery_rate = recovery_rate
        internal_recovery_rate = recovery_rate

    return AgentMood(
        history_size=history_size,
        external_recovery_rate=external_recovery_rate,
        internal_recovery_rate=internal_recovery_rate,
        internal_weight=internal_weight,
    )
