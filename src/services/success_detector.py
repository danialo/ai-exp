"""Success detector for measuring agent helpfulness.

This module detects when the agent successfully helps the user, which
contributes to the agent's internal mood (competence/self-efficacy).

Success signals:
- Positive user sentiment (thanks, appreciation, etc.)
- Follow-up questions (user engaged with response)
- Neutral continuation (user wasn't hostile)
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class SuccessDetector:
    """Detects when agent successfully helps user."""

    def __init__(self):
        """Initialize success detector with heuristic patterns."""
        # Positive feedback patterns
        self.positive_patterns = [
            r"\b(thank|thanks|thx|ty|appreciate|helpful|great|good|nice|perfect|excellent|awesome|amazing)\b",
            r"\b(you('re| are) (good|great|helpful|awesome|amazing|the best))\b",
            r"👍|❤️|🙏|😊|😄|🎉",
        ]

        # Negative feedback patterns
        self.negative_patterns = [
            r"\b(fuck|shit|damn|terrible|useless|garbage|stupid|dumb|worthless|awful)\b",
            r"\b(you('re| are) (terrible|useless|garbage|bad|wrong))\b",
            r"🖕|😠|😡|💢",
        ]

        # Compile patterns
        self.positive_regex = re.compile("|".join(self.positive_patterns), re.IGNORECASE)
        self.negative_regex = re.compile("|".join(self.negative_patterns), re.IGNORECASE)

    def detect_feedback(self, message: str) -> Optional[str]:
        """Detect if message contains positive or negative feedback.

        Args:
            message: User's message

        Returns:
            "positive", "negative", or None
        """
        # Check for positive feedback
        if self.positive_regex.search(message):
            logger.info(f"Positive feedback detected: '{message[:50]}...'")
            return "positive"

        # Check for negative feedback
        if self.negative_regex.search(message):
            logger.info(f"Negative feedback detected: '{message[:50]}...'")
            return "negative"

        return None

    def detect_success(
        self,
        user_message: str,
        user_valence: float,
        previous_valence: Optional[float] = None,
    ) -> bool:
        """Detect if agent was successful in previous interaction.

        Args:
            user_message: Current user message
            user_valence: User's current emotional valence
            previous_valence: User's previous emotional valence (if available)

        Returns:
            True if agent appears to have been helpful
        """
        # Strong signal: explicit positive feedback
        feedback = self.detect_feedback(user_message)
        if feedback == "positive":
            logger.info("Success detected: explicit positive feedback")
            return True

        # Strong signal: user became more positive
        if previous_valence is not None and user_valence > previous_valence + 0.2:
            logger.info(f"Success detected: user mood improved ({previous_valence:.3f} → {user_valence:.3f})")
            return True

        # Medium signal: user is positive/neutral and engaged
        if user_valence >= 0.0 and len(user_message.strip()) > 10:
            # Not a complaint, reasonably engaged
            if feedback != "negative":
                logger.debug("Success detected: positive/neutral engagement")
                return True

        return False


def create_success_detector() -> SuccessDetector:
    """Factory function to create a success detector.

    Returns:
        SuccessDetector instance
    """
    return SuccessDetector()
