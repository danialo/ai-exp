"""User affect detection from message text.

Detects emotional valence from user messages using pattern matching and
linguistic cues. This feeds into the agent's mood system to create authentic
responses based on accumulated emotional experiences.

Approach: Lightweight pattern-based detection, no external ML models.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class AffectDetector:
    """Detects user emotional state from message text."""

    # Emotional markers with valence scores
    ANGER_MARKERS = {
        r'\b(bullshit|fuck|damn|shit|pissed|angry|mad)\b': -0.7,
        r'\b(hate|despise|disgusted|awful|terrible)\b': -0.6,
        r'\b(annoying|frustrating|irritating)\b': -0.5,
        r'\b(stupid|dumb|idiotic)\b': -0.5,
    }

    FRUSTRATION_MARKERS = {
        r'\b(confused|lost|stuck|struggling|frustrated)\b': -0.4,
        r'\b(not working|broken|failed|error)\b': -0.3,
        r'\b(why|how come|don\'t understand)\b': -0.2,
    }

    POSITIVE_MARKERS = {
        r'\b(thanks|thank you|appreciate|grateful)\b': 0.6,
        r'\b(great|awesome|excellent|perfect|amazing)\b': 0.7,
        r'\b(love|wonderful|fantastic)\b': 0.8,
        r'\b(helpful|useful|clear|makes sense)\b': 0.5,
        r'\b(good|nice|cool|neat)\b': 0.4,
    }

    # Arousal markers (energy/intensity level)
    HIGH_AROUSAL_MARKERS = {
        r'\b(urgent|emergency|quickly|hurry|asap|now)\b': 0.8,
        r'\b(excited|thrilled|pumped|energized)\b': 0.7,
        r'\b(panicking|freaking out|stressed|overwhelmed)\b': 0.8,
        r'\b(angry|furious|enraged|livid)\b': 0.9,
        r'\b(important|critical|crucial)\b': 0.6,
    }

    LOW_AROUSAL_MARKERS = {
        r'\b(calm|relaxed|peaceful|tranquil)\b': 0.2,
        r'\b(tired|exhausted|drained|sleepy)\b': 0.1,
        r'\b(bored|meh|whatever|don\'t care)\b': 0.1,
        r'\b(slowly|gradually|eventually)\b': 0.3,
    }

    # Dominance markers (control/power level)
    HIGH_DOMINANCE_MARKERS = {
        r'\b(must|need you to|require|demand)\b': 0.8,
        r'\b(do this|make sure|ensure)\b': 0.7,
        r'\b(I know|obviously|clearly)\b': 0.6,
        r'\b(immediately|right now)\b': 0.7,
    }

    LOW_DOMINANCE_MARKERS = {
        r'\b(please help|could you|would you mind)\b': 0.2,
        r'\b(not sure|uncertain|confused|lost)\b': 0.2,
        r'\b(maybe|perhaps|possibly|might)\b': 0.3,
        r'\b(if you can|if possible|when you have time)\b': 0.1,
        r'\?$': 0.4,  # Ends with question mark
    }

    def __init__(self):
        """Initialize affect detector."""
        # Compile regex patterns for efficiency
        self.anger_patterns = [(re.compile(p, re.IGNORECASE), v)
                               for p, v in self.ANGER_MARKERS.items()]
        self.frustration_patterns = [(re.compile(p, re.IGNORECASE), v)
                                     for p, v in self.FRUSTRATION_MARKERS.items()]
        self.positive_patterns = [(re.compile(p, re.IGNORECASE), v)
                                  for p, v in self.POSITIVE_MARKERS.items()]
        self.high_arousal_patterns = [(re.compile(p, re.IGNORECASE), v)
                                      for p, v in self.HIGH_AROUSAL_MARKERS.items()]
        self.low_arousal_patterns = [(re.compile(p, re.IGNORECASE), v)
                                     for p, v in self.LOW_AROUSAL_MARKERS.items()]
        self.high_dominance_patterns = [(re.compile(p, re.IGNORECASE), v)
                                        for p, v in self.HIGH_DOMINANCE_MARKERS.items()]
        self.low_dominance_patterns = [(re.compile(p, re.IGNORECASE), v)
                                       for p, v in self.LOW_DOMINANCE_MARKERS.items()]

    def detect(self, message: str) -> float:
        """Detect emotional valence from message.

        Args:
            message: User's message text

        Returns:
            Valence score from -1.0 (very negative) to +1.0 (very positive)
        """
        if not message.strip():
            return 0.0

        signals = []

        # Check for anger markers
        for pattern, valence in self.anger_patterns:
            if pattern.search(message):
                signals.append(valence)
                logger.debug(f"Anger marker detected: {pattern.pattern}")

        # Check for frustration markers
        for pattern, valence in self.frustration_patterns:
            if pattern.search(message):
                signals.append(valence)
                logger.debug(f"Frustration marker detected: {pattern.pattern}")

        # Check for positive markers
        for pattern, valence in self.positive_patterns:
            if pattern.search(message):
                signals.append(valence)
                logger.debug(f"Positive marker detected: {pattern.pattern}")

        # Check for stylistic markers
        stylistic_valence = self._detect_stylistic_affect(message)
        if stylistic_valence != 0.0:
            signals.append(stylistic_valence)

        # Combine signals
        if not signals:
            return 0.0

        # Average with slight bias toward extreme emotions
        avg_valence = sum(signals) / len(signals)

        # Amplify if multiple signals point same direction
        if len(signals) > 1:
            if all(s < 0 for s in signals):
                avg_valence *= 1.2  # Amplify negative
            elif all(s > 0 for s in signals):
                avg_valence *= 1.2  # Amplify positive

        # Clamp to valid range
        final_valence = max(-1.0, min(1.0, avg_valence))

        logger.info(f"Detected affect: {final_valence:.3f} from message: {message[:50]}...")

        return final_valence

    def _detect_stylistic_affect(self, message: str) -> float:
        """Detect affect from stylistic cues (caps, punctuation, etc).

        Args:
            message: User's message text

        Returns:
            Valence adjustment based on style
        """
        signals = []

        # ALL CAPS (usually anger/excitement)
        caps_words = re.findall(r'\b[A-Z]{3,}\b', message)
        if caps_words:
            # Context: If with anger words → anger, otherwise → excitement
            has_anger = any(re.search(p, message, re.IGNORECASE)
                           for p, _ in self.ANGER_MARKERS.items())
            if has_anger:
                signals.append(-0.5)  # Angry caps
            else:
                signals.append(0.3)  # Excited caps

        # Multiple exclamation marks
        exclamation_count = message.count('!')
        if exclamation_count >= 3:
            signals.append(-0.4)  # Usually frustration
        elif exclamation_count == 1:
            signals.append(0.1)  # Mild emphasis

        # Multiple question marks (confusion/frustration)
        question_count = message.count('?')
        if question_count >= 2:
            signals.append(-0.3)

        # Ellipsis (uncertainty, trailing off)
        if '...' in message or message.count('.') >= 4:
            signals.append(-0.1)

        return sum(signals) / len(signals) if signals else 0.0

    def detect_arousal(self, message: str) -> float:
        """Detect emotional arousal (energy/intensity) from message.

        Args:
            message: User's message text

        Returns:
            Arousal score from 0.0 (very calm) to 1.0 (very intense/energetic)
        """
        if not message.strip():
            return 0.5  # Neutral arousal

        signals = []

        # Check for high arousal markers
        for pattern, arousal in self.high_arousal_patterns:
            if pattern.search(message):
                signals.append(arousal)
                logger.debug(f"High arousal marker detected: {pattern.pattern}")

        # Check for low arousal markers
        for pattern, arousal in self.low_arousal_patterns:
            if pattern.search(message):
                signals.append(arousal)
                logger.debug(f"Low arousal marker detected: {pattern.pattern}")

        # Stylistic arousal indicators
        # ALL CAPS indicates high arousal
        caps_words = re.findall(r'\b[A-Z]{3,}\b', message)
        if caps_words:
            signals.append(0.8)

        # Multiple exclamation marks indicate high arousal
        exclamation_count = message.count('!')
        if exclamation_count >= 3:
            signals.append(0.9)
        elif exclamation_count >= 2:
            signals.append(0.7)
        elif exclamation_count == 1:
            signals.append(0.6)

        # Multiple question marks can indicate agitation
        question_count = message.count('?')
        if question_count >= 3:
            signals.append(0.7)

        # Calculate average arousal
        if not signals:
            return 0.5  # Neutral if no markers

        avg_arousal = sum(signals) / len(signals)

        # Clamp to valid range
        return max(0.0, min(1.0, avg_arousal))

    def detect_dominance(self, message: str) -> float:
        """Detect dominance (control/power) from message.

        Args:
            message: User's message text

        Returns:
            Dominance score from 0.0 (submissive/uncertain) to 1.0 (dominant/controlling)
        """
        if not message.strip():
            return 0.5  # Neutral dominance

        signals = []

        # Check for high dominance markers
        for pattern, dominance in self.high_dominance_patterns:
            if pattern.search(message):
                signals.append(dominance)
                logger.debug(f"High dominance marker detected: {pattern.pattern}")

        # Check for low dominance markers
        for pattern, dominance in self.low_dominance_patterns:
            if pattern.search(message):
                signals.append(dominance)
                logger.debug(f"Low dominance marker detected: {pattern.pattern}")

        # Stylistic dominance indicators
        # Imperative sentences (commands) suggest high dominance
        if re.match(r'^[A-Z][^.!?]*[.!]$', message.strip()):
            # Starts with capital, ends with period or exclamation, no questions
            if '?' not in message:
                signals.append(0.7)

        # Questions often indicate lower dominance (seeking information)
        if message.strip().endswith('?'):
            signals.append(0.3)

        # Polite words indicate lower dominance
        if re.search(r'\b(please|kindly|sorry|excuse me)\b', message, re.IGNORECASE):
            signals.append(0.2)

        # Calculate average dominance
        if not signals:
            return 0.5  # Neutral if no markers

        avg_dominance = sum(signals) / len(signals)

        # Clamp to valid range
        return max(0.0, min(1.0, avg_dominance))

    def detect_vad(self, message: str) -> tuple[float, float, float]:
        """Detect all three VAD dimensions at once.

        Args:
            message: User's message text

        Returns:
            Tuple of (valence, arousal, dominance)
        """
        valence = self.detect(message)
        arousal = self.detect_arousal(message)
        dominance = self.detect_dominance(message)

        logger.info(f"Detected VAD: v={valence:.2f}, a={arousal:.2f}, d={dominance:.2f} from: {message[:50]}...")

        return (valence, arousal, dominance)

    def get_emotion_label(self, valence: float) -> str:
        """Convert valence to emotion label.

        Args:
            valence: Emotional valence

        Returns:
            Human-readable emotion label
        """
        if valence < -0.6:
            return "angry"
        elif valence < -0.3:
            return "frustrated"
        elif valence < -0.1:
            return "slightly negative"
        elif valence < 0.1:
            return "neutral"
        elif valence < 0.3:
            return "slightly positive"
        elif valence < 0.6:
            return "positive"
        else:
            return "very positive"


def create_affect_detector() -> AffectDetector:
    """Factory function to create an affect detector.

    Returns:
        AffectDetector instance
    """
    return AffectDetector()
