"""
Source Context Classifier for HTN Self-Belief Decomposer.

Computes source weight and context ID for experiences based on:
- Interaction mode (journaling, introspection, chat, roleplay)
- VAD signals (arousal reduces weight)
- Heuristic fallbacks (caps, profanity, exclamation density)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.belief_config import BeliefSystemConfig, get_belief_config


# Simple profanity word list for heuristic detection
PROFANITY_WORDS = {
    'fuck', 'fucking', 'fucked', 'shit', 'damn', 'damned', 'hell',
    'ass', 'asshole', 'bastard', 'bitch', 'crap', 'piss', 'dick',
}


@dataclass
class SourceContext:
    """
    Result of source context classification.

    Attributes:
        mode: Detected interaction mode
        source_weight: Weight for this source [0.0, 1.0]
        context_id: Unique context identifier
        details: Additional information about classification
    """
    mode: str
    source_weight: float
    context_id: str
    details: Dict[str, Any] = field(default_factory=dict)


def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """
    Get a nested value from a dictionary using dot notation.

    Args:
        data: Dictionary to traverse
        path: Dot-separated path (e.g., "affect.arousal")

    Returns:
        Value at path or None if not found
    """
    keys = path.split('.')
    value = data

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None

    return value


class SourceContextClassifier:
    """
    Classify experiences to compute source weight and context ID.

    Source weight reflects how reliable a source is for belief extraction:
    - Journaling mode: High weight (introspective, deliberate)
    - Normal chat: Medium weight
    - Roleplay: Low weight (may not reflect actual beliefs)
    - High arousal: Reduced weight (emotional state may distort)
    """

    def __init__(self, config: Optional[BeliefSystemConfig] = None):
        """
        Initialize the classifier.

        Args:
            config: Configuration object. If None, loads from default.
        """
        if config is None:
            config = get_belief_config()

        self.config = config.source_context
        self.context_config = config.context

    def classify(self, experience: Any) -> SourceContext:
        """
        Classify an experience to compute source weight and context ID.

        Args:
            experience: Experience object with content, affect, session_id, etc.

        Returns:
            SourceContext with mode, weight, context_id, and details
        """
        # Extract relevant data from experience
        content = getattr(experience, 'content', {})
        if isinstance(content, dict):
            text = content.get('text', '')
            structured = content.get('structured', {})
        else:
            text = str(content)
            structured = {}

        affect = getattr(experience, 'affect', {})
        if not isinstance(affect, dict):
            affect = {}

        session_id = getattr(experience, 'session_id', None)
        experience_id = getattr(experience, 'id', 'unknown')

        # Determine mode
        mode = self._determine_mode(structured, affect)

        # Compute base weight from mode
        base_weight = self.config.mode_weights.get(
            mode,
            self.config.mode_weights.get('unknown', 0.7)
        )

        # Track adjustments for logging
        details: Dict[str, Any] = {
            'method_used': 'mode',
            'base_weight': base_weight,
            'penalties_applied': [],
        }

        weight = base_weight

        # Apply VAD adjustment if enabled
        if self.config.vad.enabled:
            arousal = _get_nested_value(affect, self.config.vad.arousal_field)
            if arousal is not None and isinstance(arousal, (int, float)):
                vad_penalty = self.config.vad.arousal_weight * arousal
                weight = max(0.0, min(1.0, weight - vad_penalty))
                details['vad_arousal'] = arousal
                details['vad_penalty'] = vad_penalty
                details['method_used'] = 'mode+vad'

        # Apply heuristic fallbacks if enabled and VAD not used
        elif self.config.heuristic_fallback.enabled:
            penalties = self._compute_heuristic_penalties(text)
            total_penalty = sum(penalties.values())
            if total_penalty > 0:
                weight = max(0.0, weight - total_penalty)
                details['heuristic_penalties'] = penalties
                details['penalties_applied'] = list(penalties.keys())
                details['method_used'] = 'mode+heuristics'

        # Compute context ID
        context_id = self._compute_context_id(
            session_id=session_id,
            mode=mode,
            experience_id=experience_id,
        )

        return SourceContext(
            mode=mode,
            source_weight=round(weight, 4),
            context_id=context_id,
            details=details,
        )

    def _determine_mode(
        self,
        structured: Dict[str, Any],
        affect: Dict[str, Any]
    ) -> str:
        """
        Determine the interaction mode for an experience.

        Args:
            structured: Structured content metadata
            affect: Affect metadata

        Returns:
            Mode string
        """
        # Check for explicit mode in metadata
        if self.config.mode_field:
            mode = structured.get(self.config.mode_field)
            if mode and mode in self.config.mode_weights:
                return mode

        # Check for mode indicators in structured data
        if structured.get('is_journaling'):
            return 'journaling'
        if structured.get('is_introspection'):
            return 'introspection'
        if structured.get('is_roleplay'):
            return 'roleplay'

        # Check affect signals for heated mode
        arousal = _get_nested_value(affect, 'vad.a')
        if arousal is not None and arousal > 0.7:
            return 'heated'

        return 'unknown'

    def _compute_heuristic_penalties(self, text: str) -> Dict[str, float]:
        """
        Compute heuristic-based penalties for source weight.

        Args:
            text: Text content of the experience

        Returns:
            Dictionary of penalty types and amounts
        """
        penalties = {}
        heuristics = self.config.heuristic_fallback

        if not text:
            return penalties

        # Check caps ratio
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if caps_ratio > heuristics.caps_ratio_threshold:
                penalties['caps'] = heuristics.caps_penalty

        # Check exclamation density
        words = text.split()
        if words:
            exclaim_count = text.count('!')
            exclaim_density = exclaim_count / len(words)
            if exclaim_density > heuristics.exclaim_density_threshold:
                penalties['exclaim'] = heuristics.exclaim_penalty

        # Check profanity
        text_lower = text.lower()
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        if text_words & PROFANITY_WORDS:
            penalties['profanity'] = heuristics.profanity_penalty

        return penalties

    def _compute_context_id(
        self,
        session_id: Optional[str],
        mode: str,
        experience_id: str
    ) -> str:
        """
        Compute the context ID for an experience.

        Args:
            session_id: Session/conversation ID if available
            mode: Interaction mode
            experience_id: Experience ID for fallback

        Returns:
            Context ID string
        """
        strategy = self.context_config.strategy
        fallback = self.context_config.fallback

        if strategy == 'conversation_mode' and session_id:
            return f"{session_id}:{mode}"

        if fallback == 'experience_id':
            return experience_id

        return experience_id


def classify_experience(experience: Any) -> SourceContext:
    """
    Convenience function to classify a single experience.

    Args:
        experience: Experience object

    Returns:
        SourceContext result
    """
    classifier = SourceContextClassifier()
    return classifier.classify(experience)
