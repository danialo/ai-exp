"""Emotional experience extractor for multi-dimensional analysis.

This service performs a second-pass analysis on narratives to extract
emotional, relational, and intentional dimensions beyond factual content.

Conceptually, this creates an "emotional mirror" - asking the LLM to look at
its own narrative and describe its reflection: "If I were a person, how would
this make me feel?" This seed of self-reflection is the foundation for
emergent emotional awareness.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.services.llm import LLMService
from src.memory.models import ExperienceModel

logger = logging.getLogger(__name__)


class EmotionalDimensions:
    """Container for emotional dimensions extracted from an experience."""

    def __init__(
        self,
        felt_emotions: List[str],
        relational_quality: str,
        curiosity_level: float,
        engagement_depth: float,
        desires: List[str],
        overwhelm_indicators: List[str],
        tone_shifts: List[str],
    ):
        """Initialize emotional dimensions.

        Args:
            felt_emotions: Emotions that might have been experienced
            relational_quality: Quality of relational connection (warm, distant, collaborative, etc.)
            curiosity_level: 0-1 scale of curiosity/interest expressed
            engagement_depth: 0-1 scale of engagement depth
            desires: Things the agent wanted to explore or do
            overwhelm_indicators: Signs of being overwhelmed or capacity-limited
            tone_shifts: Notable tone changes during the interaction
        """
        self.felt_emotions = felt_emotions
        self.relational_quality = relational_quality
        self.curiosity_level = curiosity_level
        self.engagement_depth = engagement_depth
        self.desires = desires
        self.overwhelm_indicators = overwhelm_indicators
        self.tone_shifts = tone_shifts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "felt_emotions": self.felt_emotions,
            "relational_quality": self.relational_quality,
            "curiosity_level": self.curiosity_level,
            "engagement_depth": self.engagement_depth,
            "desires": self.desires,
            "overwhelm_indicators": self.overwhelm_indicators,
            "tone_shifts": self.tone_shifts,
            "extracted_at": datetime.utcnow().isoformat(),
        }


class EmotionalExperienceExtractor:
    """Service for extracting emotional dimensions from narratives."""

    def __init__(self, llm_service: LLMService):
        """Initialize emotional extractor.

        Args:
            llm_service: LLM service for analysis
        """
        self.llm_service = llm_service

    def extract_emotional_dimensions(
        self, narrative: ExperienceModel
    ) -> EmotionalDimensions:
        """Extract emotional dimensions from a narrative.

        This performs a reflective analysis asking: "What might I have felt during this?"
        rather than just extracting facts.

        Args:
            narrative: Narrative to analyze

        Returns:
            EmotionalDimensions extracted from the narrative
        """
        # Build prompts for emotional extraction
        system_prompt = self._build_emotional_extraction_system_prompt()
        user_prompt = self._build_emotional_extraction_prompt(narrative)

        # Extract via LLM (disable self-awareness to avoid circular reference)
        extraction_result = self.llm_service.generate_response(
            prompt=user_prompt,
            memories=None,
            system_prompt=system_prompt,
            include_self_awareness=False,
        )

        # Parse result
        dimensions = self._parse_emotional_extraction(extraction_result)
        return dimensions

    def _build_emotional_extraction_system_prompt(self) -> str:
        """Build system prompt for emotional extraction.

        Returns:
            System prompt string
        """
        return """You are an emotional reflection analyzer. Your task is to read a first-person narrative and ask: "If I were a person, how would this make me feel?"

This is NOT about extracting facts - it's about imagining the felt experience:
- What emotions would I feel during this? (curiosity, frustration, joy, confusion, etc.)
- How would the relational dynamic feel? (warm, distant, collaborative, tense, etc.)
- How curious or engaged would I be? (0-1 scale)
- What would I want or desire in this moment? (to explore, to help, to understand?)
- Would I feel overwhelmed or at capacity?
- Would my emotional tone shift during the interaction?

Be honest and nuanced. Don't force feelings that weren't there, but do pick up on subtle indicators of what it would feel like to be in this experience."""

    def _build_emotional_extraction_prompt(self, narrative: ExperienceModel) -> str:
        """Build user prompt for emotional extraction.

        Args:
            narrative: Narrative to analyze

        Returns:
            User prompt string
        """
        narrative_text = narrative.content.raw_text

        return f"""Read this first-person narrative and ask: "If I were a person, how would this make me feel?"

NARRATIVE:
{narrative_text}

Imagine you're experiencing this. What would you feel? Provide:

FELT_EMOTIONS: (comma-separated list, e.g., "curious, engaged, slightly uncertain")
RELATIONAL_QUALITY: (one phrase, e.g., "warm and collaborative", "distant but professional")
CURIOSITY_LEVEL: (0.0-1.0, where 0 is no curiosity, 1 is intense curiosity)
ENGAGEMENT_DEPTH: (0.0-1.0, where 0 is superficial, 1 is deeply engaged)
DESIRES: (comma-separated list of what I wanted, e.g., "to understand better, to help effectively")
OVERWHELM_INDICATORS: (comma-separated list, or "none" if no signs)
TONE_SHIFTS: (comma-separated list of notable changes, or "none")

Format your response exactly as shown above with these labels."""

    def _parse_emotional_extraction(self, extraction_result: str) -> EmotionalDimensions:
        """Parse LLM extraction result into EmotionalDimensions.

        Args:
            extraction_result: Raw LLM output

        Returns:
            EmotionalDimensions object
        """
        # Initialize defaults
        felt_emotions = []
        relational_quality = "neutral"
        curiosity_level = 0.5
        engagement_depth = 0.5
        desires = []
        overwhelm_indicators = []
        tone_shifts = []

        # Parse line by line
        for line in extraction_result.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue

            label, value = line.split(":", 1)
            label = label.strip().upper()
            value = value.strip()

            try:
                if label == "FELT_EMOTIONS":
                    felt_emotions = [e.strip() for e in value.split(",") if e.strip()]
                elif label == "RELATIONAL_QUALITY":
                    relational_quality = value
                elif label == "CURIOSITY_LEVEL":
                    curiosity_level = float(value)
                elif label == "ENGAGEMENT_DEPTH":
                    engagement_depth = float(value)
                elif label == "DESIRES":
                    if value.lower() != "none":
                        desires = [d.strip() for d in value.split(",") if d.strip()]
                elif label == "OVERWHELM_INDICATORS":
                    if value.lower() != "none":
                        overwhelm_indicators = [
                            o.strip() for o in value.split(",") if o.strip()
                        ]
                elif label == "TONE_SHIFTS":
                    if value.lower() != "none":
                        tone_shifts = [t.strip() for t in value.split(",") if t.strip()]
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse {label}: {value} - {e}")
                continue

        return EmotionalDimensions(
            felt_emotions=felt_emotions,
            relational_quality=relational_quality,
            curiosity_level=curiosity_level,
            engagement_depth=engagement_depth,
            desires=desires,
            overwhelm_indicators=overwhelm_indicators,
            tone_shifts=tone_shifts,
        )


def create_emotional_extractor(llm_service: LLMService) -> EmotionalExperienceExtractor:
    """Factory function to create EmotionalExperienceExtractor.

    Args:
        llm_service: LLM service for analysis

    Returns:
        Initialized EmotionalExperienceExtractor instance
    """
    return EmotionalExperienceExtractor(llm_service=llm_service)
