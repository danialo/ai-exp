"""
Stream Classifier for HTN Self-Belief Decomposer.

Maps belief_type + temporal_scope to streams (identity, state, meta, relational).
"""

from dataclasses import dataclass
from typing import Optional

from src.utils.belief_config import BeliefSystemConfig, get_belief_config
from src.services.epistemics_rules import EpistemicFrame


@dataclass
class StreamClassification:
    """
    Result of stream classification.

    Attributes:
        primary_stream: Main stream (identity, state, meta, relational)
        secondary_stream: Optional secondary stream
        confidence: Confidence in classification
    """
    primary_stream: str
    secondary_stream: Optional[str]
    confidence: float


class StreamClassifier:
    """
    Map belief_type + temporal_scope to streams.

    Uses configuration-driven mapping table with fallbacks.
    """

    def __init__(self, config: Optional[BeliefSystemConfig] = None):
        """
        Initialize the stream classifier.

        Args:
            config: Configuration object. If None, loads from default.
        """
        if config is None:
            config = get_belief_config()

        self.mapping = config.streams.mapping
        self.valid_streams = set(config.streams.types)

    def classify(
        self,
        belief_type: str,
        epistemic_frame: EpistemicFrame
    ) -> StreamClassification:
        """
        Classify a belief into streams.

        Lookup order:
        1. mapping[belief_type][temporal_scope]
        2. mapping[belief_type]["default"]
        3. mapping["default"]["default"]

        Args:
            belief_type: Ontological category (TRAIT, PREFERENCE, etc.)
            epistemic_frame: Epistemic frame with temporal_scope

        Returns:
            StreamClassification with primary, secondary, confidence
        """
        temporal_scope = epistemic_frame.temporal_scope

        # Try exact match
        if belief_type in self.mapping:
            type_mapping = self.mapping[belief_type]

            if temporal_scope in type_mapping:
                return self._to_classification(type_mapping[temporal_scope])

            if 'default' in type_mapping:
                return self._to_classification(type_mapping['default'])

        # Try default mapping
        if 'default' in self.mapping:
            default_mapping = self.mapping['default']
            if 'default' in default_mapping:
                return self._to_classification(default_mapping['default'])

        # Hardcoded fallback
        return StreamClassification(
            primary_stream='identity',
            secondary_stream=None,
            confidence=0.5,
        )

    def _to_classification(self, mapping_entry) -> StreamClassification:
        """
        Convert a mapping entry to StreamClassification.

        mapping_entry can be:
        - StreamMapping dataclass
        - dict with primary, secondary, confidence
        """
        if hasattr(mapping_entry, 'primary'):
            # StreamMapping from config
            return StreamClassification(
                primary_stream=mapping_entry.primary,
                secondary_stream=mapping_entry.secondary,
                confidence=mapping_entry.confidence,
            )
        elif isinstance(mapping_entry, dict):
            return StreamClassification(
                primary_stream=mapping_entry.get('primary', 'identity'),
                secondary_stream=mapping_entry.get('secondary'),
                confidence=mapping_entry.get('confidence', 0.5),
            )
        else:
            return StreamClassification(
                primary_stream='identity',
                secondary_stream=None,
                confidence=0.5,
            )


def classify_stream(
    belief_type: str,
    epistemic_frame: EpistemicFrame
) -> StreamClassification:
    """
    Convenience function to classify a belief into streams.

    Args:
        belief_type: Ontological category
        epistemic_frame: Epistemic frame

    Returns:
        StreamClassification
    """
    classifier = StreamClassifier()
    return classifier.classify(belief_type, epistemic_frame)
