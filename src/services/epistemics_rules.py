"""
Epistemics Rules Engine for HTN Self-Belief Decomposer.

Deterministic epistemic frame extraction using cue-based rules:
- Temporal scope (state, ongoing, habitual, transitional, past, unknown)
- Modality (certain, likely, possible, unsure)
- Degree (intensity modifier)
- Negation (polarity)

IMPORTANT: "never" is handled as BOTH negation AND habitual cue.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.belief_config import BeliefSystemConfig, get_belief_config


@dataclass
class MatchedCue:
    """A cue match found in text."""
    cue: str
    category: str
    position: int
    specificity: int = 0


@dataclass
class EpistemicFrame:
    """
    Epistemic qualifiers for a belief.

    Attributes:
        temporal_scope: state, ongoing, habitual, transitional, past, unknown
        modality: certain, likely, possible, unsure
        degree: Intensity [0.0, 1.0]
        conditional: Normalized condition if present
    """
    temporal_scope: str = "ongoing"
    modality: str = "certain"
    degree: float = 0.5
    conditional: Optional[str] = None


@dataclass
class EpistemicsResult:
    """
    Result of epistemic frame extraction.

    Attributes:
        frame: Extracted epistemic frame
        confidence: Confidence in the extraction
        signals: Matched cues and reasoning
        needs_llm_fallback: Whether LLM should be consulted
        detected_polarity: Polarity detected from negation
    """
    frame: EpistemicFrame
    confidence: float
    signals: List[Dict[str, Any]] = field(default_factory=list)
    needs_llm_fallback: bool = False
    detected_polarity: Optional[str] = None


class EpistemicsRulesEngine:
    """
    Deterministic epistemic frame extraction using cue matching.

    Order of operations:
    1. Detect negation -> affects polarity
    2. Detect modality cues -> caps certainty
    3. Detect temporal scope cues -> resolve conflicts
    4. Detect degree cues -> set intensity
    """

    def __init__(self, config: Optional[BeliefSystemConfig] = None):
        """
        Initialize the rules engine.

        Args:
            config: Configuration object. If None, loads from default.
        """
        if config is None:
            config = get_belief_config()

        self.config = config.epistemics
        self.cues = self._compile_cues()

    def _compile_cues(self) -> Dict[str, List[Tuple[str, re.Pattern]]]:
        """
        Compile cue patterns from config.

        Returns dict mapping category -> list of (cue_text, pattern).
        """
        compiled = {}

        # Negation cues
        compiled['negation'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.cues.negation
        ]

        # Modality cues
        compiled['modality_possible'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.cues.modality.possible
        ]
        compiled['modality_likely'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.cues.modality.likely
        ]
        compiled['modality_unsure'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.cues.modality.unsure
        ]

        # Temporal cues
        compiled['past'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.cues.past
        ]
        compiled['transitional'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.cues.transitional
        ]
        compiled['habitual_strong'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.cues.habitual_strong
        ]
        compiled['habitual_soft'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.cues.habitual_soft
        ]
        compiled['ongoing'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.cues.ongoing
        ]
        compiled['state'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.cues.state
        ]

        # Degree cues
        compiled['degree_strong'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.degree.strong
        ]
        compiled['degree_moderate'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.degree.moderate
        ]
        compiled['degree_weak'] = [
            (cue, re.compile(r'\b' + re.escape(cue) + r'\b', re.IGNORECASE))
            for cue in self.config.degree.weak
        ]

        return compiled

    def extract(self, atom_text: str) -> EpistemicsResult:
        """
        Extract epistemic frame using rule-based cue matching.

        Args:
            atom_text: Belief atom text

        Returns:
            EpistemicsResult with frame, confidence, signals
        """
        text_lower = atom_text.lower()
        all_signals: List[Dict[str, Any]] = []

        # Step 1: Detect negation
        polarity, negation_signals = self._detect_negation(text_lower)
        all_signals.extend(negation_signals)

        # Step 2: Detect modality
        modality, modality_confidence, modality_signals = self._detect_modality(text_lower)
        all_signals.extend(modality_signals)

        # Step 3: Detect temporal scope
        temporal_scope, temporal_signals = self._detect_temporal_scope(text_lower)
        all_signals.extend(temporal_signals)

        # Step 4: Detect degree
        degree, degree_signals = self._detect_degree(text_lower)
        all_signals.extend(degree_signals)

        # Step 5: Detect conditional (simple heuristic)
        conditional = self._detect_conditional(atom_text)

        # Compute overall confidence
        # Higher confidence if we matched specific cues
        signal_count = len([s for s in all_signals if s.get('category') not in ('negation',)])
        if signal_count == 0:
            confidence = 0.4  # Default values, low confidence
        elif signal_count == 1:
            confidence = 0.7
        else:
            confidence = min(0.9, 0.6 + 0.1 * signal_count)

        # Check if LLM fallback needed
        needs_fallback = confidence < self.config.llm_fallback_threshold

        frame = EpistemicFrame(
            temporal_scope=temporal_scope,
            modality=modality,
            degree=degree,
            conditional=conditional,
        )

        return EpistemicsResult(
            frame=frame,
            confidence=confidence,
            signals=all_signals,
            needs_llm_fallback=needs_fallback,
            detected_polarity=polarity,
        )

    def _detect_negation(self, text: str) -> Tuple[Optional[str], List[Dict]]:
        """
        Detect negation in text.

        Returns (polarity_if_negated, signals).
        Handles double negation.
        """
        signals = []
        negation_count = 0

        for cue, pattern in self.cues['negation']:
            for match in pattern.finditer(text):
                negation_count += 1
                signals.append({
                    'cue': cue,
                    'category': 'negation',
                    'position': match.start(),
                })

                # SPECIAL CASE: "never" also counts as habitual
                if cue.lower() == 'never':
                    signals.append({
                        'cue': cue,
                        'category': 'habitual_strong',
                        'position': match.start(),
                        'specificity': self.config.temporal_specificity.get('habitual', 4),
                    })

        # Determine polarity from negation count
        if negation_count == 0:
            return None, signals
        elif negation_count % 2 == 1:
            # Odd negations = negated
            return 'deny', signals
        else:
            # Even negations = affirm (double negative)
            return 'affirm', signals

    def _detect_modality(self, text: str) -> Tuple[str, float, List[Dict]]:
        """
        Detect modality (certainty level) from cues.

        Returns (modality, confidence_cap, signals).
        """
        signals = []
        detected_modalities = []

        modality_categories = [
            ('modality_possible', 'possible'),
            ('modality_likely', 'likely'),
            ('modality_unsure', 'unsure'),
        ]

        for cat_key, modality_name in modality_categories:
            for cue, pattern in self.cues.get(cat_key, []):
                for match in pattern.finditer(text):
                    detected_modalities.append(modality_name)
                    signals.append({
                        'cue': cue,
                        'category': f'modality:{modality_name}',
                        'position': match.start(),
                    })

        if not detected_modalities:
            return self.config.default_modality, 1.0, signals

        # Use the most constraining modality found
        modality_order = ['unsure', 'possible', 'likely', 'certain']
        most_constraining = min(
            detected_modalities,
            key=lambda m: modality_order.index(m) if m in modality_order else 10
        )

        confidence_cap = self.config.modality_confidence_caps.get(most_constraining, 1.0)

        return most_constraining, confidence_cap, signals

    def _detect_temporal_scope(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Detect temporal scope from cues.

        Uses specificity_then_rightmost conflict resolution.
        """
        signals = []
        temporal_matches: List[MatchedCue] = []

        temporal_categories = [
            ('past', 'past'),
            ('transitional', 'transitional'),
            ('habitual_strong', 'habitual'),
            ('habitual_soft', 'habitual'),
            ('ongoing', 'ongoing'),
            ('state', 'state'),
        ]

        for cat_key, scope_name in temporal_categories:
            specificity = self.config.temporal_specificity.get(scope_name, 1)

            for cue, pattern in self.cues.get(cat_key, []):
                for match in pattern.finditer(text):
                    # Skip if already handled as negation+habitual for "never"
                    if cue.lower() == 'never' and cat_key == 'habitual_strong':
                        # Already added in negation detection, but add to matches
                        pass

                    temporal_matches.append(MatchedCue(
                        cue=cue,
                        category=scope_name,
                        position=match.start(),
                        specificity=specificity,
                    ))

                    signals.append({
                        'cue': cue,
                        'category': f'temporal:{scope_name}',
                        'position': match.start(),
                        'specificity': specificity,
                    })

        if not temporal_matches:
            return self.config.default_temporal_scope, signals

        # Resolve conflicts: specificity DESC, then position DESC (rightmost)
        temporal_matches.sort(key=lambda m: (m.specificity, m.position), reverse=True)
        winner = temporal_matches[0]

        return winner.category, signals

    def _detect_degree(self, text: str) -> Tuple[float, List[Dict]]:
        """
        Detect degree (intensity) from cues.
        """
        signals = []
        detected_degrees = []

        degree_categories = [
            ('degree_strong', 'strong'),
            ('degree_moderate', 'moderate'),
            ('degree_weak', 'weak'),
        ]

        for cat_key, degree_name in degree_categories:
            for cue, pattern in self.cues.get(cat_key, []):
                for match in pattern.finditer(text):
                    detected_degrees.append(degree_name)
                    signals.append({
                        'cue': cue,
                        'category': f'degree:{degree_name}',
                        'position': match.start(),
                    })

        if not detected_degrees:
            return self.config.degree_values.get('default', 0.5), signals

        # Use strongest degree found (degree_strong > moderate > weak)
        degree_priority = {'strong': 3, 'moderate': 2, 'weak': 1}
        strongest = max(detected_degrees, key=lambda d: degree_priority.get(d, 0))

        return self.config.degree_values.get(strongest, 0.5), signals

    def _detect_conditional(self, text: str) -> Optional[str]:
        """
        Detect conditional clause (simple heuristic).
        """
        # Look for common conditional patterns
        conditional_patterns = [
            r'\bif\s+(.+?)\bthen\b',
            r'\bwhen\s+(.+?),',
            r'\bunless\s+(.+)',
        ]

        for pattern in conditional_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None


def extract_epistemics(atom_text: str) -> EpistemicsResult:
    """
    Convenience function to extract epistemics.

    Args:
        atom_text: Belief atom text

    Returns:
        EpistemicsResult
    """
    engine = EpistemicsRulesEngine()
    return engine.extract(atom_text)
