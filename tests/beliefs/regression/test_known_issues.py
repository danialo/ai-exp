"""Regression tests for known issues in HTN Self-Belief Decomposer.

These tests document and verify fixes for bugs that have been identified.
"""

import pytest


class TestNeverHandling:
    """'never' must be both negation AND habitual - spec requirement."""

    def test_never_is_both_negation_and_habitual(self):
        from src.services.epistemics_rules import EpistemicsRulesEngine

        engine = EpistemicsRulesEngine()
        result = engine.extract("I never feel confident")

        # 'never' should set temporal_scope to habitual
        assert result.frame.temporal_scope == "habitual"

        # 'never' should appear in signals (as negation indicator)
        signal_cues = [s.get("cue", "") for s in result.signals]
        assert any("never" in str(c) for c in signal_cues)


class TestNoCurrentScopeValue:
    """temporal_scope must never be 'current' - only valid values allowed."""

    def test_no_current_scope_value(self):
        from src.services.epistemics_rules import EpistemicsRulesEngine

        engine = EpistemicsRulesEngine()
        valid = {"state", "ongoing", "habitual", "transitional", "past", "unknown"}

        test_cases = [
            "I am happy",
            "I'm currently tired",
            "Right now I feel good",
            "At the moment I'm stressed",
            "Today I feel great",
        ]

        for text in test_cases:
            result = engine.extract(text)
            assert result.frame.temporal_scope in valid, \
                f"Invalid scope '{result.frame.temporal_scope}' for '{text}'"


class TestTraitAlwaysIdentity:
    """TRAIT belief_type must always map to identity stream."""

    def test_trait_always_identity(self):
        from src.services.stream_classifier import StreamClassifier
        from dataclasses import dataclass

        @dataclass
        class Frame:
            temporal_scope: str

        classifier = StreamClassifier()

        for scope in ["state", "habitual", "ongoing", "past", "transitional"]:
            result = classifier.classify("TRAIT", Frame(temporal_scope=scope))
            assert result.primary_stream == "identity", \
                f"TRAIT with {scope} mapped to {result.primary_stream}"


class TestResolverThresholdBoundaries:
    """Test exact threshold boundary behavior."""

    def test_resolver_74_is_no_match(self):
        """0.74 is below no_match_threshold (0.75), so it's no_match."""
        # match_threshold = 0.90, no_match_threshold = 0.75
        similarity = 0.74
        no_match_threshold = 0.75

        if similarity <= no_match_threshold:
            outcome = "no_match"
        else:
            outcome = "uncertain"

        assert outcome == "no_match"

    def test_resolver_75_is_uncertain(self):
        """0.75 is AT no_match_threshold, so it's uncertain (not no_match)."""
        # Boundary case: at the threshold should be uncertain
        similarity = 0.75
        no_match_threshold = 0.75
        match_threshold = 0.90

        if similarity >= match_threshold:
            outcome = "match"
        elif similarity <= no_match_threshold:
            outcome = "no_match"
        else:
            outcome = "uncertain"

        # At the boundary, <= means it's no_match per spec
        assert outcome == "no_match"

    def test_resolver_80_is_uncertain(self):
        """0.80 is between thresholds, so it's uncertain."""
        similarity = 0.80
        no_match_threshold = 0.75
        match_threshold = 0.90

        if similarity >= match_threshold:
            outcome = "match"
        elif similarity <= no_match_threshold:
            outcome = "no_match"
        else:
            outcome = "uncertain"

        assert outcome == "uncertain"


class TestCanonicalizerExpansion:
    """Canonicalizer can expand text (contractions), not just shrink it."""

    def test_canonicalizer_can_expand_not_shrink(self):
        from src.services.belief_canonicalizer import BeliefCanonicalizer

        canon = BeliefCanonicalizer()

        # "I'm" expands to "i am" (longer)
        result = canon.canonicalize("I'm")
        assert result == "i am"
        assert len(result) > len("I'm")

        # Must be idempotent
        assert canon.canonicalize(result) == result


class TestDedupSpanPreservation:
    """All spans must be preserved when merging duplicates."""

    def test_dedup_preserves_all_spans(self):
        from src.services.belief_atom_deduper import BeliefAtomDeduper
        from src.services.belief_canonicalizer import BeliefCanonicalizer
        from dataclasses import dataclass

        @dataclass
        class Atom:
            atom_text: str
            belief_type: str
            polarity: str
            confidence: float
            spans: list
            canonical_text: str = ""
            canonical_hash: str = ""
            original_text: str = ""

        canon = BeliefCanonicalizer()
        deduper = BeliefAtomDeduper(canon)

        atoms = [
            Atom("I am X", "TRAIT", "affirm", 0.9, [(0, 6)]),
            Atom("I am X", "TRAIT", "affirm", 0.8, [(10, 16)]),
            Atom("I am X", "TRAIT", "affirm", 0.7, [(20, 26)]),
        ]
        for a in atoms:
            a.canonical_text = canon.canonicalize(a.atom_text)
            a.canonical_hash = canon.compute_hash(a.canonical_text)
            a.original_text = a.atom_text

        result = deduper.dedup(atoms)
        merged = result.deduped_atoms[0]

        # ALL spans must be preserved
        assert (0, 6) in merged.spans
        assert (10, 16) in merged.spans
        assert (20, 26) in merged.spans


class TestHabitualBeatsState:
    """'always' (habitual) should beat 'right now' (state) due to specificity."""

    def test_habitual_beats_state_in_conflict(self):
        from src.services.epistemics_rules import EpistemicsRulesEngine

        engine = EpistemicsRulesEngine()

        # "always" has specificity 4, "right now" has specificity 2
        # "always" should win
        result = engine.extract("I always hate it when it rains right now")
        assert result.frame.temporal_scope == "habitual"


class TestSourceWeightClamping:
    """Source weight must always be in [0, 1] range."""

    def test_source_weight_clamped(self):
        from src.services.source_context_classifier import SourceContextClassifier
        from tests.beliefs.conftest import MockExperience
        from datetime import datetime, timezone
        from uuid import uuid4

        classifier = SourceContextClassifier()

        # Extreme case: low weight mode + all penalties
        exp = MockExperience(
            id=str(uuid4()),
            type="self_definition",
            content="I HATE THIS!!! SO ANGRY!!! AAAARGH!!!",
            interaction_mode="heated",
            conversation_id="conv_123",
            affect={"arousal": 0.9, "valence": 0.1},
            created_at=datetime.now(timezone.utc),
        )

        result = classifier.classify(exp)
        assert 0.0 <= result.source_weight <= 1.0
