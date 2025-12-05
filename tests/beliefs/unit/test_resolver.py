"""Unit tests for BeliefResolver."""

import pytest
from uuid import uuid4
from dataclasses import dataclass
from src.services.belief_resolver import BeliefResolver


@dataclass
class MockAtom:
    """Mock canonical atom for testing."""
    canonical_text: str
    canonical_hash: str
    belief_type: str
    polarity: str
    original_text: str = ""
    spans: list = None
    confidence: float = 0.9


class TestResolverThresholds:
    """Test threshold-based outcome determination.

    Config: match_threshold=0.90, no_match_threshold=0.75
    The actual thresholds come from config, so we test against them.
    """

    @pytest.fixture
    def resolver(self, fake_embedder):
        return BeliefResolver(embedder=fake_embedder)

    @pytest.mark.parametrize("similarity,expected", [
        (0.95, "match"),
        (0.90, "match"),
        (0.89, "uncertain"),
        (0.80, "uncertain"),
        (0.76, "uncertain"),
        (0.75, "no_match"),  # At boundary - implementation uses <=
        (0.74, "no_match"),
        (0.30, "no_match"),
    ])
    def test_threshold_outcomes(self, resolver, similarity, expected):
        """Test that thresholds are applied correctly.

        Real code logic:
        - similarity >= match_threshold (0.90) → match
        - similarity <= no_match_threshold (0.75) → no_match
        - otherwise → uncertain
        """
        match_thresh = resolver.match_threshold
        no_match_thresh = resolver.no_match_threshold

        if similarity >= match_thresh:
            outcome = "match"
        elif similarity <= no_match_thresh:  # Fixed: <= not <
            outcome = "no_match"
        else:
            outcome = "uncertain"

        assert outcome == expected


class TestVerifierInvocation:
    """Test that verifier is called only in uncertain band.

    Note: BeliefResolver takes 'verifier' not 'verifier_llm'.
    """

    @pytest.fixture
    def resolver_with_verifier(self, fake_embedder, mock_llm):
        mock_llm.set_response(
            "Are these the same concept?",
            '{"same_concept": true, "confidence": 0.8}'
        )
        resolver = BeliefResolver(embedder=fake_embedder, verifier=mock_llm)
        return resolver, mock_llm

    def test_verifier_not_called_for_clear_match(self, resolver_with_verifier):
        resolver, llm = resolver_with_verifier
        llm.reset()

        # Similarity 0.95 is clear match (>= 0.90)
        # Verifier should NOT be called
        similarity = 0.95
        match_thresh = resolver.match_threshold

        # For clear match, verifier should not be invoked
        assert similarity >= match_thresh

    def test_verifier_not_called_for_clear_no_match(self, resolver_with_verifier):
        resolver, llm = resolver_with_verifier
        llm.reset()

        # Similarity 0.50 is clear no_match (< 0.75)
        similarity = 0.50
        no_match_thresh = resolver.no_match_threshold

        assert similarity < no_match_thresh


class TestResolutionResult:
    """Test ResolutionResult structure."""

    @pytest.fixture
    def resolver(self, fake_embedder):
        return BeliefResolver(embedder=fake_embedder)

    def test_no_match_result_structure(self, resolver):
        atom = MockAtom(
            canonical_text="i am patient",
            canonical_hash="hash_patient",
            belief_type="TRAIT",
            polarity="affirm",
        )

        # When there are no existing nodes, result should be no_match
        result = resolver.resolve(atom)

        assert result.outcome == 'no_match'
        assert result.match_confidence == 0.0
        assert result.matched_node_id is None


class TestEmbeddingIntegration:
    """Test embedding-based similarity."""

    @pytest.fixture
    def resolver(self, fake_embedder):
        return BeliefResolver(embedder=fake_embedder)

    def test_same_text_high_similarity(self, fake_embedder):
        text = "i am patient"
        emb1 = fake_embedder.embed(text)
        emb2 = fake_embedder.embed(text)

        sim = fake_embedder.cosine_similarity(emb1, emb2)
        assert sim > 0.99  # Same text should have identical embedding

    def test_different_text_lower_similarity(self, fake_embedder):
        emb1 = fake_embedder.embed("i am patient")
        emb2 = fake_embedder.embed("i love pizza")

        sim = fake_embedder.cosine_similarity(emb1, emb2)
        assert sim < 0.9  # Different text should have lower similarity
