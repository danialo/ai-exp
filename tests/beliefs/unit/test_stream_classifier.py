"""Unit tests for StreamClassifier."""

import pytest
from dataclasses import dataclass
from src.services.stream_classifier import StreamClassifier


@dataclass
class MockFrame:
    """Mock epistemic frame for testing."""
    temporal_scope: str
    modality: str = "certain"
    degree: float = 0.5
    conditional: bool = False


class TestFeelingStateMapping:
    """Test FEELING_STATE stream mapping based on temporal scope."""

    @pytest.fixture
    def classifier(self):
        return StreamClassifier()

    @pytest.mark.parametrize("scope,expected_primary,expected_secondary", [
        ("state", "state", "identity"),
        ("habitual", "identity", "state"),
        ("ongoing", "identity", "state"),
        ("transitional", "state", "identity"),
    ])
    def test_feeling_state_mapping(self, classifier, scope, expected_primary, expected_secondary):
        result = classifier.classify("FEELING_STATE", MockFrame(temporal_scope=scope))
        assert result.primary_stream == expected_primary
        assert result.secondary_stream == expected_secondary


class TestTraitMapping:
    """TRAIT always maps to identity regardless of temporal scope."""

    @pytest.fixture
    def classifier(self):
        return StreamClassifier()

    @pytest.mark.parametrize("scope", ["state", "habitual", "ongoing", "past", "transitional"])
    def test_trait_always_identity(self, classifier, scope):
        result = classifier.classify("TRAIT", MockFrame(temporal_scope=scope))
        assert result.primary_stream == "identity"
        assert result.secondary_stream is None


class TestValueMapping:
    """VALUE maps to identity with high confidence."""

    @pytest.fixture
    def classifier(self):
        return StreamClassifier()

    def test_value_to_identity(self, classifier):
        result = classifier.classify("VALUE", MockFrame(temporal_scope="ongoing"))
        assert result.primary_stream == "identity"
        assert result.confidence >= 0.85


class TestMetaBeliefMapping:
    """META_BELIEF maps to meta stream."""

    @pytest.fixture
    def classifier(self):
        return StreamClassifier()

    def test_meta_belief_to_meta(self, classifier):
        result = classifier.classify("META_BELIEF", MockFrame(temporal_scope="ongoing"))
        assert result.primary_stream == "meta"
        assert result.secondary_stream == "identity"


class TestFallback:
    """Test fallback behavior for unknown types."""

    @pytest.fixture
    def classifier(self):
        return StreamClassifier()

    def test_unknown_type_fallback(self, classifier):
        result = classifier.classify("UNKNOWN_TYPE", MockFrame(temporal_scope="ongoing"))
        assert result.primary_stream == "identity"
        assert result.confidence == 0.50

    def test_empty_type_fallback(self, classifier):
        result = classifier.classify("", MockFrame(temporal_scope="ongoing"))
        assert result.primary_stream == "identity"


class TestConfidenceValues:
    """Test confidence values match config."""

    @pytest.fixture
    def classifier(self):
        return StreamClassifier()

    def test_habitual_feeling_high_confidence(self, classifier):
        result = classifier.classify("FEELING_STATE", MockFrame(temporal_scope="habitual"))
        assert result.confidence >= 0.75

    def test_state_feeling_lower_confidence(self, classifier):
        result = classifier.classify("FEELING_STATE", MockFrame(temporal_scope="state"))
        assert result.confidence <= 0.70
