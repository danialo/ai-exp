"""Unit tests for SourceContextClassifier."""

import pytest
from src.services.source_context_classifier import SourceContextClassifier
from datetime import datetime, timezone
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class SourceContextExperience:
    """Test experience object for SourceContextClassifier.

    The classifier expects:
    - id: experience ID
    - content: dict with 'text' and 'structured' keys, OR a string
    - affect: dict with optional 'vad' sub-dict
    - session_id: for context_id computation
    """
    id: str
    content: Dict[str, Any]
    affect: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None


class TestModeWeights:
    """Test mode-based weight assignment."""

    @pytest.fixture
    def classifier(self):
        return SourceContextClassifier()

    def _make_experience(
        self,
        mode: Optional[str] = None,
        text: str = "I am patient",
        is_journaling: bool = False,
        is_introspection: bool = False,
        is_roleplay: bool = False,
        arousal: Optional[float] = None,
    ) -> SourceContextExperience:
        structured = {}
        if mode:
            structured['interaction_mode'] = mode
        if is_journaling:
            structured['is_journaling'] = True
        if is_introspection:
            structured['is_introspection'] = True
        if is_roleplay:
            structured['is_roleplay'] = True

        affect = {}
        if arousal is not None:
            affect['vad'] = {'a': arousal}

        return SourceContextExperience(
            id=str(uuid4()),
            content={'text': text, 'structured': structured},
            affect=affect,
            session_id="conv_123",
        )

    def test_journaling_mode_high_weight(self, classifier):
        exp = self._make_experience(is_journaling=True)
        result = classifier.classify(exp)
        # Journaling has highest weight
        assert result.source_weight >= 0.9

    def test_introspection_mode_high_weight(self, classifier):
        exp = self._make_experience(is_introspection=True)
        result = classifier.classify(exp)
        assert result.source_weight >= 0.85

    def test_roleplay_mode_low_weight(self, classifier):
        exp = self._make_experience(is_roleplay=True)
        result = classifier.classify(exp)
        # Roleplay should have lower weight
        assert result.source_weight <= 0.5

    def test_heated_mode_from_arousal(self, classifier):
        exp = self._make_experience(arousal=0.85)  # High arousal
        result = classifier.classify(exp)
        assert result.mode == 'heated'

    def test_unknown_mode_has_fallback(self, classifier):
        exp = self._make_experience()  # No mode indicators
        result = classifier.classify(exp)
        assert result.source_weight > 0


class TestHeuristicPenalties:
    """Test heuristic-based penalties."""

    @pytest.fixture
    def classifier(self):
        return SourceContextClassifier()

    def _make_experience(self, text: str) -> SourceContextExperience:
        return SourceContextExperience(
            id=str(uuid4()),
            content={'text': text, 'structured': {}},
            affect={},
            session_id="conv_123",
        )

    def test_caps_penalty_applied(self, classifier):
        normal_exp = self._make_experience("I am patient")
        caps_exp = self._make_experience("I AM VERY ANGRY ABOUT THIS")

        normal_result = classifier.classify(normal_exp)
        caps_result = classifier.classify(caps_exp)

        # Caps should get penalized
        assert caps_result.source_weight <= normal_result.source_weight

    def test_exclaim_penalty_applied(self, classifier):
        normal_exp = self._make_experience("I am patient")
        exclaim_exp = self._make_experience("Wow! Amazing! Incredible! Unbelievable!")

        normal_result = classifier.classify(normal_exp)
        exclaim_result = classifier.classify(exclaim_exp)

        # Exclamation-heavy text should get penalized
        assert exclaim_result.source_weight <= normal_result.source_weight

    def test_weight_clamped_to_zero_one(self, classifier):
        # Extreme case: high arousal + caps + exclamations
        exp = SourceContextExperience(
            id=str(uuid4()),
            content={
                'text': "I HATE THIS!!! SO ANGRY!!! AAAARGH!!!",
                'structured': {}
            },
            affect={'vad': {'a': 0.95}},
            session_id="conv_123",
        )
        result = classifier.classify(exp)
        assert 0.0 <= result.source_weight <= 1.0


class TestContextId:
    """Test context_id generation."""

    @pytest.fixture
    def classifier(self):
        return SourceContextClassifier()

    def test_context_id_uses_session_and_mode(self, classifier):
        exp = SourceContextExperience(
            id="exp_xyz",
            content={
                'text': "I am patient",
                'structured': {'is_journaling': True}
            },
            affect={},
            session_id="conv_abc",
        )
        result = classifier.classify(exp)
        # Context ID should combine session and mode
        assert "conv_abc" in result.context_id
        assert "journaling" in result.context_id

    def test_fallback_to_experience_id(self, classifier):
        exp = SourceContextExperience(
            id="exp_xyz",
            content={'text': "I am patient", 'structured': {}},
            affect={},
            session_id=None,
        )
        result = classifier.classify(exp)
        assert result.context_id == "exp_xyz"


class TestSourceContextResult:
    """Test SourceContext result structure."""

    @pytest.fixture
    def classifier(self):
        return SourceContextClassifier()

    def test_result_has_required_fields(self, classifier):
        exp = SourceContextExperience(
            id="exp_xyz",
            content={
                'text': "I am patient",
                'structured': {'is_journaling': True}
            },
            affect={},
            session_id="conv_123",
        )
        result = classifier.classify(exp)

        assert hasattr(result, 'mode')
        assert hasattr(result, 'source_weight')
        assert hasattr(result, 'context_id')

    def test_details_contains_base_weight(self, classifier):
        exp = SourceContextExperience(
            id="exp_xyz",
            content={
                'text': "I am patient",
                'structured': {'is_journaling': True}
            },
            affect={},
            session_id="conv_123",
        )
        result = classifier.classify(exp)

        # Details should contain the base weight before penalties
        assert 'base_weight' in result.details
