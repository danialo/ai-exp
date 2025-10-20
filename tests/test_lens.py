"""Tests for experience lens pass."""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from src.pipeline.lens import ExperienceLens, create_experience_lens
from src.services.retrieval import RetrievalResult


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    service = Mock()
    service.generate_response.return_value = "This is a draft response about Python imports."
    return service


@pytest.fixture
def mock_retrieval_service():
    """Mock retrieval service."""
    service = Mock()
    return service


@pytest.fixture
def sample_memories_positive():
    """Sample memories with positive valence."""
    return [
        RetrievalResult(
            experience_id="exp_001",
            prompt_text="How do Python imports work?",
            response_text="Python imports load modules...",
            valence=0.5,
            similarity_score=0.9,
            recency_score=0.8,
            combined_score=0.88,
            created_at=datetime.now(timezone.utc),
        ),
        RetrievalResult(
            experience_id="exp_002",
            prompt_text="Tell me about modules",
            response_text="Modules are Python files...",
            valence=0.3,
            similarity_score=0.7,
            recency_score=0.6,
            combined_score=0.68,
            created_at=datetime.now(timezone.utc),
        ),
    ]


@pytest.fixture
def sample_memories_negative():
    """Sample memories with negative valence."""
    return [
        RetrievalResult(
            experience_id="exp_003",
            prompt_text="I'm struggling with imports",
            response_text="Import errors can be frustrating...",
            valence=-0.7,
            similarity_score=0.9,
            recency_score=0.8,
            combined_score=0.88,
            created_at=datetime.now(timezone.utc),
        ),
        RetrievalResult(
            experience_id="exp_004",
            prompt_text="This isn't working",
            response_text="Let's troubleshoot...",
            valence=-0.5,
            similarity_score=0.8,
            recency_score=0.7,
            combined_score=0.78,
            created_at=datetime.now(timezone.utc),
        ),
    ]


def test_create_experience_lens(mock_llm_service, mock_retrieval_service):
    """Test factory function creates lens instance."""
    lens = create_experience_lens(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
    )
    assert isinstance(lens, ExperienceLens)
    assert lens.llm_service == mock_llm_service
    assert lens.retrieval_service == mock_retrieval_service
    assert lens.top_k_memories == 3
    assert lens.valence_threshold == -0.2


def test_process_without_memories(mock_llm_service, mock_retrieval_service):
    """Test lens pass without retrieved memories."""
    mock_retrieval_service.retrieve_similar.return_value = []

    lens = ExperienceLens(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
    )

    result = lens.process(prompt="Test prompt", retrieve_memories=True)

    # Should call retrieval
    mock_retrieval_service.retrieve_similar.assert_called_once()

    # Should generate draft
    mock_llm_service.generate_response.assert_called_once()

    # No memories, so blended valence is 0
    assert result.blended_valence == 0.0

    # Draft and augmented should be same (no tone adjustment for neutral valence)
    assert result.draft_response == result.augmented_response

    # No citations
    assert result.citations == []
    assert result.retrieved_experience_ids == []


def test_process_with_positive_memories(
    mock_llm_service, mock_retrieval_service, sample_memories_positive
):
    """Test lens pass with positive valence memories."""
    mock_retrieval_service.retrieve_similar.return_value = sample_memories_positive

    lens = ExperienceLens(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
    )

    result = lens.process(prompt="Test prompt")

    # Should retrieve memories
    assert len(result.retrieved_experience_ids) == 2
    assert "exp_001" in result.retrieved_experience_ids

    # Blended valence should be positive (weighted average of 0.5 and 0.3)
    assert result.blended_valence > 0

    # No tone adjustment for positive valence
    assert not result.augmented_response.startswith("I understand")
    assert not result.augmented_response.startswith("I notice")

    # Should have citations
    assert len(result.citations) == 2
    assert "[exp_001]" in result.citations
    assert "_References:" in result.augmented_response


def test_process_with_negative_memories(
    mock_llm_service, mock_retrieval_service, sample_memories_negative
):
    """Test lens pass with negative valence memories."""
    mock_retrieval_service.retrieve_similar.return_value = sample_memories_negative

    lens = ExperienceLens(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        valence_threshold=-0.2,
    )

    result = lens.process(prompt="Test prompt")

    # Blended valence should be negative
    assert result.blended_valence < -0.2

    # Should have empathetic preface
    assert (
        result.augmented_response.startswith("I understand")
        or result.augmented_response.startswith("I notice")
        or result.augmented_response.startswith("Based on")
    )

    # Draft should NOT have preface
    assert not result.draft_response.startswith("I understand")

    # Should have citations
    assert len(result.citations) == 2
    assert "_References:" in result.augmented_response


def test_process_skip_retrieval(mock_llm_service, mock_retrieval_service):
    """Test lens pass with retrieval disabled."""
    lens = ExperienceLens(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
    )

    result = lens.process(prompt="Test prompt", retrieve_memories=False)

    # Should NOT call retrieval
    mock_retrieval_service.retrieve_similar.assert_not_called()

    # Should still generate draft
    mock_llm_service.generate_response.assert_called_once()

    # No memories
    assert result.blended_valence == 0.0
    assert result.citations == []


def test_calculate_blended_valence_empty():
    """Test valence calculation with empty memories."""
    lens = ExperienceLens(
        llm_service=Mock(),
        retrieval_service=Mock(),
    )

    valence = lens._calculate_blended_valence([])
    assert valence == 0.0


def test_calculate_blended_valence_weighted(sample_memories_positive):
    """Test valence calculation uses weighted average."""
    lens = ExperienceLens(
        llm_service=Mock(),
        retrieval_service=Mock(),
    )

    # Memory 1: valence=0.5, score=0.88
    # Memory 2: valence=0.3, score=0.68
    # Weighted avg: (0.5*0.88 + 0.3*0.68) / (0.88 + 0.68) = 0.413
    valence = lens._calculate_blended_valence(sample_memories_positive)
    assert 0.40 < valence < 0.43


def test_tone_adjustment_strong_negative():
    """Test empathetic preface for strongly negative valence."""
    lens = ExperienceLens(
        llm_service=Mock(),
        retrieval_service=Mock(),
    )

    adjusted = lens._apply_tone_adjustment("Draft response", valence=-0.8)
    assert "I understand this topic has been challenging" in adjusted
    assert "Draft response" in adjusted


def test_tone_adjustment_moderate_negative():
    """Test empathetic preface for moderately negative valence."""
    lens = ExperienceLens(
        llm_service=Mock(),
        retrieval_service=Mock(),
    )

    adjusted = lens._apply_tone_adjustment("Draft response", valence=-0.4)
    assert "I notice we've discussed related challenges" in adjusted
    assert "Draft response" in adjusted


def test_tone_adjustment_mild_negative():
    """Test empathetic preface for mildly negative valence."""
    lens = ExperienceLens(
        llm_service=Mock(),
        retrieval_service=Mock(),
        valence_threshold=-0.2,
    )

    adjusted = lens._apply_tone_adjustment("Draft response", valence=-0.25)
    assert "Based on our previous conversations" in adjusted
    assert "Draft response" in adjusted


def test_tone_adjustment_neutral():
    """Test no adjustment for neutral valence."""
    lens = ExperienceLens(
        llm_service=Mock(),
        retrieval_service=Mock(),
    )

    draft = "Draft response"
    adjusted = lens._apply_tone_adjustment(draft, valence=0.0)
    assert adjusted == draft


def test_tone_adjustment_positive():
    """Test no adjustment for positive valence."""
    lens = ExperienceLens(
        llm_service=Mock(),
        retrieval_service=Mock(),
    )

    draft = "Draft response"
    adjusted = lens._apply_tone_adjustment(draft, valence=0.5)
    assert adjusted == draft


def test_append_citations():
    """Test citation appending."""
    lens = ExperienceLens(
        llm_service=Mock(),
        retrieval_service=Mock(),
    )

    response = "This is a response"
    citations = ["[exp_001]", "[exp_002]"]

    result = lens._append_citations(response, citations)

    assert "This is a response" in result
    assert "_References: [exp_001] [exp_002]_" in result


def test_append_citations_empty():
    """Test citation appending with empty list."""
    lens = ExperienceLens(
        llm_service=Mock(),
        retrieval_service=Mock(),
    )

    response = "This is a response"
    result = lens._append_citations(response, [])

    assert result == response
    assert "_References:" not in result


def test_custom_system_prompt(mock_llm_service, mock_retrieval_service):
    """Test lens pass with custom system prompt."""
    mock_retrieval_service.retrieve_similar.return_value = []

    lens = ExperienceLens(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
    )

    custom_prompt = "You are a helpful assistant."
    result = lens.process(
        prompt="Test prompt",
        system_prompt=custom_prompt,
        retrieve_memories=False,
    )

    # Should pass custom system prompt to LLM
    mock_llm_service.generate_response.assert_called_once()
    call_args = mock_llm_service.generate_response.call_args
    assert call_args[1]["system_prompt"] == custom_prompt
