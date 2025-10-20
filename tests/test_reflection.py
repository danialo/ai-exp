"""Tests for reflection writer."""

import pytest
from unittest.mock import Mock, call
from datetime import datetime, timezone

from src.pipeline.reflection import ReflectionWriter, create_reflection_writer
from src.memory.models import ExperienceType, Actor, CaptureMethod


@pytest.fixture
def mock_raw_store():
    """Mock raw store."""
    store = Mock()
    store.append_experience.return_value = "obs_2025-10-19T12:00:00Z_1234"
    return store


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = Mock()
    return store


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider."""
    provider = Mock()
    provider.embed.return_value = [0.1] * 384  # Mock embedding vector
    return provider


@pytest.fixture
def reflection_writer(mock_raw_store, mock_vector_store, mock_embedding_provider):
    """Create reflection writer instance."""
    return ReflectionWriter(
        raw_store=mock_raw_store,
        vector_store=mock_vector_store,
        embedding_provider=mock_embedding_provider,
    )


def test_create_reflection_writer(mock_raw_store, mock_vector_store, mock_embedding_provider):
    """Test factory function creates writer instance."""
    writer = create_reflection_writer(
        raw_store=mock_raw_store,
        vector_store=mock_vector_store,
        embedding_provider=mock_embedding_provider,
    )
    assert isinstance(writer, ReflectionWriter)
    assert writer.raw_store == mock_raw_store
    assert writer.vector_store == mock_vector_store
    assert writer.embedding_provider == mock_embedding_provider


def test_record_reflection_with_memories(reflection_writer):
    """Test recording reflection with retrieved memories."""
    reflection_id = reflection_writer.record_reflection(
        interaction_id="exp_001",
        prompt="What is Python?",
        response="Python is a high-level programming language.",
        retrieved_ids=["exp_002", "exp_003"],
        blended_valence=0.3,
    )

    # Should generate embedding
    reflection_writer.embedding_provider.embed.assert_called_once()

    # Should store in vector store with reflection label
    reflection_writer.vector_store.upsert.assert_called_once()
    upsert_call = reflection_writer.vector_store.upsert.call_args
    assert upsert_call[1]["metadata"]["label"] == "reflection"
    assert upsert_call[1]["metadata"]["parent_interaction"] == "exp_001"
    assert upsert_call[1]["metadata"]["retrieved_count"] == 2

    # Should append to raw store
    reflection_writer.raw_store.append_experience.assert_called_once()
    exp_arg = reflection_writer.raw_store.append_experience.call_args[0][0]

    # Check experience type and properties
    assert exp_arg.type == ExperienceType.OBSERVATION
    assert exp_arg.provenance.actor == Actor.AGENT
    assert exp_arg.provenance.method == CaptureMethod.MODEL_INFER
    assert "exp_001" in exp_arg.parents
    assert "exp_002" in exp_arg.parents
    assert "exp_003" in exp_arg.parents
    assert "reflection" in exp_arg.affect.labels

    # Check structured content
    assert exp_arg.content.structured["parent_interaction"] == "exp_001"
    assert exp_arg.content.structured["retrieved_experiences"] == ["exp_002", "exp_003"]
    assert exp_arg.content.structured["retrieved_count"] == 2
    assert exp_arg.content.structured["blended_valence"] == 0.3

    # Should return stored ID
    assert reflection_id == "obs_2025-10-19T12:00:00Z_1234"


def test_record_reflection_without_memories(reflection_writer):
    """Test recording reflection without retrieved memories."""
    reflection_id = reflection_writer.record_reflection(
        interaction_id="exp_001",
        prompt="What is Python?",
        response="Python is a programming language.",
        retrieved_ids=[],
        blended_valence=0.0,
    )

    # Should still create reflection
    reflection_writer.raw_store.append_experience.assert_called_once()
    exp_arg = reflection_writer.raw_store.append_experience.call_args[0][0]

    # Check no retrieved experiences in parents (only interaction)
    assert exp_arg.parents == ["exp_001"]
    assert exp_arg.content.structured["retrieved_count"] == 0

    # Reflection note should mention no prior experiences
    note = exp_arg.content.structured["reflection_note"]
    assert "without retrieving prior experiences" in note


def test_generate_reflection_note_no_memories(reflection_writer):
    """Test reflection note generation without memories."""
    note = reflection_writer._generate_reflection_note(
        prompt="What is Python?",
        response="Python is a programming language.",
        retrieved_ids=[],
        blended_valence=0.0,
    )

    assert "without retrieving prior experiences" in note
    assert "What is Python?" in note
    assert "novel topic" in note or "fresh start" in note


def test_generate_reflection_note_single_memory(reflection_writer):
    """Test reflection note generation with single memory."""
    note = reflection_writer._generate_reflection_note(
        prompt="Tell me about imports",
        response="Imports allow you to use code from other modules.",
        retrieved_ids=["exp_002"],
        blended_valence=0.5,
    )

    assert "1 relevant experience" in note
    assert "mildly positive" in note
    assert "exp_002" in note
    assert "Tell me about imports" in note


def test_generate_reflection_note_multiple_memories(reflection_writer):
    """Test reflection note generation with multiple memories."""
    note = reflection_writer._generate_reflection_note(
        prompt="How do I use decorators?",
        response="Decorators are a way to modify functions...",
        retrieved_ids=["exp_001", "exp_002", "exp_003", "exp_004"],
        blended_valence=-0.4,
    )

    assert "4 relevant experience" in note
    assert "mildly negative" in note
    assert "exp_001" in note
    assert "exp_002" in note
    assert "exp_003" in note
    # Should show "and 1 more" for the 4th experience
    assert "and 1 more" in note


def test_describe_affect_ranges(reflection_writer):
    """Test affect description for different valence ranges."""
    assert reflection_writer._describe_affect(-0.8) == "strongly negative"
    assert reflection_writer._describe_affect(-0.3) == "mildly negative"
    assert reflection_writer._describe_affect(0.0) == "neutral"
    assert reflection_writer._describe_affect(0.1) == "neutral"
    assert reflection_writer._describe_affect(0.3) == "mildly positive"
    assert reflection_writer._describe_affect(0.7) == "strongly positive"


def test_extract_response_focus_short(reflection_writer):
    """Test response focus extraction for short responses."""
    response = "Python is great."
    focus = reflection_writer._extract_response_focus(response)
    assert focus == "Python is great"


def test_extract_response_focus_long(reflection_writer):
    """Test response focus extraction for long responses."""
    response = (
        "This is a very long response that goes on and on with lots of detail "
        "about many different topics and concepts that are all important to understand. "
        "But we only want the first part."
    )
    focus = reflection_writer._extract_response_focus(response, max_length=60)
    assert len(focus) <= 65  # max_length + "..."
    assert focus.endswith("...")
    assert "This is a very long response" in focus


def test_extract_response_focus_multi_sentence(reflection_writer):
    """Test response focus extracts first sentence."""
    response = "Python is great. It's very easy to learn. Many people use it."
    focus = reflection_writer._extract_response_focus(response)
    assert focus == "Python is great"
    assert "easy to learn" not in focus


def test_truncate_short_text(reflection_writer):
    """Test truncation with text shorter than max length."""
    text = "Short text"
    truncated = reflection_writer._truncate(text, 50)
    assert truncated == "Short text"


def test_truncate_long_text(reflection_writer):
    """Test truncation with text longer than max length."""
    text = "This is a very long piece of text that needs to be truncated at word boundaries"
    truncated = reflection_writer._truncate(text, 30)
    assert len(truncated) <= 35  # max_length + "..."
    assert truncated.endswith("...")
    assert "This is a very long piece" in truncated


def test_reflection_affect_inherits_valence(reflection_writer):
    """Test that reflection affect inherits blended valence."""
    reflection_writer.record_reflection(
        interaction_id="exp_001",
        prompt="Test prompt",
        response="Test response",
        retrieved_ids=["exp_002"],
        blended_valence=-0.6,
    )

    exp_arg = reflection_writer.raw_store.append_experience.call_args[0][0]
    assert exp_arg.affect.vad.v == -0.6
    assert exp_arg.affect.intensity == 0.6  # abs(valence)
    assert "reflection" in exp_arg.affect.labels


def test_reflection_embedding_metadata(reflection_writer):
    """Test that reflection embedding has proper metadata."""
    reflection_writer.record_reflection(
        interaction_id="exp_123",
        prompt="Test prompt",
        response="Test response",
        retrieved_ids=["exp_456", "exp_789"],
        blended_valence=0.2,
    )

    # Check vector store upsert call
    upsert_call = reflection_writer.vector_store.upsert.call_args
    metadata = upsert_call[1]["metadata"]

    assert metadata["label"] == "reflection"
    assert metadata["parent_interaction"] == "exp_123"
    assert metadata["retrieved_count"] == 2
    assert "text_preview" in metadata


def test_reflection_id_format(reflection_writer):
    """Test that reflection IDs use 'obs_' prefix."""
    reflection_id = reflection_writer.record_reflection(
        interaction_id="exp_001",
        prompt="Test",
        response="Response",
        retrieved_ids=[],
        blended_valence=0.0,
    )

    # Returned ID comes from mock, but check the experience object
    exp_arg = reflection_writer.raw_store.append_experience.call_args[0][0]
    assert exp_arg.id.startswith("obs_")
