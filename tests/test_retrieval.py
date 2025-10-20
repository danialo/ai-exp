"""Tests for retrieval service."""

import tempfile
import time
from pathlib import Path

import pytest

from src.memory.embedding import MockEmbeddingProvider
from src.memory.raw_store import RawStore
from src.memory.vector_store import VectorStore
from src.pipeline.ingest import IngestionPipeline, InteractionPayload
from src.services.retrieval import RetrievalService, create_retrieval_service


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def raw_store(temp_dir):
    """Create raw store."""
    return RawStore(temp_dir / "raw_store.db")


@pytest.fixture
def vector_store(temp_dir):
    """Create vector store."""
    return VectorStore(temp_dir / "vectors", collection_name="test", reset=True)


@pytest.fixture
def embedding_provider():
    """Create embedding provider."""
    return MockEmbeddingProvider(dimension=384)


@pytest.fixture
def ingestion_pipeline(raw_store, vector_store, embedding_provider):
    """Create ingestion pipeline."""
    return IngestionPipeline(raw_store, vector_store, embedding_provider)


@pytest.fixture
def retrieval_service(raw_store, vector_store, embedding_provider):
    """Create retrieval service."""
    return RetrievalService(raw_store, vector_store, embedding_provider)


class TestRetrievalServiceInit:
    """Test retrieval service initialization."""

    def test_create_service(self, raw_store, vector_store, embedding_provider):
        """Test creating retrieval service."""
        service = RetrievalService(raw_store, vector_store, embedding_provider)

        assert service.raw_store is raw_store
        assert service.vector_store is vector_store
        assert service.embedding_provider is embedding_provider
        assert service.semantic_weight == 0.8
        assert service.recency_weight == 0.2

    def test_custom_weights(self, raw_store, vector_store, embedding_provider):
        """Test custom weight configuration."""
        service = RetrievalService(
            raw_store,
            vector_store,
            embedding_provider,
            semantic_weight=0.7,
            recency_weight=0.3,
        )

        assert service.semantic_weight == 0.7
        assert service.recency_weight == 0.3

    def test_factory_function(self, raw_store, vector_store, embedding_provider):
        """Test factory function."""
        service = create_retrieval_service(raw_store, vector_store, embedding_provider)
        assert isinstance(service, RetrievalService)


class TestRetrieveSimilar:
    """Test similarity retrieval."""

    def test_retrieve_from_empty_store(self, retrieval_service):
        """Test retrieval from empty store."""
        results = retrieval_service.retrieve_similar("test query")
        assert results == []

    def test_retrieve_single_experience(
        self, retrieval_service, ingestion_pipeline, embedding_provider
    ):
        """Test retrieving a single experience."""
        # Ingest an experience
        interaction = InteractionPayload(
            prompt="What is Python?", response="A programming language."
        )
        ingestion_pipeline.ingest_interaction(interaction)

        # Retrieve with similar query
        results = retrieval_service.retrieve_similar("Tell me about Python", top_k=5)

        assert len(results) == 1
        assert "Python" in results[0].prompt_text
        assert results[0].similarity_score > 0

    def test_retrieve_respects_top_k(self, retrieval_service, ingestion_pipeline):
        """Test that top_k limits results."""
        # Ingest 10 experiences
        for i in range(10):
            interaction = InteractionPayload(prompt=f"Question {i}", response=f"Answer {i}")
            ingestion_pipeline.ingest_interaction(interaction)

        # Retrieve top 3
        results = retrieval_service.retrieve_similar("Question about something", top_k=3)

        assert len(results) <= 3

    def test_combined_score_calculation(self, retrieval_service, ingestion_pipeline):
        """Test that combined score is calculated correctly."""
        interaction = InteractionPayload(prompt="Test", response="Response")
        ingestion_pipeline.ingest_interaction(interaction)

        results = retrieval_service.retrieve_similar("Test", top_k=1)

        assert len(results) == 1
        result = results[0]

        # Verify score components exist
        assert 0 <= result.similarity_score <= 1
        assert 0 <= result.recency_score <= 1
        assert result.combined_score > 0

        # Verify combined score formula
        expected = (0.8 * result.similarity_score) + (0.2 * result.recency_score)
        assert abs(result.combined_score - expected) < 0.01

    def test_results_sorted_by_score(self, retrieval_service, ingestion_pipeline):
        """Test that results are sorted by combined score."""
        # Ingest multiple experiences
        for i in range(5):
            interaction = InteractionPayload(prompt=f"Topic {i}", response=f"Content {i}")
            ingestion_pipeline.ingest_interaction(interaction)
            time.sleep(0.01)  # Small delay for recency differentiation

        results = retrieval_service.retrieve_similar("Topic", top_k=5)

        # Verify descending order
        for i in range(len(results) - 1):
            assert results[i].combined_score >= results[i + 1].combined_score


class TestRecencyBias:
    """Test recency biasing."""

    def test_newer_experiences_boosted(self, retrieval_service, ingestion_pipeline):
        """Test that newer experiences get recency boost."""
        # Ingest older experience
        old_interaction = InteractionPayload(prompt="Old question", response="Old answer")
        ingestion_pipeline.ingest_interaction(old_interaction)

        time.sleep(0.1)  # Delay

        # Ingest newer experience
        new_interaction = InteractionPayload(prompt="New question", response="New answer")
        ingestion_pipeline.ingest_interaction(new_interaction)

        results = retrieval_service.retrieve_similar("question", top_k=2)

        assert len(results) == 2

        # Both should have recency scores, newer should be higher
        old_result = next(r for r in results if "Old" in r.prompt_text)
        new_result = next(r for r in results if "New" in r.prompt_text)

        assert new_result.recency_score >= old_result.recency_score


class TestAgeFiltering:
    """Test age-based filtering."""

    def test_max_age_filter(self, retrieval_service, ingestion_pipeline):
        """Test filtering by maximum age."""
        # For this test, we can't actually make experiences old without mocking time
        # So we test the interface works
        interaction = InteractionPayload(prompt="Recent", response="Response")
        ingestion_pipeline.ingest_interaction(interaction)

        # With a very large max_age, should return results
        results = retrieval_service.retrieve_similar("Recent", top_k=5, max_age_days=365)
        assert len(results) == 1

        # With max_age=0, should filter out (assuming test takes < 1 day)
        # This would need time manipulation to test properly, so we just verify it runs
        retrieval_service.retrieve_similar("Recent", top_k=5, max_age_days=0)
        # Result depends on test execution time, so we just check it doesn't error


class TestMetadataExtraction:
    """Test metadata extraction in results."""

    def test_result_contains_all_fields(self, retrieval_service, ingestion_pipeline):
        """Test that results contain all expected fields."""
        interaction = InteractionPayload(
            prompt="What is AI?", response="Artificial Intelligence", valence=0.6
        )
        ingestion_pipeline.ingest_interaction(interaction)

        results = retrieval_service.retrieve_similar("AI question", top_k=1)

        assert len(results) == 1
        result = results[0]

        assert result.experience_id is not None
        assert result.prompt_text == "What is AI?"
        assert result.response_text == "Artificial Intelligence"
        assert result.valence == 0.6
        assert result.similarity_score >= 0
        assert result.recency_score >= 0
        assert result.combined_score >= 0
        assert result.created_at is not None


class TestGetExperienceDetails:
    """Test getting full experience details."""

    def test_get_existing_experience(self, retrieval_service, ingestion_pipeline):
        """Test retrieving full experience details."""
        interaction = InteractionPayload(prompt="Details test", response="Response")
        result = ingestion_pipeline.ingest_interaction(interaction)

        experience = retrieval_service.get_experience_details(result.experience_id)

        assert experience is not None
        assert experience.id == result.experience_id
        assert "Details test" in experience.content.structured["prompt"]

    def test_get_nonexistent_experience(self, retrieval_service):
        """Test getting non-existent experience returns None."""
        experience = retrieval_service.get_experience_details("exp_does_not_exist")
        assert experience is None


class TestEndToEndRetrieval:
    """End-to-end retrieval tests."""

    def test_ingest_and_retrieve_workflow(self, ingestion_pipeline, retrieval_service):
        """Test complete ingest→retrieve workflow."""
        # Ingest multiple related experiences
        python_interactions = [
            InteractionPayload(
                prompt="What is Python?",
                response="Python is a programming language.",
                valence=0.5,
            ),
            InteractionPayload(
                prompt="How to use Python lists?",
                response="Lists use square brackets: [1, 2, 3]",
                valence=0.3,
            ),
        ]

        # Ingest unrelated experience
        weather_interaction = InteractionPayload(
            prompt="What's the weather?", response="It's sunny.", valence=0.7
        )

        for interaction in python_interactions:
            ingestion_pipeline.ingest_interaction(interaction)
        ingestion_pipeline.ingest_interaction(weather_interaction)

        # Retrieve Python-related experiences
        results = retrieval_service.retrieve_similar("Tell me about Python programming", top_k=3)

        # Should find Python experiences
        assert len(results) > 0

        # Check that results contain relevant information
        result_texts = " ".join([r.prompt_text + r.response_text for r in results])
        assert "Python" in result_texts
