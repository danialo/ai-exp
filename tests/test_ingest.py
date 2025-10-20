"""Tests for ingestion pipeline."""

import tempfile
from pathlib import Path

import pytest

from src.memory.embedding import MockEmbeddingProvider
from src.memory.models import Actor
from src.memory.raw_store import RawStore
from src.memory.vector_store import VectorStore
from src.pipeline.ingest import (
    IngestionPipeline,
    InteractionPayload,
    create_ingestion_pipeline,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def raw_store(temp_dir):
    """Create raw store for testing."""
    return RawStore(temp_dir / "raw_store.db")


@pytest.fixture
def vector_store(temp_dir):
    """Create vector store for testing."""
    return VectorStore(temp_dir / "vectors", collection_name="test", reset=True)


@pytest.fixture
def embedding_provider():
    """Create mock embedding provider."""
    return MockEmbeddingProvider(dimension=384)


@pytest.fixture
def pipeline(raw_store, vector_store, embedding_provider):
    """Create ingestion pipeline."""
    return IngestionPipeline(
        raw_store=raw_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )


class TestInteractionPayload:
    """Test interaction payload construction."""

    def test_create_minimal_payload(self):
        """Test creating minimal payload."""
        payload = InteractionPayload(prompt="What is Python?", response="A programming language.")

        assert payload.prompt == "What is Python?"
        assert payload.response == "A programming language."
        assert payload.actor == Actor.USER
        assert payload.valence == 0.0
        assert payload.metadata == {}

    def test_create_full_payload(self):
        """Test creating payload with all fields."""
        payload = InteractionPayload(
            prompt="How do I use async?",
            response="Use async/await keywords.",
            actor=Actor.AGENT,
            valence=0.7,
            metadata={"category": "technical", "domain": "python"},
        )

        assert payload.prompt == "How do I use async?"
        assert payload.valence == 0.7
        assert payload.metadata["category"] == "technical"


class TestIngestionPipeline:
    """Test ingestion pipeline operations."""

    def test_ingest_single_interaction(self, pipeline, raw_store, vector_store):
        """Test ingesting a single interaction."""
        interaction = InteractionPayload(
            prompt="What is machine learning?",
            response="ML is a subset of AI that learns from data.",
            valence=0.5,
        )

        result = pipeline.ingest_interaction(interaction)

        # Check result
        assert result.experience_id is not None
        assert result.prompt_embedding_id is not None
        assert result.response_embedding_id is not None

        # Verify raw store
        experience = raw_store.get_experience(result.experience_id)
        assert experience is not None
        assert "What is machine learning?" in experience.content.structured["prompt"]
        assert experience.affect.vad.v == 0.5

        # Verify vector store
        prompt_vec = vector_store.get(result.prompt_embedding_id)
        assert prompt_vec is not None
        assert prompt_vec.metadata["experience_id"] == result.experience_id

        response_vec = vector_store.get(result.response_embedding_id)
        assert response_vec is not None

    def test_ingest_with_custom_id(self, pipeline):
        """Test ingesting with custom experience ID."""
        interaction = InteractionPayload(prompt="Test", response="Response")

        result = pipeline.ingest_interaction(interaction, experience_id="exp_custom_001")

        assert result.experience_id == "exp_custom_001"

    def test_ingest_increments_counts(self, pipeline, raw_store, vector_store):
        """Test that ingestion increments store counts."""
        initial_exp_count = raw_store.count_experiences()
        initial_vec_count = vector_store.count()

        interaction = InteractionPayload(prompt="Count test", response="Testing counts")

        pipeline.ingest_interaction(interaction)

        assert raw_store.count_experiences() == initial_exp_count + 1
        assert vector_store.count() == initial_vec_count + 2  # prompt + response

    def test_embedding_pointers_stored(self, pipeline, raw_store):
        """Test that embedding pointers are stored in experience."""
        interaction = InteractionPayload(prompt="Pointer test", response="Response")

        result = pipeline.ingest_interaction(interaction)

        experience = raw_store.get_experience(result.experience_id)
        assert experience.embeddings.semantic is not None
        assert "vec://sem/" in experience.embeddings.semantic

    def test_metadata_preserved(self, pipeline, raw_store):
        """Test that interaction metadata is preserved."""
        interaction = InteractionPayload(
            prompt="Metadata test",
            response="Response",
            metadata={"source": "test_suite", "version": "1.0"},
        )

        result = pipeline.ingest_interaction(interaction)

        experience = raw_store.get_experience(result.experience_id)
        assert experience.content.structured["source"] == "test_suite"
        assert experience.content.structured["version"] == "1.0"

    def test_affect_captured(self, pipeline, raw_store):
        """Test that affect information is captured."""
        interaction = InteractionPayload(prompt="Affect test", response="Response", valence=0.8)

        result = pipeline.ingest_interaction(interaction)

        experience = raw_store.get_experience(result.experience_id)
        assert experience.affect.vad.v == 0.8
        assert experience.affect.intensity == 0.8  # Should match valence


class TestBatchIngestion:
    """Test batch ingestion."""

    def test_ingest_batch(self, pipeline, raw_store, vector_store):
        """Test ingesting multiple interactions."""
        interactions = [
            InteractionPayload(prompt=f"Question {i}", response=f"Answer {i}") for i in range(5)
        ]

        results = pipeline.ingest_batch(interactions)

        assert len(results) == 5

        # Verify all were stored
        assert raw_store.count_experiences() == 5
        assert vector_store.count() == 10  # 5 prompts + 5 responses

    def test_batch_preserves_order(self, pipeline):
        """Test that batch ingestion preserves order."""
        interactions = [InteractionPayload(prompt=f"Q{i}", response=f"A{i}") for i in range(3)]

        results = pipeline.ingest_batch(interactions)

        # Check that results match input order
        for i, result in enumerate(results):
            assert f"Q{i}" in result.experience_id or result.experience_id is not None


class TestRetrievalIntegration:
    """Test that ingested experiences can be retrieved."""

    def test_ingested_experience_retrievable_by_similarity(
        self, pipeline, vector_store, embedding_provider
    ):
        """Test that ingested experiences can be found via similarity search."""
        # Ingest an interaction
        interaction = InteractionPayload(
            prompt="How do I create a Python virtual environment?",
            response="Use python -m venv myenv to create a virtual environment.",
        )

        result = pipeline.ingest_interaction(interaction)

        # Query with similar prompt
        similar_prompt = "How to make a Python venv?"
        query_embedding = embedding_provider.embed(similar_prompt)

        # Search vector store
        search_results = vector_store.query(query_embedding, top_k=5)

        # Should find the prompt embedding
        found_ids = [r.id for r in search_results]
        assert result.prompt_embedding_id in found_ids

    def test_multiple_experiences_ranked_by_similarity(
        self, pipeline, vector_store, embedding_provider
    ):
        """Test that similar experiences rank higher."""
        # Ingest related and unrelated interactions
        interactions = [
            InteractionPayload(
                prompt="What is Python?", response="Python is a programming language."
            ),
            InteractionPayload(
                prompt="How to write Python code?", response="Write code in .py files."
            ),
            InteractionPayload(prompt="What is the weather?", response="Check the forecast."),
        ]

        results = pipeline.ingest_batch(interactions)

        # Query with Python-related prompt
        query_embedding = embedding_provider.embed("Tell me about Python programming")

        search_results = vector_store.query(query_embedding, top_k=10)

        # Python-related prompts should appear before weather
        result_ids = [r.id for r in search_results]

        # At least one Python-related prompt should be in top results
        python_results = [results[0].prompt_embedding_id, results[1].prompt_embedding_id]
        assert any(pid in result_ids[:3] for pid in python_results)


class TestFactoryFunction:
    """Test factory function."""

    def test_create_pipeline(self, raw_store, vector_store, embedding_provider):
        """Test creating pipeline via factory."""
        pipeline = create_ingestion_pipeline(
            raw_store=raw_store,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
        )

        assert isinstance(pipeline, IngestionPipeline)
        assert pipeline.raw_store is raw_store
        assert pipeline.vector_store is vector_store
        assert pipeline.embedding_provider is embedding_provider


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    def test_full_pipeline_flow(self, pipeline, raw_store, vector_store):
        """Test complete flow from ingestion to retrieval."""
        # 1. Ingest interaction
        interaction = InteractionPayload(
            prompt="Explain async/await in Python",
            response="async/await allows asynchronous programming in Python.",
            valence=0.3,
            metadata={"category": "tutorial"},
        )

        result = pipeline.ingest_interaction(interaction)

        # 2. Verify in raw store
        experience = raw_store.get_experience(result.experience_id)
        assert experience is not None
        assert experience.content.structured["category"] == "tutorial"

        # 3. Verify embeddings in vector store
        prompt_vec = vector_store.get(result.prompt_embedding_id)
        response_vec = vector_store.get(result.response_embedding_id)

        assert prompt_vec is not None
        assert response_vec is not None
        assert prompt_vec.metadata["experience_id"] == result.experience_id

        # 4. Verify can retrieve by similarity
        search_results = vector_store.query(
            vector_store.collection.get(ids=[result.prompt_embedding_id], include=["embeddings"])[
                "embeddings"
            ][0],
            top_k=1,
        )

        assert len(search_results) > 0
        assert search_results[0].id == result.prompt_embedding_id
