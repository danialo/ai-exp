"""Tests for vector store."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.memory.embedding import MockEmbeddingProvider
from src.memory.vector_store import VectorStore, VectorStoreResult, create_vector_store


@pytest.fixture
def temp_dir():
    """Create temporary directory for vector store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def vector_store(temp_dir):
    """Create vector store for testing."""
    return VectorStore(
        persist_directory=temp_dir / "vectors",
        collection_name="test_collection",
        reset=True,
    )


@pytest.fixture
def embedding_provider():
    """Create mock embedding provider."""
    return MockEmbeddingProvider(dimension=384)


class TestVectorStoreBasics:
    """Test basic vector store operations."""

    def test_create_store(self, temp_dir):
        """Test creating vector store."""
        store = VectorStore(temp_dir / "test", collection_name="test")
        assert store.persist_directory.exists()
        assert store.collection_name == "test"

    def test_create_with_factory(self, temp_dir):
        """Test factory function."""
        store = create_vector_store(persist_directory=temp_dir / "factory", reset=True)
        assert isinstance(store, VectorStore)

    def test_reset_collection(self, temp_dir):
        """Test resetting collection."""
        # Create and populate
        store1 = VectorStore(temp_dir / "reset_test", collection_name="test", reset=False)
        vector = np.random.randn(384).astype(np.float32)
        store1.upsert("id1", vector)
        assert store1.count() == 1

        # Reset should clear
        store2 = VectorStore(temp_dir / "reset_test", collection_name="test", reset=True)
        assert store2.count() == 0


class TestUpsertAndGet:
    """Test upserting and retrieving vectors."""

    def test_upsert_single_vector(self, vector_store):
        """Test upserting a single vector."""
        vector = np.random.randn(384).astype(np.float32)
        vector_store.upsert("exp_001", vector, metadata={"type": "prompt"})

        result = vector_store.get("exp_001")
        assert result is not None
        assert result.id == "exp_001"
        assert result.metadata["type"] == "prompt"

    def test_upsert_updates_existing(self, vector_store):
        """Test that upsert updates existing vectors."""
        vec1 = np.random.randn(384).astype(np.float32)
        vec2 = np.random.randn(384).astype(np.float32)

        vector_store.upsert("exp_001", vec1, metadata={"version": "1"})
        vector_store.upsert("exp_001", vec2, metadata={"version": "2"})

        # Should still be only 1 vector
        assert vector_store.count() == 1

        result = vector_store.get("exp_001")
        assert result.metadata["version"] == "2"

    def test_upsert_batch(self, vector_store):
        """Test batch upsert."""
        ids = ["exp_001", "exp_002", "exp_003"]
        vectors = [np.random.randn(384).astype(np.float32) for _ in range(3)]
        metadatas = [{"type": "prompt"}, {"type": "response"}, {"type": "prompt"}]

        vector_store.upsert_batch(ids, vectors, metadatas)

        assert vector_store.count() == 3

        result = vector_store.get("exp_002")
        assert result is not None
        assert result.metadata["type"] == "response"

    def test_get_nonexistent(self, vector_store):
        """Test getting nonexistent vector."""
        result = vector_store.get("exp_does_not_exist")
        assert result is None


class TestQuery:
    """Test similarity search queries."""

    def test_query_returns_nearest(self, vector_store, embedding_provider):
        """Test that query returns nearest neighbors."""
        # Create embeddings for similar and dissimilar texts
        texts = [
            "The cat sat on the mat.",
            "A cat was on a mat.",
            "Python is a programming language.",
            "Machine learning is awesome.",
        ]

        embeddings = embedding_provider.embed_batch(texts)

        # Store vectors
        for i, emb in enumerate(embeddings):
            vector_store.upsert(f"exp_{i}", emb, metadata={"text": texts[i]})

        # Query with first text
        query_embedding = embeddings[0]
        results = vector_store.query(query_embedding, top_k=4)

        # First result should be the query itself
        assert results[0].id == "exp_0"
        assert results[0].score > 0.99  # Nearly identical

        # Should return 4 results
        assert len(results) == 4
        # Results should be ordered by descending score
        assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))

    def test_query_top_k_limit(self, vector_store):
        """Test that top_k limits results."""
        # Add 10 vectors
        for i in range(10):
            vec = np.random.randn(384).astype(np.float32)
            vector_store.upsert(f"exp_{i}", vec)

        query_vec = np.random.randn(384).astype(np.float32)
        results = vector_store.query(query_vec, top_k=3)

        assert len(results) == 3

    def test_query_with_metadata_filter(self, vector_store):
        """Test querying with metadata filter."""
        # Add vectors with different types
        for i in range(5):
            vec = np.random.randn(384).astype(np.float32)
            vector_store.upsert(
                f"exp_{i}", vec, metadata={"type": "prompt" if i % 2 == 0 else "response"}
            )

        query_vec = np.random.randn(384).astype(np.float32)
        results = vector_store.query(query_vec, top_k=10, where={"type": "prompt"})

        # Should only return prompt types
        assert all(r.metadata.get("type") == "prompt" for r in results)
        assert len(results) == 3  # 3 prompts (0, 2, 4)

    def test_query_empty_store(self, vector_store):
        """Test querying empty store."""
        query_vec = np.random.randn(384).astype(np.float32)
        results = vector_store.query(query_vec, top_k=5)
        assert results == []


class TestSelfNearestNeighbor:
    """Test that vectors return themselves as nearest neighbor."""

    def test_self_nearest_neighbor(self, vector_store, embedding_provider):
        """Test that a vector is its own nearest neighbor."""
        text = "This is a test sentence."
        embedding = embedding_provider.embed(text)

        vector_store.upsert("exp_test", embedding, metadata={"text": text})

        # Query with same embedding
        results = vector_store.query(embedding, top_k=1)

        assert len(results) == 1
        assert results[0].id == "exp_test"
        assert results[0].score > 0.99  # Nearly perfect match


class TestDelete:
    """Test vector deletion."""

    def test_delete_vector(self, vector_store):
        """Test deleting a vector."""
        vec = np.random.randn(384).astype(np.float32)
        vector_store.upsert("exp_001", vec)

        assert vector_store.count() == 1

        vector_store.delete("exp_001")

        assert vector_store.count() == 0
        assert vector_store.get("exp_001") is None

    def test_delete_nonexistent(self, vector_store):
        """Test deleting nonexistent vector (should not error)."""
        vector_store.delete("exp_does_not_exist")
        # Should not raise error


class TestCount:
    """Test counting vectors."""

    def test_count_empty(self, vector_store):
        """Test counting empty store."""
        assert vector_store.count() == 0

    def test_count_after_inserts(self, vector_store):
        """Test count after inserting vectors."""
        for i in range(7):
            vec = np.random.randn(384).astype(np.float32)
            vector_store.upsert(f"exp_{i}", vec)

        assert vector_store.count() == 7


class TestReset:
    """Test resetting store."""

    def test_reset_clears_all(self, vector_store):
        """Test that reset clears all vectors."""
        for i in range(5):
            vec = np.random.randn(384).astype(np.float32)
            vector_store.upsert(f"exp_{i}", vec)

        assert vector_store.count() == 5

        vector_store.reset()

        assert vector_store.count() == 0


class TestVectorStoreResult:
    """Test VectorStoreResult class."""

    def test_create_result(self):
        """Test creating result object."""
        result = VectorStoreResult(id="exp_001", score=0.95, metadata={"type": "prompt"})

        assert result.id == "exp_001"
        assert result.score == 0.95
        assert result.metadata["type"] == "prompt"

    def test_result_default_metadata(self):
        """Test result with no metadata."""
        result = VectorStoreResult(id="exp_002", score=0.85)
        assert result.metadata == {}
