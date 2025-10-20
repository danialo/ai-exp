"""Tests for embedding provider."""

import numpy as np
import pytest

from src.memory.embedding import (
    MockEmbeddingProvider,
    SentenceTransformerEmbedding,
    create_embedding_provider,
)


class TestMockEmbeddingProvider:
    """Test mock embedding provider."""

    def test_deterministic_embeddings(self):
        """Test that same text produces same embedding."""
        provider = MockEmbeddingProvider(dimension=384)

        text = "This is a test"
        emb1 = provider.embed(text)
        emb2 = provider.embed(text)

        assert np.allclose(emb1, emb2)
        assert len(emb1) == 384

    def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        provider = MockEmbeddingProvider(dimension=384)

        emb1 = provider.embed("First text")
        emb2 = provider.embed("Second text")

        # Should be different (with high probability)
        assert not np.allclose(emb1, emb2)

    def test_unit_normalized(self):
        """Test that embeddings are unit normalized."""
        provider = MockEmbeddingProvider(dimension=384)

        embedding = provider.embed("Test text")
        norm = np.linalg.norm(embedding)

        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_batch_embeddings(self):
        """Test batch embedding generation."""
        provider = MockEmbeddingProvider(dimension=128)

        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 128 for emb in embeddings)

        # Batch results should match individual results
        for text, batch_emb in zip(texts, embeddings):
            single_emb = provider.embed(text)
            assert np.allclose(batch_emb, single_emb)

    def test_get_dimension(self):
        """Test getting embedding dimension."""
        provider = MockEmbeddingProvider(dimension=512)
        assert provider.get_dimension() == 512

    def test_get_model_name(self):
        """Test getting model name."""
        provider = MockEmbeddingProvider()
        assert provider.get_model_name() == "mock-embeddings"


class TestSentenceTransformerEmbedding:
    """Test sentence-transformer embedding provider."""

    @pytest.fixture
    def provider(self):
        """Create sentence-transformer provider."""
        return SentenceTransformerEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    def test_lazy_loading(self):
        """Test that model is lazy-loaded."""
        provider = SentenceTransformerEmbedding()
        assert provider._model is None

        # First embed triggers loading
        provider.embed("Test")
        assert provider._model is not None

    def test_embed_single_text(self, provider):
        """Test embedding single text."""
        text = "This is a test sentence."
        embedding = provider.embed(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # all-MiniLM-L6-v2 dimension

    def test_embed_batch(self, provider):
        """Test batch embedding."""
        texts = [
            "First sentence about Python programming.",
            "Second sentence about machine learning.",
            "Third sentence about data science.",
        ]
        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape == (384,) for emb in embeddings)

    def test_semantic_similarity(self, provider):
        """Test that similar texts have higher similarity."""
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a mat."
        text3 = "Python is a programming language."

        emb1 = provider.embed(text1)
        emb2 = provider.embed(text2)
        emb3 = provider.embed(text3)

        # Cosine similarity
        sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

        # Similar sentences should have higher similarity
        assert sim_12 > sim_13

    def test_get_dimension(self, provider):
        """Test getting dimension."""
        dim = provider.get_dimension()
        assert dim == 384

    def test_get_model_name(self, provider):
        """Test getting model name."""
        name = provider.get_model_name()
        assert "all-MiniLM-L6-v2" in name

    def test_deterministic_embeddings(self, provider):
        """Test that same text produces same embedding."""
        text = "Deterministic test"
        emb1 = provider.embed(text)
        emb2 = provider.embed(text)

        assert np.allclose(emb1, emb2)


class TestFactoryFunction:
    """Test embedding provider factory."""

    def test_create_mock_provider(self):
        """Test creating mock provider."""
        provider = create_embedding_provider(use_mock=True)
        assert isinstance(provider, MockEmbeddingProvider)

    def test_create_real_provider_default(self):
        """Test creating real provider with defaults."""
        provider = create_embedding_provider(use_mock=False)
        assert isinstance(provider, SentenceTransformerEmbedding)
        assert "all-MiniLM-L6-v2" in provider.get_model_name()

    def test_create_real_provider_custom_model(self):
        """Test creating real provider with custom model."""
        provider = create_embedding_provider(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            use_mock=False,
        )
        assert isinstance(provider, SentenceTransformerEmbedding)
