"""Embedding provider abstraction for semantic vector generation.

Supports both local sentence-transformers models and OpenAI embeddings.
MVP focuses on sentence-transformers for offline operation.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding vector dimension.

        Returns:
            Dimension of embedding vectors
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model identifier.

        Returns:
            Model name/version string
        """
        pass


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Embedding provider using sentence-transformers library.

    Uses local models for offline operation. Default model is
    'all-MiniLM-L6-v2' (384 dimensions, good balance of speed/quality).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize sentence-transformer model.

        Args:
            model_name: HuggingFace model name or path
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            # Cache dimension
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text.

        Args:
            text: Input text

        Returns:
            Embedding vector (numpy array)
        """
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        # Convert to list of arrays
        return [embeddings[i] for i in range(len(embeddings))]

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Vector dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        if self._dimension is None:
            # Trigger lazy load
            _ = self.model
        return self._dimension

    def get_model_name(self) -> str:
        """Get model identifier.

        Returns:
            Model name
        """
        return self.model_name


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing.

    Returns deterministic embeddings that simulate semantic similarity
    based on word overlap and token similarity.
    """

    def __init__(self, dimension: int = 384):
        """Initialize mock provider.

        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.model_name = "mock-embeddings"

    def _tokenize(self, text: str) -> set[str]:
        """Simple tokenization for similarity calculation."""
        # Lowercase and split on whitespace/punctuation
        import re
        tokens = re.findall(r'\w+', text.lower())
        return set(tokens)

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding with semantic similarity.

        Creates embeddings where similar texts (by token overlap) produce
        similar vectors, simulating real semantic embeddings.

        Args:
            text: Input text

        Returns:
            Deterministic vector that encodes token information
        """
        tokens = self._tokenize(text)

        # Create base embedding from text hash for determinism
        seed = hash(text) % (2**32)
        rng = np.random.RandomState(seed)
        base = rng.randn(self.dimension).astype(np.float32) * 0.3  # Scale down

        # Add components for each token to simulate semantic similarity
        for token in tokens:
            token_seed = hash(token) % (2**32)
            token_rng = np.random.RandomState(token_seed)
            token_vec = token_rng.randn(self.dimension).astype(np.float32)
            # Weight by 1/sqrt(num_tokens) to keep magnitude reasonable
            base += token_vec / np.sqrt(max(len(tokens), 1))

        # Normalize to unit length
        embedding = base / np.linalg.norm(base)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate mock embeddings for batch.

        Args:
            texts: List of texts

        Returns:
            List of mock embeddings
        """
        return [self.embed(text) for text in texts]

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Dimension
        """
        return self.dimension

    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Model identifier
        """
        return self.model_name


def create_embedding_provider(
    model_name: Optional[str] = None,
    use_mock: bool = False,
) -> EmbeddingProvider:
    """Factory function to create embedding provider.

    Args:
        model_name: Model to use (defaults to all-MiniLM-L6-v2)
        use_mock: Use mock provider for testing

    Returns:
        EmbeddingProvider instance
    """
    if use_mock:
        return MockEmbeddingProvider()

    if model_name is None:
        model_name = "all-MiniLM-L6-v2"

    return SentenceTransformerEmbedding(model_name)
