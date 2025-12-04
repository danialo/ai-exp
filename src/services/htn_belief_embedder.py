"""
HTN Belief Embedder for HTN Self-Belief Decomposer.

Computes embeddings for belief atoms before resolution.
Supports linear scan for small node counts, requires vector index for larger.
"""

import logging
from typing import List, Optional

import numpy as np

from src.utils.belief_config import BeliefSystemConfig, get_belief_config

logger = logging.getLogger(__name__)


class HTNBeliefEmbedder:
    """
    Compute and manage embeddings for belief atoms.

    Supports:
    - Embedding text to vectors
    - Batch embedding for efficiency
    - Serialization for database storage
    - Cosine similarity computation
    - Linear scan vs vector index decision
    """

    def __init__(self, config: Optional[BeliefSystemConfig] = None):
        """
        Initialize the embedder.

        Args:
            config: Configuration object. If None, loads from default.
        """
        if config is None:
            config = get_belief_config()

        self.config = config.embeddings
        self.enabled = config.embeddings.enabled
        self.model_name = config.embeddings.model
        self.dimension = config.embeddings.dimension
        self.batch_size = config.embeddings.batch_size
        self.linear_scan_max = config.embeddings.linear_scan_max_nodes
        self.fallback_to_text = config.embeddings.fallback_to_text_similarity
        self.text_similarity_method = config.embeddings.text_similarity_method

        # Lazy load embedding model
        self._model = None

    def _get_model(self):
        """Lazy load the embedding model."""
        if self._model is None and self.enabled:
            try:
                # Try sentence-transformers first (local)
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded SentenceTransformer model for HTN embedder")
            except ImportError:
                logger.warning("sentence-transformers not available, embeddings disabled")
                self.enabled = False
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.enabled = False
        return self._model

    def embed(self, text: str) -> Optional[np.ndarray]:
        """
        Compute embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Numpy array of embedding or None if disabled
        """
        if not self.enabled:
            return None

        model = self._get_model()
        if model is None:
            return None

        try:
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Batch embed multiple texts for efficiency.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (or None for failures)
        """
        if not self.enabled or not texts:
            return [None] * len(texts)

        model = self._get_model()
        if model is None:
            return [None] * len(texts)

        try:
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                embeddings = model.encode(batch, convert_to_numpy=True)
                all_embeddings.extend([e.astype(np.float32) for e in embeddings])
            return all_embeddings
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
            return [None] * len(texts)

    def serialize(self, embedding: np.ndarray) -> bytes:
        """
        Serialize embedding for database storage.

        Args:
            embedding: Numpy array

        Returns:
            Bytes representation
        """
        return embedding.tobytes()

    def deserialize(self, data: bytes, dimension: Optional[int] = None) -> np.ndarray:
        """
        Deserialize embedding from database.

        Args:
            data: Bytes from database
            dimension: Expected dimension (uses config if not provided)

        Returns:
            Numpy array
        """
        if dimension is None:
            # Infer from data size
            dimension = len(data) // 4  # float32 = 4 bytes
        return np.frombuffer(data, dtype=np.float32).reshape(-1)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity in [-1, 1]
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def should_use_linear_scan(self, node_count: int) -> bool:
        """
        Determine if linear scan should be used.

        Args:
            node_count: Current number of belief nodes

        Returns:
            True if linear scan is appropriate
        """
        return node_count < self.linear_scan_max

    def text_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute text similarity using configured method.

        Fallback when embeddings are not available.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Similarity in [0, 1]
        """
        if self.text_similarity_method == 'levenshtein_ratio':
            return self._levenshtein_ratio(text_a, text_b)
        else:
            # Default to exact match
            return 1.0 if text_a.lower() == text_b.lower() else 0.0

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """
        Compute Levenshtein similarity ratio.

        Returns value in [0, 1] where 1 = identical.
        """
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()

        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Dynamic programming for edit distance
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )

        distance = dp[len1][len2]
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len)


def get_htn_embedder() -> HTNBeliefEmbedder:
    """Get a singleton HTN embedder instance."""
    return HTNBeliefEmbedder()
