"""
Embedding cache with TTL and invalidation logic.

Caches embeddings by content hash to avoid redundant computation. Implements
smart invalidation based on text delta and similarity thresholds.
"""

import hashlib
import time
from typing import Optional, Dict, Tuple
import numpy as np
from src.memory.embedding import EmbeddingProvider
# Import will be done in functions to avoid circular import


class EmbeddingCache:
    """
    LRU cache for text embeddings with TTL and similarity-based invalidation.

    Caches embeddings keyed by SHA1 hash of text + model version. Re-embeds
    only when content changes significantly or cache expires.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        ttl_seconds: int = 300,
        sim_threshold: float = 0.15,
        buffer_change_threshold: float = 0.25,
        model_version: str = "v1"
    ):
        """
        Initialize embedding cache.

        Args:
            embedding_provider: Provider for generating embeddings
            ttl_seconds: Time-to-live for cache entries (default 5 min)
            sim_threshold: Cosine distance threshold for re-embedding (0.15)
            buffer_change_threshold: Buffer change ratio to trigger re-embed (0.25)
            model_version: Model version string for cache key
        """
        self.provider = embedding_provider
        self.ttl = ttl_seconds
        self.sim_threshold = sim_threshold
        self.buffer_change_threshold = buffer_change_threshold
        self.model_version = model_version

        # Cache: key -> (embedding, timestamp, text_len)
        self._cache: Dict[str, Tuple[np.ndarray, float, int]] = {}

        # Track last embedding for similarity check
        self._last_text: Optional[str] = None
        self._last_embedding: Optional[np.ndarray] = None
        self._last_buffer_size: int = 0

    def _make_key(self, text: str) -> str:
        """
        Generate cache key from text content.

        Uses SHA1 of last 512 chars + model version for key.

        Args:
            text: Input text

        Returns:
            Cache key string
        """
        # Use last 512 chars for key (most recent context)
        window = text[-512:] if len(text) > 512 else text

        # Hash with model version
        content = f"{window}:{self.model_version}"
        return hashlib.sha1(content.encode('utf-8')).hexdigest()

    def _should_reembed(
        self,
        text: str,
        buffer_size: int,
        cached_entry: Optional[Tuple[np.ndarray, float, int]]
    ) -> bool:
        """
        Determine if text should be re-embedded.

        Args:
            text: Current text
            buffer_size: Current buffer size
            cached_entry: Cached entry (embedding, timestamp, text_len) or None

        Returns:
            True if re-embedding needed
        """
        # No cache entry -> embed
        if cached_entry is None:
            return True

        cached_emb, cached_ts, cached_len = cached_entry

        # Check TTL expiry
        if time.time() - cached_ts > self.ttl:
            return True

        # Check if buffer changed significantly
        if self._last_buffer_size > 0:
            change_ratio = abs(buffer_size - self._last_buffer_size) / self._last_buffer_size
            if change_ratio > self.buffer_change_threshold:
                return True

        # Check text similarity to last embedding
        if self._last_text is not None and self._last_embedding is not None:
            # Compute cosine distance
            dist = self._cosine_distance(text, self._last_text)
            if dist > self.sim_threshold:
                return True

        return False

    def _cosine_distance(self, text1: str, text2: str) -> float:
        """
        Compute cosine distance between two texts.

        Uses simple character-level comparison for efficiency.
        For exact embeddings, would need to compute both vectors.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Approximate cosine distance [0, 1]
        """
        # Simple heuristic: Jaccard distance on character 3-grams
        def get_ngrams(s: str, n: int = 3) -> set:
            return set(s[i:i+n] for i in range(len(s) - n + 1))

        if not text1 or not text2:
            return 1.0

        ng1 = get_ngrams(text1[-512:])  # Last 512 chars
        ng2 = get_ngrams(text2[-512:])

        if not ng1 or not ng2:
            return 1.0

        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)

        jaccard_sim = intersection / union if union > 0 else 0.0
        # Convert to distance
        return 1.0 - jaccard_sim

    async def get_embedding(
        self,
        text: str,
        buffer_size: int
    ) -> Optional[np.ndarray]:
        """
        Get embedding for text, using cache when possible.

        Args:
            text: Text to embed
            buffer_size: Current percept buffer size

        Returns:
            Embedding vector, or None if embedding fails
        """
        if not text:
            return None

        key = self._make_key(text)
        cached_entry = self._cache.get(key)

        # Check if re-embedding needed
        needs_reembed = self._should_reembed(text, buffer_size, cached_entry)

        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[CACHE] key={key[:16]}, cached={cached_entry is not None}, needs_reembed={needs_reembed}")

        if needs_reembed:
            try:
                # Generate new embedding
                embedding = await self._embed(text)

                if embedding is not None:
                    # Update cache
                    self._cache[key] = (embedding, time.time(), len(text))

                    # Update last state
                    self._last_text = text
                    self._last_embedding = embedding
                    self._last_buffer_size = buffer_size

                    # Track cache miss
                    try:
                        from src.services import awareness_metrics
                        metrics = awareness_metrics.get_metrics()
                        metrics.increment_counter("embedding_cache_misses")
                    except:
                        pass
                    return embedding

            except Exception as e:
                logger.error(f"[CACHE] Exception during embedding: {e}")
                # Fall back to cached if available
                if cached_entry is not None:
                    try:
                        from src.services import awareness_metrics
                        metrics = awareness_metrics.get_metrics()
                        metrics.increment_counter("embedding_cache_hits")
                    except:
                        pass
                    return cached_entry[0]
                return None

        # Use cached embedding
        if cached_entry is not None:
            try:
                from src.services import awareness_metrics
                metrics = awareness_metrics.get_metrics()
                metrics.increment_counter("embedding_cache_hits")
            except:
                pass
            return cached_entry[0]

        # No cache and didn't reembed - shouldn't happen
        logger.warning(f"[CACHE] No cached entry and didn't reembed - returning None")
        return None

    async def _embed(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding using provider.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            # Use last 512 chars for embedding
            window = text[-512:] if len(text) > 512 else text

            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[EMBED] Generating embedding for {len(window)} chars")

            embedding = self.provider.embed(window)

            # Ensure correct dtype
            if embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)

            logger.debug(f"[EMBED] Generated embedding shape: {embedding.shape}")
            return embedding

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[EMBED] Failed to generate embedding: {e}")
            return None

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._last_text = None
        self._last_embedding = None
        self._last_buffer_size = 0

    def prune_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, (_, ts, _) in self._cache.items()
            if now - ts > self.ttl
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        return {
            "size": len(self._cache),
            "ttl_seconds": self.ttl,
            "sim_threshold": self.sim_threshold,
            "buffer_change_threshold": self.buffer_change_threshold,
            "has_last_embedding": self._last_embedding is not None
        }
