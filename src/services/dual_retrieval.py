"""Dual-index retrieval service for short-term and long-term memory.

Provides weighted retrieval across two vector stores:
- Short-term: Recent session experiences (high weight)
- Long-term: Consolidated narrative memories (lower weight, broader context)
"""

from typing import Optional, List
from pathlib import Path

import numpy as np

from src.memory.vector_store import VectorStore, VectorStoreResult, create_vector_store
from src.memory.raw_store import RawStore
from src.memory.embedding import EmbeddingProvider
from src.memory.models import ExperienceModel


class DualIndexResult:
    """Result from dual-index retrieval."""

    def __init__(
        self,
        experience_id: str,
        combined_score: float,
        short_term_score: float,
        long_term_score: float,
        is_consolidated: bool,
    ):
        """Initialize result.

        Args:
            experience_id: Experience ID
            combined_score: Weighted combined score
            short_term_score: Score from short-term index
            long_term_score: Score from long-term index
            is_consolidated: Whether this is a consolidated narrative
        """
        self.experience_id = experience_id
        self.combined_score = combined_score
        self.short_term_score = short_term_score
        self.long_term_score = long_term_score
        self.is_consolidated = is_consolidated


class DualIndexRetrieval:
    """Service for dual-index retrieval across short-term and long-term memory."""

    def __init__(
        self,
        raw_store: RawStore,
        short_term_store: VectorStore,
        long_term_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        short_term_weight: float = 0.7,
        long_term_weight: float = 0.3,
        decay_enabled: bool = True,
    ):
        """Initialize dual-index retrieval service.

        Args:
            raw_store: Raw experience store
            short_term_store: Vector store for short-term (session) memories
            long_term_store: Vector store for long-term (narrative) memories
            embedding_provider: Embedding provider
            short_term_weight: Weight for short-term results (0-1)
            long_term_weight: Weight for long-term results (0-1)
            decay_enabled: Whether to apply decay factors to scores
        """
        self.raw_store = raw_store
        self.short_term_store = short_term_store
        self.long_term_store = long_term_store
        self.embedding_provider = embedding_provider
        self.short_term_weight = short_term_weight
        self.long_term_weight = long_term_weight
        self.decay_enabled = decay_enabled

        # Validate weights
        if not abs((short_term_weight + long_term_weight) - 1.0) < 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {short_term_weight + long_term_weight}"
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[DualIndexResult]:
        """Retrieve experiences from both short-term and long-term stores.

        Args:
            query: Query text
            top_k: Maximum number of results to return
            min_score: Minimum combined score threshold

        Returns:
            List of DualIndexResults, sorted by combined score
        """
        # Embed query
        query_embedding = self.embedding_provider.embed(query)

        # Query both stores (request more than top_k to account for overlap)
        short_term_results = self.short_term_store.query(
            vector=query_embedding,
            top_k=top_k * 2,
        )
        long_term_results = self.long_term_store.query(
            vector=query_embedding,
            top_k=top_k,
        )

        # Combine results
        combined_results = self._combine_results(
            short_term_results,
            long_term_results,
        )

        # Filter by minimum score
        filtered = [r for r in combined_results if r.combined_score >= min_score]

        # Sort by combined score and limit to top_k
        filtered.sort(key=lambda r: r.combined_score, reverse=True)
        return filtered[:top_k]

    def _combine_results(
        self,
        short_term_results: List[VectorStoreResult],
        long_term_results: List[VectorStoreResult],
    ) -> List[DualIndexResult]:
        """Combine results from both indexes with weighted scoring.

        Args:
            short_term_results: Results from short-term index
            long_term_results: Results from long-term index

        Returns:
            List of DualIndexResults with combined scores
        """
        # Build lookup dicts
        short_term_dict = {r.id: r.score for r in short_term_results}
        long_term_dict = {r.id: r.score for r in long_term_results}

        # Get all unique experience IDs
        all_ids = set(short_term_dict.keys()) | set(long_term_dict.keys())

        combined = []
        for exp_id in all_ids:
            st_score = short_term_dict.get(exp_id, 0.0)
            lt_score = long_term_dict.get(exp_id, 0.0)

            # Calculate weighted combined score
            combined_score = (
                st_score * self.short_term_weight +
                lt_score * self.long_term_weight
            )

            # Apply decay factor if enabled
            if self.decay_enabled:
                combined_score = self._apply_decay(exp_id, combined_score)

            # Check if this is a consolidated narrative
            is_consolidated = exp_id in long_term_dict

            combined.append(DualIndexResult(
                experience_id=exp_id,
                combined_score=combined_score,
                short_term_score=st_score,
                long_term_score=lt_score,
                is_consolidated=is_consolidated,
            ))

        return combined

    def _apply_decay(self, experience_id: str, score: float) -> float:
        """Apply decay factor to a score.

        Args:
            experience_id: Experience ID
            score: Original score

        Returns:
            Adjusted score with decay applied
        """
        # Import here to avoid circular dependency
        from src.memory.models import MemoryDecayMetrics
        from sqlmodel import Session as DBSession

        with DBSession(self.raw_store.engine) as db:
            metrics = db.get(MemoryDecayMetrics, experience_id)
            if metrics:
                return score * metrics.decay_factor

        return score

    def retrieve_experiences(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[ExperienceModel]:
        """Retrieve full experience objects.

        Args:
            query: Query text
            top_k: Maximum number of results

        Returns:
            List of ExperienceModels
        """
        results = self.retrieve(query=query, top_k=top_k)

        experiences = []
        for result in results:
            exp = self.raw_store.get_experience(result.experience_id)
            if exp:
                experiences.append(exp)

        return experiences


def create_dual_index_retrieval(
    raw_store: RawStore,
    embedding_provider: EmbeddingProvider,
    short_term_path: Optional[str | Path] = None,
    long_term_path: Optional[str | Path] = None,
    short_term_weight: float = 0.7,
    long_term_weight: float = 0.3,
) -> DualIndexRetrieval:
    """Factory function to create DualIndexRetrieval service.

    Args:
        raw_store: Raw experience store
        embedding_provider: Embedding provider
        short_term_path: Path to short-term vector index
        long_term_path: Path to long-term vector index
        short_term_weight: Weight for short-term results
        long_term_weight: Weight for long-term results

    Returns:
        Initialized DualIndexRetrieval instance
    """
    if short_term_path is None:
        short_term_path = Path("data/vector_index_short_term/")
    if long_term_path is None:
        long_term_path = Path("data/vector_index_long_term/")

    short_term_store = create_vector_store(
        persist_directory=short_term_path,
        collection_name="short_term_experiences",
    )
    long_term_store = create_vector_store(
        persist_directory=long_term_path,
        collection_name="long_term_narratives",
    )

    return DualIndexRetrieval(
        raw_store=raw_store,
        short_term_store=short_term_store,
        long_term_store=long_term_store,
        embedding_provider=embedding_provider,
        short_term_weight=short_term_weight,
        long_term_weight=long_term_weight,
    )
