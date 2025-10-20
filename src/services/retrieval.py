"""Retrieval service for semantic memory search.

Provides recency-biased similarity search over experiences with
configurable weighting between semantic similarity and recency.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.memory.embedding import EmbeddingProvider
from src.memory.models import ExperienceModel
from src.memory.raw_store import RawStore
from src.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval query."""

    experience_id: str
    prompt_text: str
    response_text: str
    valence: float
    similarity_score: float
    recency_score: float
    combined_score: float
    created_at: datetime


class RetrievalService:
    """Service for retrieving relevant experiences via semantic search."""

    def __init__(
        self,
        raw_store: RawStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        semantic_weight: float = 0.8,
        recency_weight: float = 0.2,
    ):
        """Initialize retrieval service.

        Args:
            raw_store: Raw experience store
            vector_store: Vector index
            embedding_provider: Embedding generator
            semantic_weight: Weight for semantic similarity (default 0.8)
            recency_weight: Weight for recency (default 0.2)
        """
        self.raw_store = raw_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight

        # Validate weights sum to 1.0
        total_weight = semantic_weight + recency_weight
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(
                f"Weights sum to {total_weight}, not 1.0. "
                f"Normalizing: semantic={semantic_weight/total_weight:.2f}, "
                f"recency={recency_weight/total_weight:.2f}"
            )
            self.semantic_weight = semantic_weight / total_weight
            self.recency_weight = recency_weight / total_weight

    def retrieve_similar(
        self,
        prompt: str,
        top_k: int = 5,
        max_age_days: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Retrieve experiences similar to the given prompt.

        Args:
            prompt: Query prompt
            top_k: Number of results to return
            max_age_days: Optional maximum age of experiences in days

        Returns:
            List of RetrievalResults, sorted by combined score (highest first)
        """
        logger.info(f"Retrieving similar experiences for prompt (top_k={top_k})")

        # Generate query embedding
        query_embedding = self.embedding_provider.embed(prompt)

        # Query vector store (get more candidates for recency filtering)
        # Each experience has 2 vectors (prompt + response), so request extra
        # to ensure we get enough unique experiences after deduplication
        candidate_multiplier = 3 if max_age_days else 3
        vector_results = self.vector_store.query(
            query_embedding,
            top_k=top_k * candidate_multiplier,
        )

        if not vector_results:
            logger.info("No similar experiences found")
            return []

        # Extract experience IDs from vector results
        # Vector IDs are in format: {experience_id}_prompt or {experience_id}_response
        experience_ids = set()
        similarity_scores = {}

        for vec_result in vector_results:
            # Parse experience ID from vector ID
            parts = vec_result.id.rsplit("_", 1)
            if len(parts) == 2:
                exp_id = parts[0]
                experience_ids.add(exp_id)
                # Keep highest similarity score if multiple vectors per experience
                # Clamp score to [0, 1] to handle floating-point precision issues
                score = max(0.0, min(1.0, vec_result.score))
                if exp_id not in similarity_scores or score > similarity_scores[exp_id]:
                    similarity_scores[exp_id] = score

        # Fetch experiences from raw store
        results = []
        now = datetime.now(timezone.utc)

        for exp_id in experience_ids:
            experience = self.raw_store.get_experience(exp_id)
            if experience is None:
                logger.warning(f"Experience {exp_id} not found in raw store")
                continue

            # Apply age filter if specified
            if max_age_days:
                age = now - experience.created_at
                if age > timedelta(days=max_age_days):
                    logger.debug(f"Skipping {exp_id}: too old ({age.days} days)")
                    continue

            # Calculate recency score (newer = higher score)
            age_seconds = (now - experience.created_at).total_seconds()
            # Decay over 30 days: recency = exp(-age_days / 30)
            age_days = age_seconds / (24 * 3600)
            recency_score = 2 ** (-age_days / 30)  # Exponential decay

            # Calculate combined score
            similarity = similarity_scores.get(exp_id, 0.0)
            combined_score = (self.semantic_weight * similarity) + (
                self.recency_weight * recency_score
            )

            # Extract prompt and response from structured content
            prompt_text = experience.content.structured.get("prompt", "")
            response_text = experience.content.structured.get("response", "")
            valence = experience.affect.vad.v

            results.append(
                RetrievalResult(
                    experience_id=exp_id,
                    prompt_text=prompt_text,
                    response_text=response_text,
                    valence=valence,
                    similarity_score=similarity,
                    recency_score=recency_score,
                    combined_score=combined_score,
                    created_at=experience.created_at,
                )
            )

        # Sort by combined score (descending)
        results.sort(key=lambda r: r.combined_score, reverse=True)

        # Return top_k results
        final_results = results[:top_k]

        logger.info(
            f"Retrieved {len(final_results)} experiences "
            f"(filtered from {len(results)} candidates)"
        )

        return final_results

    def get_experience_details(self, experience_id: str) -> Optional[ExperienceModel]:
        """Retrieve full experience details by ID.

        Args:
            experience_id: Experience ID

        Returns:
            ExperienceModel if found, None otherwise
        """
        return self.raw_store.get_experience(experience_id)


def create_retrieval_service(
    raw_store: RawStore,
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
    semantic_weight: float = 0.8,
    recency_weight: float = 0.2,
) -> RetrievalService:
    """Factory function to create retrieval service.

    Args:
        raw_store: Raw store instance
        vector_store: Vector store instance
        embedding_provider: Embedding provider instance
        semantic_weight: Weight for semantic similarity
        recency_weight: Weight for recency

    Returns:
        RetrievalService instance
    """
    return RetrievalService(
        raw_store=raw_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        semantic_weight=semantic_weight,
        recency_weight=recency_weight,
    )
