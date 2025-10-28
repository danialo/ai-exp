"""Weighted belief-memory retrieval service.

Retrieves from both beliefs and memories with configurable weighting based on query type.
Implements query type detection to adjust weights dynamically:
- Ontological queries: Heavy belief weighting (who am I? what do I believe?)
- Experiential queries: Heavy memory weighting (what happened? tell me about X)
- General queries: Memory-only retrieval
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.services.belief_vector_store import BeliefVectorStore, BeliefVectorResult
from src.services.retrieval import RetrievalService, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class CombinedRetrievalResult:
    """Combined result from belief and memory retrieval."""

    source: str  # 'belief' or 'memory'
    content: str
    score: float
    metadata: dict


class BeliefMemoryRetrieval:
    """Service for weighted retrieval from beliefs and memories.

    Detects query type and adjusts weighting accordingly:
    - Self-referential queries retrieve from both with configured weights
    - Standard queries retrieve from memories only
    """

    def __init__(
        self,
        belief_vector_store: BeliefVectorStore,
        memory_retrieval_service: RetrievalService,
        belief_weight: float = 0.7,
        memory_weight: float = 0.3,
    ):
        """Initialize belief-memory retrieval service.

        Args:
            belief_vector_store: Belief vector store
            memory_retrieval_service: Memory retrieval service
            belief_weight: Default weight for beliefs in self-queries
            memory_weight: Default weight for memories in self-queries
        """
        self.belief_vector_store = belief_vector_store
        self.memory_retrieval_service = memory_retrieval_service
        self.default_belief_weight = belief_weight
        self.default_memory_weight = memory_weight

        # Validate weights
        total = belief_weight + memory_weight
        if not (0.99 <= total <= 1.01):
            logger.warning(f"Weights sum to {total}, normalizing")
            self.default_belief_weight = belief_weight / total
            self.default_memory_weight = memory_weight / total

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        detect_query_type: bool = True,
    ) -> tuple[list[BeliefVectorResult], list[RetrievalResult]]:
        """Main retrieval method with query type detection.

        Args:
            query: Query text
            top_k: Total number of results to return
            detect_query_type: Whether to detect and adjust for query type

        Returns:
            Tuple of (belief_results, memory_results)
        """
        # Detect query type if enabled
        if detect_query_type:
            query_type = self._detect_query_type(query)
            logger.info(f"Detected query type: {query_type}")

            if query_type == "ontological":
                # Self-referential: use belief-heavy weighting
                return self._retrieve_weighted(
                    query,
                    top_k,
                    belief_weight=self.default_belief_weight,
                    memory_weight=self.default_memory_weight,
                )
            elif query_type == "experiential":
                # Experiential: use memory-heavy weighting
                return self._retrieve_weighted(
                    query,
                    top_k,
                    belief_weight=0.2,
                    memory_weight=0.8,
                )
            else:
                # General: memory-only
                memories = self.memory_retrieval_service.retrieve_similar(query, top_k=top_k)
                return ([], memories)
        else:
            # No detection: use default weights
            return self._retrieve_weighted(
                query,
                top_k,
                belief_weight=self.default_belief_weight,
                memory_weight=self.default_memory_weight,
            )

    def _detect_query_type(self, query: str) -> str:
        """Detect query type for appropriate weighting.

        Args:
            query: Query text

        Returns:
            'ontological', 'experiential', or 'general'
        """
        query_lower = query.lower()

        # Ontological patterns (self-referential, belief-based)
        ontological_patterns = [
            r"\b(who|what) (am i|are you)\b",
            r"\bdo (i|you) (believe|think|feel|have)\b",
            r"\b(what do|tell me about) (i|you) (believe|think|feel)\b",
            r"\b(my|your) (beliefs|feelings|emotions|consciousness|self|identity|values)\b",
            r"\bam (i|you) (conscious|sentient|aware)\b",
            r"\b(can|do) (i|you) (feel|experience|perceive)\b",
        ]

        for pattern in ontological_patterns:
            if re.search(pattern, query_lower):
                return "ontological"

        # Experiential patterns (memory-based, historical)
        experiential_patterns = [
            r"\b(what happened|tell me about|remember when)\b",
            r"\b(last time|previously|before|earlier)\b",
            r"\b(our conversation|we talked about|we discussed)\b",
            r"\b(what did|what have|what was)\b",
        ]

        for pattern in experiential_patterns:
            if re.search(pattern, query_lower):
                return "experiential"

        # Default: general query
        return "general"

    def _retrieve_weighted(
        self,
        query: str,
        top_k: int,
        belief_weight: float,
        memory_weight: float,
    ) -> tuple[list[BeliefVectorResult], list[RetrievalResult]]:
        """Retrieve from both sources with weighting.

        Args:
            query: Query text
            top_k: Total results to return
            belief_weight: Weight for belief results
            memory_weight: Weight for memory results

        Returns:
            Tuple of (belief_results, memory_results)
        """
        # Calculate how many results to fetch from each source
        # Fetch more than needed to have good candidates for merging
        belief_fetch_k = max(3, int(top_k * belief_weight * 1.5))
        memory_fetch_k = max(3, int(top_k * memory_weight * 1.5))

        # Retrieve from beliefs
        belief_results = self.belief_vector_store.query_beliefs(
            query=query,
            top_k=belief_fetch_k,
            min_confidence=0.3,  # Filter low-confidence beliefs
        )

        # Retrieve from memories
        memory_results = self.memory_retrieval_service.retrieve_similar(
            prompt=query,
            top_k=memory_fetch_k,
        )

        logger.info(
            f"Retrieved {len(belief_results)} beliefs and {len(memory_results)} memories "
            f"(weights: {belief_weight:.1f} beliefs / {memory_weight:.1f} memories)"
        )

        return (belief_results, memory_results)

    def merge_weighted_results(
        self,
        belief_results: list[BeliefVectorResult],
        memory_results: list[RetrievalResult],
        belief_weight: float,
        memory_weight: float,
        top_k: int,
    ) -> list[CombinedRetrievalResult]:
        """Merge and rank results by weighted scores.

        Args:
            belief_results: Results from belief retrieval
            memory_results: Results from memory retrieval
            belief_weight: Weight for belief scores
            memory_weight: Weight for memory scores
            top_k: Number of top results to return

        Returns:
            List of combined results sorted by weighted score
        """
        combined = []

        # Add belief results with weighted scores
        for belief in belief_results:
            weighted_score = belief.similarity_score * belief_weight * belief.confidence
            combined.append(
                CombinedRetrievalResult(
                    source="belief",
                    content=belief.statement,
                    score=weighted_score,
                    metadata={
                        "belief_id": belief.belief_id,
                        "belief_type": belief.belief_type,
                        "confidence": belief.confidence,
                        "evidence_count": belief.evidence_count,
                    },
                )
            )

        # Add memory results with weighted scores
        for memory in memory_results:
            weighted_score = memory.combined_score * memory_weight
            combined.append(
                CombinedRetrievalResult(
                    source="memory",
                    content=f"{memory.prompt_text}\n{memory.response_text}",
                    score=weighted_score,
                    metadata={
                        "experience_id": memory.experience_id,
                        "valence": memory.valence,
                        "similarity": memory.similarity_score,
                        "recency": memory.recency_score,
                    },
                )
            )

        # Sort by weighted score
        combined.sort(key=lambda x: x.score, reverse=True)

        # Return top_k results
        return combined[:top_k]


def create_belief_memory_retrieval(
    belief_vector_store: BeliefVectorStore,
    memory_retrieval_service: RetrievalService,
    belief_weight: float = 0.7,
    memory_weight: float = 0.3,
) -> BeliefMemoryRetrieval:
    """Factory function to create a BeliefMemoryRetrieval instance.

    Args:
        belief_vector_store: Belief vector store
        memory_retrieval_service: Memory retrieval service
        belief_weight: Weight for beliefs in self-queries
        memory_weight: Weight for memories in self-queries

    Returns:
        Configured BeliefMemoryRetrieval instance
    """
    return BeliefMemoryRetrieval(
        belief_vector_store=belief_vector_store,
        memory_retrieval_service=memory_retrieval_service,
        belief_weight=belief_weight,
        memory_weight=memory_weight,
    )
