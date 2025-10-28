"""Belief vector store for semantic belief retrieval.

Specialized vector store for beliefs, worldview statements, and supporting narratives.
Extends the base vector store pattern with belief-specific metadata and operations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from src.memory.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class BeliefVectorResult:
    """Result from belief vector store query."""

    belief_id: str
    statement: str
    belief_type: str  # 'core', 'peripheral', 'worldview', 'narrative'
    similarity_score: float
    confidence: float
    evidence_count: int
    immutable: bool
    created_at: datetime


class BeliefVectorStore:
    """Vector store specialized for beliefs and worldview statements.

    Stores beliefs with rich metadata including type, confidence, evidence references,
    and mutability. Supports weighted retrieval based on belief type and confidence.
    """

    def __init__(
        self,
        persist_directory: str | Path,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "beliefs",
        reset: bool = False,
    ):
        """Initialize belief vector store.

        Args:
            persist_directory: Directory to persist vector index
            embedding_provider: Provider for generating embeddings
            collection_name: Name of the ChromaDB collection
            reset: If True, delete existing collection and start fresh
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        if reset:
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"Reset belief collection: {collection_name}")
            except ValueError:
                pass  # Collection doesn't exist

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(f"Belief vector store initialized: {self.persist_directory}")

    def embed_belief(
        self,
        belief_id: str,
        statement: str,
        belief_type: str,
        confidence: float = 1.0,
        immutable: bool = False,
        evidence_ids: Optional[list[str]] = None,
        created_at: Optional[datetime] = None,
    ) -> None:
        """Embed a belief statement into the vector store.

        Args:
            belief_id: Unique identifier for the belief
            statement: The belief statement text
            belief_type: Type of belief ('core', 'peripheral', 'worldview', 'narrative')
            confidence: Confidence level (0.0-1.0)
            immutable: Whether belief can be modified
            evidence_ids: List of experience IDs supporting this belief
            created_at: Timestamp when belief was formed
        """
        # Generate embedding
        embedding = self.embedding_provider.embed(statement)

        # Prepare metadata
        metadata = {
            "statement": statement,
            "belief_type": belief_type,
            "confidence": confidence,
            "immutable": immutable,
            "evidence_count": len(evidence_ids) if evidence_ids else 0,
            "created_at": created_at.isoformat() if created_at else datetime.now().isoformat(),
        }

        # Store evidence IDs as comma-separated string (ChromaDB limitation)
        if evidence_ids:
            metadata["evidence_ids"] = ",".join(evidence_ids)

        # Convert numpy array to list
        vector_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        # Upsert into collection
        self.collection.upsert(
            ids=[belief_id],
            embeddings=[vector_list],
            metadatas=[metadata],
        )

        logger.info(f"Embedded belief: {belief_id} (type={belief_type}, confidence={confidence})")

    def query_beliefs(
        self,
        query: str,
        top_k: int = 5,
        belief_types: Optional[list[str]] = None,
        min_confidence: float = 0.0,
    ) -> list[BeliefVectorResult]:
        """Query beliefs by semantic similarity.

        Args:
            query: Query text to search for
            top_k: Number of results to return
            belief_types: Optional filter by belief types
            min_confidence: Minimum confidence threshold

        Returns:
            List of belief results sorted by similarity
        """
        # Generate query embedding
        query_embedding = self.embedding_provider.embed(query)
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        # Build metadata filter
        where = None
        if belief_types or min_confidence > 0:
            where = {}
            if belief_types:
                where["belief_type"] = {"$in": belief_types}
            if min_confidence > 0:
                where["confidence"] = {"$gte": min_confidence}

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=top_k,
            where=where,
        )

        # Parse results
        belief_results = []
        if results and results["ids"]:
            for i, belief_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1.0 - distance  # Convert distance to similarity

                result = BeliefVectorResult(
                    belief_id=belief_id,
                    statement=metadata.get("statement", ""),
                    belief_type=metadata.get("belief_type", "unknown"),
                    similarity_score=similarity,
                    confidence=metadata.get("confidence", 0.0),
                    evidence_count=metadata.get("evidence_count", 0),
                    immutable=metadata.get("immutable", False),
                    created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                )
                belief_results.append(result)

        logger.info(f"Query returned {len(belief_results)} beliefs (query='{query[:50]}...')")
        return belief_results

    def update_belief_confidence(self, belief_id: str, new_confidence: float) -> None:
        """Update confidence level of an existing belief.

        Args:
            belief_id: ID of belief to update
            new_confidence: New confidence value (0.0-1.0)
        """
        # Get existing belief
        result = self.collection.get(ids=[belief_id])
        if not result or not result["ids"]:
            logger.warning(f"Belief not found for update: {belief_id}")
            return

        # Update metadata
        metadata = result["metadatas"][0]
        metadata["confidence"] = new_confidence

        # Re-upsert with updated metadata (keeping same embedding)
        self.collection.update(
            ids=[belief_id],
            metadatas=[metadata],
        )

        logger.info(f"Updated belief confidence: {belief_id} → {new_confidence}")

    def delete_belief(self, belief_id: str) -> None:
        """Delete a belief from the vector store.

        Args:
            belief_id: ID of belief to delete
        """
        try:
            self.collection.delete(ids=[belief_id])
            logger.info(f"Deleted belief: {belief_id}")
        except Exception as e:
            logger.error(f"Error deleting belief {belief_id}: {e}")

    def get_all_beliefs(self) -> list[BeliefVectorResult]:
        """Retrieve all beliefs from the store.

        Returns:
            List of all beliefs
        """
        # Get all items (ChromaDB doesn't have a direct "get all" so we use a large limit)
        results = self.collection.get(limit=1000)

        belief_results = []
        if results and results["ids"]:
            for i, belief_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                result = BeliefVectorResult(
                    belief_id=belief_id,
                    statement=metadata.get("statement", ""),
                    belief_type=metadata.get("belief_type", "unknown"),
                    similarity_score=1.0,  # No similarity for get_all
                    confidence=metadata.get("confidence", 0.0),
                    evidence_count=metadata.get("evidence_count", 0),
                    immutable=metadata.get("immutable", False),
                    created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                )
                belief_results.append(result)

        return belief_results

    def count(self) -> int:
        """Get total number of beliefs in the store.

        Returns:
            Count of beliefs
        """
        return self.collection.count()


def create_belief_vector_store(
    persist_directory: str | Path,
    embedding_provider: EmbeddingProvider,
    reset: bool = False,
) -> BeliefVectorStore:
    """Factory function to create a BeliefVectorStore instance.

    Args:
        persist_directory: Directory for persistence
        embedding_provider: Embedding provider instance
        reset: Whether to reset the collection

    Returns:
        Configured BeliefVectorStore instance
    """
    return BeliefVectorStore(
        persist_directory=persist_directory,
        embedding_provider=embedding_provider,
        reset=reset,
    )
