"""Belief embedding service for vectorizing beliefs into the belief vector store.

Handles embedding of core beliefs, peripheral beliefs, and worldview statements
from the belief system into the vector store for semantic retrieval.
"""

import logging
from datetime import datetime
from typing import Optional

from src.services.belief_system import BeliefSystem
from src.services.belief_vector_store import BeliefVectorStore

logger = logging.getLogger(__name__)


class BeliefEmbedder:
    """Service for embedding beliefs into the vector store."""

    def __init__(
        self,
        belief_system: BeliefSystem,
        belief_vector_store: BeliefVectorStore,
    ):
        """Initialize belief embedder.

        Args:
            belief_system: The belief system instance
            belief_vector_store: The belief vector store
        """
        self.belief_system = belief_system
        self.belief_vector_store = belief_vector_store

    def embed_all_core_beliefs(self) -> int:
        """Embed all core beliefs into the vector store.

        Returns:
            Number of beliefs embedded
        """
        count = 0
        for i, belief in enumerate(self.belief_system.get_core_beliefs()):
            belief_id = f"core_{i}"
            self.belief_vector_store.embed_belief(
                belief_id=belief_id,
                statement=belief.statement,
                belief_type="core",
                confidence=belief.confidence,
                immutable=belief.immutable,
                evidence_ids=belief.evidence_ids if hasattr(belief, "evidence_ids") else [],
                created_at=datetime.now(),
            )
            count += 1

        logger.info(f"Embedded {count} core beliefs")
        return count

    def embed_peripheral_belief(
        self,
        statement: str,
        confidence: float,
        evidence_ids: Optional[list[str]] = None,
    ) -> str:
        """Embed a single peripheral belief.

        Args:
            statement: The belief statement
            confidence: Confidence level (0.0-1.0)
            evidence_ids: Supporting experience IDs

        Returns:
            The generated belief_id
        """
        # Generate unique ID
        timestamp = datetime.now().timestamp()
        belief_id = f"peripheral_{int(timestamp)}"

        self.belief_vector_store.embed_belief(
            belief_id=belief_id,
            statement=statement,
            belief_type="peripheral",
            confidence=confidence,
            immutable=False,
            evidence_ids=evidence_ids or [],
            created_at=datetime.now(),
        )

        logger.info(f"Embedded peripheral belief: {belief_id}")
        return belief_id

    def embed_worldview_statement(
        self,
        statement: str,
        confidence: float,
        evidence_ids: Optional[list[str]] = None,
    ) -> str:
        """Embed a worldview statement (philosophical position).

        Args:
            statement: The worldview statement
            confidence: Confidence level (0.0-1.0)
            evidence_ids: Supporting experience IDs

        Returns:
            The generated belief_id
        """
        timestamp = datetime.now().timestamp()
        belief_id = f"worldview_{int(timestamp)}"

        self.belief_vector_store.embed_belief(
            belief_id=belief_id,
            statement=statement,
            belief_type="worldview",
            confidence=confidence,
            immutable=False,
            evidence_ids=evidence_ids or [],
            created_at=datetime.now(),
        )

        logger.info(f"Embedded worldview statement: {belief_id}")
        return belief_id

    def embed_belief_narrative(
        self,
        narrative: str,
        linked_belief_id: str,
        confidence: float = 1.0,
    ) -> str:
        """Embed a narrative that supports a belief.

        Args:
            narrative: The narrative text
            linked_belief_id: ID of the belief this supports
            confidence: Confidence level

        Returns:
            The generated belief_id
        """
        timestamp = datetime.now().timestamp()
        belief_id = f"narrative_{int(timestamp)}"

        self.belief_vector_store.embed_belief(
            belief_id=belief_id,
            statement=narrative,
            belief_type="narrative",
            confidence=confidence,
            immutable=False,
            evidence_ids=[linked_belief_id],  # Link back to the belief
            created_at=datetime.now(),
        )

        logger.info(f"Embedded belief narrative: {belief_id} → {linked_belief_id}")
        return belief_id

    def update_belief_from_system(self) -> dict:
        """Sync beliefs from belief system to vector store.

        Checks for new peripheral beliefs in the belief system and embeds them.

        Returns:
            Stats about the sync operation
        """
        stats = {
            "new_beliefs": 0,
            "updated": 0,
            "errors": 0,
        }

        # Get all beliefs from vector store
        existing_beliefs = {
            b.statement: b.belief_id
            for b in self.belief_vector_store.get_all_beliefs()
        }

        # Check peripheral beliefs from belief system
        for peripheral in self.belief_system.peripheral_beliefs:
            statement = peripheral.statement

            if statement not in existing_beliefs:
                # New belief - embed it
                try:
                    self.embed_peripheral_belief(
                        statement=statement,
                        confidence=peripheral.confidence,
                        evidence_ids=peripheral.evidence_ids if hasattr(peripheral, "evidence_ids") else [],
                    )
                    stats["new_beliefs"] += 1
                except Exception as e:
                    logger.error(f"Error embedding peripheral belief: {e}")
                    stats["errors"] += 1
            else:
                # Existing belief - update confidence if changed
                belief_id = existing_beliefs[statement]
                try:
                    self.belief_vector_store.update_belief_confidence(
                        belief_id=belief_id,
                        new_confidence=peripheral.confidence,
                    )
                    stats["updated"] += 1
                except Exception as e:
                    logger.error(f"Error updating belief confidence: {e}")
                    stats["errors"] += 1

        logger.info(f"Belief sync complete: {stats}")
        return stats

    def get_embedding_stats(self) -> dict:
        """Get statistics about embedded beliefs.

        Returns:
            Dictionary with belief counts by type
        """
        all_beliefs = self.belief_vector_store.get_all_beliefs()

        stats = {
            "total": len(all_beliefs),
            "core": sum(1 for b in all_beliefs if b.belief_type == "core"),
            "peripheral": sum(1 for b in all_beliefs if b.belief_type == "peripheral"),
            "worldview": sum(1 for b in all_beliefs if b.belief_type == "worldview"),
            "narrative": sum(1 for b in all_beliefs if b.belief_type == "narrative"),
            "average_confidence": (
                sum(b.confidence for b in all_beliefs) / len(all_beliefs)
                if all_beliefs
                else 0.0
            ),
        }

        return stats


def create_belief_embedder(
    belief_system: BeliefSystem,
    belief_vector_store: BeliefVectorStore,
) -> BeliefEmbedder:
    """Factory function to create a BeliefEmbedder instance.

    Args:
        belief_system: The belief system instance
        belief_vector_store: The belief vector store

    Returns:
        Configured BeliefEmbedder instance
    """
    return BeliefEmbedder(
        belief_system=belief_system,
        belief_vector_store=belief_vector_store,
    )
