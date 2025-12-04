"""
Conflict Detection Engine for HTN Self-Belief Decomposer.

Detects contradictions and tension candidates between beliefs:
- Contradiction: Same concept with opposite polarity
- Tension: High semantic similarity with opposite polarity

Temporal scope exclusion: past vs present beliefs don't conflict.
"""

import logging
import uuid as uuid_module
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
from sqlmodel import Session, select

from src.utils.belief_config import BeliefSystemConfig, get_belief_config
from src.services.htn_belief_embedder import HTNBeliefEmbedder
from src.memory.models.belief_node import BeliefNode
from src.memory.models.belief_occurrence import BeliefOccurrence
from src.memory.models.conflict_edge import ConflictEdge

logger = logging.getLogger(__name__)


@dataclass
class ConflictCandidate:
    """A potential conflict between two beliefs."""
    node_a: BeliefNode
    node_b: BeliefNode
    similarity: float
    conflict_type: str  # contradiction, tension
    reason: str


class ConflictEngine:
    """
    Detect contradictions and tension between belief nodes.

    Types of conflicts:
    - Contradiction: Same base concept, opposite polarity
    - Tension: High semantic similarity, opposite polarity, different concepts

    Scoped to O(k): Only checks against top_k_conflict_check most similar nodes.
    """

    def __init__(
        self,
        config: Optional[BeliefSystemConfig] = None,
        embedder: Optional[HTNBeliefEmbedder] = None,
        db_session: Optional[Session] = None
    ):
        """
        Initialize the conflict engine.

        Args:
            config: Configuration object
            embedder: Embedding service
            db_session: Database session
        """
        if config is None:
            config = get_belief_config()

        self.config = config.resolution.tension
        self.embedder = embedder or HTNBeliefEmbedder(config)
        self.db = db_session

        self.enabled = config.resolution.tension.enabled
        self.embedding_threshold = config.resolution.tension.embedding_threshold
        self.top_k = config.resolution.tension.top_k_conflict_check

    def detect_conflicts(
        self,
        new_node: BeliefNode,
        new_embedding: Optional[np.ndarray],
        new_occurrence: BeliefOccurrence
    ) -> List[ConflictEdge]:
        """
        Detect conflicts between a new node and existing nodes.

        Args:
            new_node: The newly created/updated belief node
            new_embedding: Embedding of the new node
            new_occurrence: The occurrence that triggered this

        Returns:
            List of ConflictEdge objects created
        """
        if not self.enabled:
            return []

        if not self.db:
            return []

        # Get top_k most similar existing nodes by embedding
        candidates = self._get_conflict_candidates(new_node, new_embedding)

        if not candidates:
            return []

        edges_created = []

        for candidate_node, similarity in candidates:
            # Skip self
            if candidate_node.belief_id == new_node.belief_id:
                continue

            # Check if should skip due to temporal scope
            if self._should_skip_conflict(new_node, candidate_node, new_occurrence):
                continue

            # Check for hard contradiction
            if self._is_hard_contradiction(new_node, candidate_node):
                edge = self._create_or_update_edge(
                    new_node,
                    candidate_node,
                    'contradiction',
                    new_occurrence
                )
                if edge:
                    edges_created.append(edge)
                continue

            # Check for tension candidate
            if self._is_tension_candidate(new_node, candidate_node, similarity):
                edge = self._create_or_update_edge(
                    new_node,
                    candidate_node,
                    'tension',
                    new_occurrence
                )
                if edge:
                    edges_created.append(edge)

        return edges_created

    def _get_conflict_candidates(
        self,
        node: BeliefNode,
        embedding: Optional[np.ndarray]
    ) -> List[Tuple[BeliefNode, float]]:
        """
        Get top_k most similar nodes for conflict checking.

        Only considers nodes with opposite polarity.
        """
        if not self.db:
            return []

        # Get all nodes with opposite polarity
        opposite_polarity = 'deny' if node.polarity == 'affirm' else 'affirm'

        other_nodes = self.db.exec(
            select(BeliefNode).where(
                BeliefNode.polarity == opposite_polarity,
                BeliefNode.belief_id != node.belief_id
            )
        ).all()

        if not other_nodes:
            return []

        # Compute similarities
        candidates = []

        if embedding is not None and self.embedder.enabled:
            for other in other_nodes:
                if other.embedding:
                    other_embedding = self.embedder.deserialize(other.embedding)
                    sim = self.embedder.cosine_similarity(embedding, other_embedding)
                    candidates.append((other, sim))
                else:
                    sim = self.embedder.text_similarity(
                        node.canonical_text,
                        other.canonical_text
                    )
                    candidates.append((other, sim))
        else:
            for other in other_nodes:
                sim = self.embedder.text_similarity(
                    node.canonical_text,
                    other.canonical_text
                )
                candidates.append((other, sim))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:self.top_k]

    def _should_skip_conflict(
        self,
        node_a: BeliefNode,
        node_b: BeliefNode,
        occ_a: BeliefOccurrence
    ) -> bool:
        """
        Check temporal scope exclusion rule.

        If one node has temporal_scope="past" and the other has
        ongoing/habitual/state, do NOT emit contradiction.

        "I used to love hiking" does not conflict with "I dislike hiking" (current).
        """
        # Get most recent occurrence for node_b
        occ_b = self._get_most_recent_occurrence(node_b.belief_id)
        if not occ_b:
            return False

        scope_a = occ_a.epistemic_frame.get('temporal_scope', 'unknown')
        scope_b = occ_b.epistemic_frame.get('temporal_scope', 'unknown')

        past_scopes = {'past'}
        present_scopes = {'ongoing', 'habitual', 'state'}

        if (scope_a in past_scopes and scope_b in present_scopes) or \
           (scope_b in past_scopes and scope_a in present_scopes):
            logger.debug(
                f"Skipping conflict due to temporal scope: "
                f"{scope_a} vs {scope_b}"
            )
            return True

        return False

    def _get_most_recent_occurrence(
        self,
        belief_id: uuid_module.UUID
    ) -> Optional[BeliefOccurrence]:
        """Get the most recent occurrence for a belief node."""
        if not self.db:
            return None

        return self.db.exec(
            select(BeliefOccurrence).where(
                BeliefOccurrence.belief_id == belief_id,
                BeliefOccurrence.deleted_at.is_(None)
            ).order_by(BeliefOccurrence.created_at.desc())
        ).first()

    def _is_hard_contradiction(
        self,
        node_a: BeliefNode,
        node_b: BeliefNode
    ) -> bool:
        """
        Check if two nodes are a hard contradiction.

        True if:
        - Same canonical text base (ignoring negation markers)
        - Opposite polarity

        We already filter for opposite polarity in candidates,
        so we just check if canonical texts are very similar.
        """
        # Very high text similarity = same concept
        sim = self.embedder.text_similarity(
            node_a.canonical_text,
            node_b.canonical_text
        )

        # Consider it a hard contradiction if text similarity is very high
        # This catches cases like "i am patient" vs "i am not patient"
        # where canonicalization might make them very similar
        return sim >= 0.95

    def _is_tension_candidate(
        self,
        node_a: BeliefNode,
        node_b: BeliefNode,
        similarity: float
    ) -> bool:
        """
        Check if two nodes are tension candidates.

        True if:
        - similarity >= tension threshold
        - opposite polarity (already filtered in candidates)
        - NOT a hard contradiction
        """
        if self._is_hard_contradiction(node_a, node_b):
            return False

        return similarity >= self.embedding_threshold

    def _create_or_update_edge(
        self,
        node_a: BeliefNode,
        node_b: BeliefNode,
        conflict_type: str,
        occurrence: BeliefOccurrence
    ) -> Optional[ConflictEdge]:
        """
        Create a new conflict edge or update existing.

        Args:
            node_a: First belief node
            node_b: Second belief node
            conflict_type: 'contradiction' or 'tension'
            occurrence: The occurrence that triggered this

        Returns:
            The created/updated ConflictEdge
        """
        if not self.db:
            return ConflictEdge(
                edge_id=uuid_module.uuid4(),
                from_belief_id=node_a.belief_id,
                to_belief_id=node_b.belief_id,
                conflict_type=conflict_type,
                status='tolerated',
                evidence_occurrence_ids=[str(occurrence.occurrence_id)],
            )

        # Normalize ordering for consistent lookup
        from_id = min(node_a.belief_id, node_b.belief_id, key=str)
        to_id = max(node_a.belief_id, node_b.belief_id, key=str)

        # Check if edge exists
        existing = self.db.exec(
            select(ConflictEdge).where(
                ConflictEdge.from_belief_id == from_id,
                ConflictEdge.to_belief_id == to_id,
                ConflictEdge.conflict_type == conflict_type
            )
        ).first()

        if existing:
            # Update existing edge
            if str(occurrence.occurrence_id) not in existing.evidence_occurrence_ids:
                existing.evidence_occurrence_ids.append(str(occurrence.occurrence_id))
            existing.updated_at = datetime.now(timezone.utc)
            self.db.add(existing)
            self.db.commit()
            self.db.refresh(existing)
            return existing

        # Create new edge
        edge = ConflictEdge(
            edge_id=uuid_module.uuid4(),
            from_belief_id=from_id,
            to_belief_id=to_id,
            conflict_type=conflict_type,
            status='tolerated',
            evidence_occurrence_ids=[str(occurrence.occurrence_id)],
        )

        self.db.add(edge)
        self.db.commit()
        self.db.refresh(edge)

        logger.info(
            f"Created {conflict_type} edge between "
            f"{node_a.canonical_text[:30]} and {node_b.canonical_text[:30]}"
        )

        return edge

    def get_conflicts_for_node(
        self,
        node_id: uuid_module.UUID
    ) -> List[ConflictEdge]:
        """
        Get all conflict edges involving a specific node.

        Args:
            node_id: The belief node ID

        Returns:
            List of ConflictEdge objects
        """
        if not self.db:
            return []

        return list(self.db.exec(
            select(ConflictEdge).where(
                (ConflictEdge.from_belief_id == node_id) |
                (ConflictEdge.to_belief_id == node_id)
            )
        ).all())

    def count_recent_conflicts(
        self,
        node_id: uuid_module.UUID,
        window_days: int = 30
    ) -> int:
        """
        Count conflicts involving a node within a time window.

        Args:
            node_id: The belief node ID
            window_days: Look back window in days

        Returns:
            Count of conflict edges
        """
        if not self.db:
            return 0

        cutoff = datetime.now(timezone.utc).timestamp() - (window_days * 86400)
        cutoff_dt = datetime.fromtimestamp(cutoff, tz=timezone.utc)

        conflicts = self.db.exec(
            select(ConflictEdge).where(
                (ConflictEdge.from_belief_id == node_id) |
                (ConflictEdge.to_belief_id == node_id),
                ConflictEdge.updated_at >= cutoff_dt
            )
        ).all()

        return len(conflicts)
