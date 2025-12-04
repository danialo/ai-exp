"""
Activation Service for HTN Self-Belief Decomposer.

Computes belief activation with time-based decay.
Activation = sum of source_weight * exp(-age_days / half_life)
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional
import uuid as uuid_module

from sqlmodel import Session, select

from src.utils.belief_config import BeliefSystemConfig, get_belief_config
from src.memory.models.belief_node import BeliefNode
from src.memory.models.belief_occurrence import BeliefOccurrence
from src.memory.models.stream_assignment import StreamAssignment

logger = logging.getLogger(__name__)


class ActivationService:
    """
    Compute and update belief activation with decay.

    Activation is a recency-weighted sum of occurrences:
    activation = Σ (source_weight * exp(-age_days / half_life))

    Half-life varies by stream:
    - identity: 60 days (slow decay for core beliefs)
    - state: 7 days (fast decay for transient states)
    - meta: 30 days
    - relational: 30 days
    """

    def __init__(
        self,
        config: Optional[BeliefSystemConfig] = None,
        db_session: Optional[Session] = None
    ):
        """
        Initialize the activation service.

        Args:
            config: Configuration object
            db_session: Database session
        """
        if config is None:
            config = get_belief_config()

        self.config = config.scoring
        self.half_life_days = config.scoring.half_life_days
        self.db = db_session

    def compute_activation(
        self,
        node: BeliefNode,
        stream: Optional[StreamAssignment] = None
    ) -> float:
        """
        Compute activation from all non-deleted occurrences.

        activation = Σ (source_weight * exp(-age_days / half_life))

        Args:
            node: The belief node
            stream: Optional stream assignment (for half-life lookup)

        Returns:
            Computed activation value
        """
        if not self.db:
            return 0.0

        # Get half-life based on stream
        if stream:
            half_life = self.half_life_days.get(stream.primary_stream, 30)
        else:
            half_life = 30  # Default

        # Get all non-deleted occurrences
        occurrences = self.db.exec(
            select(BeliefOccurrence).where(
                BeliefOccurrence.belief_id == node.belief_id,
                BeliefOccurrence.deleted_at.is_(None)
            )
        ).all()

        if not occurrences:
            return 0.0

        now = datetime.now(timezone.utc)
        total_activation = 0.0

        for occ in occurrences:
            # Calculate age in days
            age_days = (now - occ.created_at).total_seconds() / 86400

            # Exponential decay
            weight = occ.source_weight * math.exp(-age_days / half_life)
            total_activation += weight

        return total_activation

    def update_activation(
        self,
        node: BeliefNode,
        stream: Optional[StreamAssignment] = None
    ) -> float:
        """
        Compute and save activation to the node.

        Args:
            node: The belief node
            stream: Optional stream assignment

        Returns:
            The new activation value
        """
        activation = self.compute_activation(node, stream)

        node.activation = activation
        node.last_reinforced_at = datetime.now(timezone.utc)

        if self.db:
            self.db.add(node)
            self.db.commit()

        return activation

    def get_stream_for_node(
        self,
        node_id: uuid_module.UUID
    ) -> Optional[StreamAssignment]:
        """Get stream assignment for a node."""
        if not self.db:
            return None

        return self.db.exec(
            select(StreamAssignment).where(
                StreamAssignment.belief_id == node_id
            )
        ).first()

    def update_all_activations(self) -> int:
        """
        Update activations for all nodes.

        Returns:
            Count of nodes updated
        """
        if not self.db:
            return 0

        nodes = self.db.exec(select(BeliefNode)).all()
        count = 0

        for node in nodes:
            stream = self.get_stream_for_node(node.belief_id)
            self.update_activation(node, stream)
            count += 1

        return count
