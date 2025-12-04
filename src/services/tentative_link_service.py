"""
Tentative Link Service for HTN Self-Belief Decomposer.

Manages uncertain match links between belief nodes.
- Creates links for uncertain matches
- Tracks support evidence
- Updates confidence using decay formula
- Auto-accepts/rejects based on thresholds (NO auto-merge!)
"""

import logging
import math
import uuid as uuid_module
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from sqlmodel import Session, select

from src.utils.belief_config import BeliefSystemConfig, get_belief_config
from src.memory.models.belief_node import BeliefNode
from src.memory.models.tentative_link import TentativeLink

logger = logging.getLogger(__name__)


@dataclass
class TentativeLinkUpdate:
    """
    Result of a tentative link update.

    Attributes:
        link: The updated link
        old_status: Previous status
        new_status: Current status
        old_confidence: Previous confidence
        new_confidence: Current confidence
        merge_required: True if auto-accepted (NOT auto-merged!)
    """
    link: TentativeLink
    old_status: str
    new_status: str
    old_confidence: float
    new_confidence: float
    merge_required: bool  # True if status changed to accepted


class TentativeLinkService:
    """
    Manage uncertain match links between belief nodes.

    IMPORTANT: Auto-accept sets status="accepted" but does NOT auto-merge.
    Merge is a separate future operation that requires explicit invocation.
    """

    def __init__(
        self,
        config: Optional[BeliefSystemConfig] = None,
        db_session: Optional[Session] = None
    ):
        """
        Initialize the service.

        Args:
            config: Configuration object
            db_session: Database session
        """
        if config is None:
            config = get_belief_config()

        self.config = config.resolution.tentative_link
        self.db = db_session

        self.auto_accept_threshold = config.resolution.tentative_link.auto_accept_threshold
        self.auto_reject_threshold = config.resolution.tentative_link.auto_reject_threshold

        params = config.resolution.tentative_link.confidence_params
        self.a = params.a  # support_both weight
        self.b = params.b  # support_one weight
        self.c = params.c  # age decay per day

    def _normalize_ids(
        self,
        id_a: uuid_module.UUID,
        id_b: uuid_module.UUID
    ) -> Tuple[uuid_module.UUID, uuid_module.UUID]:
        """
        Normalize IDs so from_id < to_id (by string comparison).

        This ensures consistent ordering for the unique constraint.
        """
        if str(id_a) < str(id_b):
            return id_a, id_b
        return id_b, id_a

    def create_or_update(
        self,
        node_a: BeliefNode,
        node_b: BeliefNode,
        initial_confidence: float,
        signals: dict,
        extractor_version: str
    ) -> TentativeLink:
        """
        Create a new tentative link or update an existing one.

        Args:
            node_a: First belief node
            node_b: Second belief node
            initial_confidence: Initial confidence from resolution
            signals: Reasoning and similarity scores
            extractor_version: Current extractor version

        Returns:
            The created or updated TentativeLink
        """
        from_id, to_id = self._normalize_ids(node_a.belief_id, node_b.belief_id)

        # Check if link exists
        existing = self._get_link(from_id, to_id)

        if existing:
            # Update existing link
            existing.support_both += 1
            existing.last_support_at = datetime.now(timezone.utc)
            existing.updated_at = datetime.now(timezone.utc)

            # Merge signals
            if existing.signals:
                existing.signals['updates'] = existing.signals.get('updates', [])
                existing.signals['updates'].append(signals)
            else:
                existing.signals = {'initial': signals}

            if self.db:
                self.db.add(existing)
                self.db.commit()
                self.db.refresh(existing)

            return existing

        # Create new link
        link = TentativeLink(
            link_id=uuid_module.uuid4(),
            from_belief_id=from_id,
            to_belief_id=to_id,
            confidence=initial_confidence,
            status='pending',
            support_both=1,
            support_one=0,
            last_support_at=datetime.now(timezone.utc),
            signals=signals,
            extractor_version=extractor_version,
        )

        if self.db:
            self.db.add(link)
            self.db.commit()
            self.db.refresh(link)

        return link

    def _get_link(
        self,
        from_id: uuid_module.UUID,
        to_id: uuid_module.UUID
    ) -> Optional[TentativeLink]:
        """Get existing link between two nodes."""
        if not self.db:
            return None

        return self.db.exec(
            select(TentativeLink).where(
                TentativeLink.from_belief_id == from_id,
                TentativeLink.to_belief_id == to_id
            )
        ).first()

    def record_uncertain_match(self, link: TentativeLink) -> None:
        """
        Record an uncertain match involving both nodes in a link.

        Increments support_both and updates last_support_at.
        """
        link.support_both += 1
        link.last_support_at = datetime.now(timezone.utc)
        link.updated_at = datetime.now(timezone.utc)

        if self.db:
            self.db.add(link)
            self.db.commit()

    def record_definite_match_to_one(
        self,
        link: TentativeLink,
        matched_node_id: uuid_module.UUID
    ) -> None:
        """
        Record a definite match to one side of an existing link.

        This indicates evidence that the two nodes are NOT the same
        concept (since a new atom matched definitively to just one).

        Increments support_one.
        """
        link.support_one += 1
        link.updated_at = datetime.now(timezone.utc)

        if self.db:
            self.db.add(link)
            self.db.commit()

    def update_confidence(self, link: TentativeLink) -> TentativeLinkUpdate:
        """
        Update link confidence using decay formula.

        confidence = sigmoid(a * support_both - b * support_one - c * age_days)

        Then check auto_accept/auto_reject thresholds:
        - If confidence >= auto_accept_threshold: status = "accepted"
        - If confidence <= auto_reject_threshold: status = "rejected"

        NOTE: "accepted" means merge is recommended, NOT that merge occurs!

        Returns:
            TentativeLinkUpdate with old/new values and merge_required flag
        """
        old_status = link.status
        old_confidence = link.confidence

        # Compute age in days
        if link.last_support_at:
            age_ref = link.last_support_at
        else:
            age_ref = link.created_at

        now = datetime.now(timezone.utc)
        age_days = (now - age_ref).total_seconds() / 86400

        # Compute confidence using sigmoid
        raw_score = (
            self.a * link.support_both
            - self.b * link.support_one
            - self.c * age_days
        )
        new_confidence = 1.0 / (1.0 + math.exp(-raw_score))

        link.confidence = new_confidence
        link.updated_at = now

        merge_required = False

        # Check thresholds
        if new_confidence >= self.auto_accept_threshold:
            if link.status != 'accepted':
                link.status = 'accepted'
                merge_required = True
                logger.info(
                    f"TentativeLink {link.link_id} auto-accepted "
                    f"(confidence={new_confidence:.3f}). Merge required."
                )
        elif new_confidence <= self.auto_reject_threshold:
            if link.status != 'rejected':
                link.status = 'rejected'
                logger.info(
                    f"TentativeLink {link.link_id} auto-rejected "
                    f"(confidence={new_confidence:.3f})"
                )

        if self.db:
            self.db.add(link)
            self.db.commit()

        return TentativeLinkUpdate(
            link=link,
            old_status=old_status,
            new_status=link.status,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            merge_required=merge_required,
        )

    def get_pending_links_for_node(
        self,
        node_id: uuid_module.UUID
    ) -> List[TentativeLink]:
        """
        Get all pending links involving a specific node.

        Args:
            node_id: The belief node ID

        Returns:
            List of pending TentativeLinks
        """
        if not self.db:
            return []

        # Query links where node is either from or to
        links = self.db.exec(
            select(TentativeLink).where(
                TentativeLink.status == 'pending',
                (TentativeLink.from_belief_id == node_id) |
                (TentativeLink.to_belief_id == node_id)
            )
        ).all()

        return list(links)

    def get_all_pending_links(self) -> List[TentativeLink]:
        """Get all pending tentative links."""
        if not self.db:
            return []

        return list(self.db.exec(
            select(TentativeLink).where(TentativeLink.status == 'pending')
        ).all())

    def get_accepted_links(self) -> List[TentativeLink]:
        """
        Get all accepted links that need merging.

        These are links where status='accepted' but nodes haven't been merged.
        """
        if not self.db:
            return []

        return list(self.db.exec(
            select(TentativeLink).where(TentativeLink.status == 'accepted')
        ).all())
