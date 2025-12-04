"""
TentativeLink model - uncertain identity resolution between belief nodes.

When the resolver is uncertain whether two belief expressions refer to the
same concept, it creates a TentativeLink instead of merging. Links accumulate
evidence over time and may be auto-accepted (marking for future merge) or
auto-rejected based on confidence thresholds.

IMPORTANT: Auto-accept sets status="accepted" but does NOT auto-merge nodes.
Merge is a separate future operation that requires explicit invocation.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel


class TentativeLink(SQLModel, table=True):
    """
    Uncertain identity resolution between two BeliefNodes.

    Links track evidence for and against merging two beliefs:
    - support_both: Count of uncertain matches involving both nodes
    - support_one: Count of definite matches to one side only

    Confidence is computed as:
        sigmoid(a * support_both - b * support_one - c * age_days)

    IMPORTANT: Normalized ordering is enforced - from_belief_id < to_belief_id
    (by string comparison on UUID). The service layer must swap IDs if needed
    before insert/query.

    Attributes:
        link_id: Unique identifier for this link
        from_belief_id: First belief node (smaller UUID string)
        to_belief_id: Second belief node (larger UUID string)
        confidence: Current merge confidence [0, 1]
        status: pending, accepted, rejected
        support_both: Count of uncertain matches involving both
        support_one: Count of definite matches to one side
        last_support_at: When evidence was last added
        signals: JSON object with reasoning, similarity scores
        extractor_version: Version hash when link was created
        created_at: When this link was created
        updated_at: When this link was last modified
    """

    __tablename__ = "tentative_links"
    __table_args__ = (
        UniqueConstraint(
            "from_belief_id",
            "to_belief_id",
            name="uq_tentative_link_pair"
        ),
    )

    link_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for this link"
    )

    from_belief_id: uuid.UUID = Field(
        foreign_key="belief_nodes.belief_id",
        index=True,
        description="First belief node (smaller UUID by string comparison)"
    )

    to_belief_id: uuid.UUID = Field(
        foreign_key="belief_nodes.belief_id",
        index=True,
        description="Second belief node (larger UUID by string comparison)"
    )

    confidence: float = Field(
        description="Current merge confidence [0, 1]"
    )

    status: str = Field(
        default="pending",
        description="Link status: pending, accepted, rejected. "
                    "accepted means merge is recommended (NOT auto-merged)"
    )

    support_both: int = Field(
        default=0,
        description="Count of uncertain matches involving both nodes"
    )

    support_one: int = Field(
        default=0,
        description="Count of definite matches to one side only"
    )

    last_support_at: Optional[datetime] = Field(
        default=None,
        description="When evidence was last added"
    )

    # JSON field: {reasoning: str, similarity_scores: dict, ...}
    signals: dict = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Reasoning, similarity scores, and other signals"
    )

    extractor_version: str = Field(
        description="Version hash when link was created"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this link was created"
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this link was last modified"
    )

    class Config:
        """SQLModel configuration."""
        arbitrary_types_allowed = True


# Valid status values
LINK_STATUSES = ["pending", "accepted", "rejected"]
