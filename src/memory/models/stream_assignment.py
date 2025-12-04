"""
StreamAssignment model - soft stream assignments with migration tracking.

Beliefs are assigned to streams (identity, state, meta, relational) based on
their type and temporal scope. Over time, state beliefs may migrate to identity
if they show sustained patterns (spread + diversity thresholds).

Migration is "sticky" via the ratchet mechanism - once migrated to identity,
beliefs don't demote back to state without explicit triggers.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class StreamAssignment(SQLModel, table=True):
    """
    Soft stream assignment for a BeliefNode.

    Streams:
    - identity: Core self-concept (who I am)
    - state: Current emotional/mental state (how I feel)
    - meta: Beliefs about beliefs (what I think about myself)
    - relational: Relationships and social patterns

    Migration tracking:
    - STATE → IDENTITY: When spread and diversity thresholds are met
    - migrated_from tracks the original stream for audit

    Attributes:
        belief_id: Reference to the BeliefNode (also primary key)
        primary_stream: Main stream assignment
        secondary_stream: Optional secondary stream
        confidence: Confidence in stream assignment
        migrated_from: Original stream if migrated (audit trail)
        updated_at: When assignment was last modified
    """

    __tablename__ = "stream_assignments"

    belief_id: uuid.UUID = Field(
        foreign_key="belief_nodes.belief_id",
        primary_key=True,
        description="Reference to the BeliefNode"
    )

    primary_stream: str = Field(
        description="Primary stream: identity, state, meta, relational"
    )

    secondary_stream: Optional[str] = Field(
        default=None,
        description="Optional secondary stream"
    )

    confidence: float = Field(
        description="Confidence in stream assignment"
    )

    migrated_from: Optional[str] = Field(
        default=None,
        description="Original stream if migrated (audit trail)"
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When assignment was last modified"
    )

    class Config:
        """SQLModel configuration."""
        arbitrary_types_allowed = True


# Valid stream types
STREAM_TYPES = ["identity", "state", "meta", "relational"]
