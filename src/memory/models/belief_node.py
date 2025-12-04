"""
BeliefNode model - canonical belief concepts.

A BeliefNode represents a unique self-belief concept that may be reinforced
by multiple occurrences (evidence events) over time.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, LargeBinary
from sqlmodel import Field, SQLModel


class BeliefNode(SQLModel, table=True):
    """
    Canonical belief concept.

    Example: "I value honesty" as a unique concept, regardless of how many
    times it's been expressed or in what context.

    Attributes:
        belief_id: Unique identifier for this belief concept
        canonical_text: Normalized text of the belief (e.g., "i value honesty")
        canonical_hash: SHA256 hash of canonical_text for uniqueness/lookup
        belief_type: Ontological category (TRAIT, PREFERENCE, VALUE, etc.)
        polarity: Whether this is an affirmation or denial
        created_at: When this belief was first extracted
        last_reinforced_at: When this belief was last supported by evidence
        activation: Current activation level (recency-weighted)
        core_score: Centrality score based on support, spread, diversity
        status: Lifecycle status (surface, developing, core, orphaned)
        embedding: Serialized numpy array of embedding vector
    """

    __tablename__ = "belief_nodes"

    belief_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for this belief concept"
    )

    canonical_text: str = Field(
        index=True,
        description="Normalized text of the belief"
    )

    canonical_hash: str = Field(
        unique=True,
        index=True,
        description="SHA256 hash of canonical_text for uniqueness"
    )

    belief_type: str = Field(
        description="Ontological category: TRAIT, PREFERENCE, VALUE, CAPABILITY_LIMIT, "
                    "FEELING_STATE, META_BELIEF, RELATIONAL, BELIEF_ABOUT_SELF"
    )

    polarity: str = Field(
        description="Whether belief is affirmed or denied: affirm, deny"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this belief concept was first extracted"
    )

    last_reinforced_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this belief was last supported by evidence"
    )

    activation: float = Field(
        default=0.0,
        description="Current activation level (recency-weighted sum of occurrences)"
    )

    core_score: float = Field(
        default=0.0,
        description="Centrality score based on support, spread, diversity"
    )

    status: str = Field(
        default="surface",
        description="Lifecycle status: surface, developing, core, orphaned, merged_into:<id>"
    )

    # Store embedding as binary blob (serialized numpy array)
    embedding: Optional[bytes] = Field(
        default=None,
        sa_column=Column(LargeBinary),
        description="Serialized numpy array of embedding vector"
    )

    class Config:
        """SQLModel configuration."""
        arbitrary_types_allowed = True


# Valid values for belief_type
BELIEF_TYPES = [
    "TRAIT",
    "PREFERENCE",
    "VALUE",
    "CAPABILITY_LIMIT",
    "FEELING_STATE",
    "META_BELIEF",
    "RELATIONAL",
    "BELIEF_ABOUT_SELF",
]

# Valid values for polarity
POLARITIES = ["affirm", "deny"]

# Valid values for status
STATUSES = ["surface", "developing", "core", "orphaned"]
