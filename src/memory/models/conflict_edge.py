"""
ConflictEdge model - contradiction/tension relationships between beliefs.

Conflicts are detected when two beliefs have:
- Contradiction: Same concept with opposite polarity (hard conflict)
- Tension: High semantic similarity with opposite polarity (soft conflict)

Temporal scope is considered: "I used to love X" does not contradict "I hate X"
if the former has temporal_scope="past" and latter has ongoing/habitual/state.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel


class ConflictEdge(SQLModel, table=True):
    """
    Contradiction or tension relationship between two BeliefNodes.

    Types:
    - contradiction: Same base concept, opposite polarity
    - tension: High semantic similarity, opposite polarity, different concepts

    Status:
    - unresolved: Active conflict needing attention
    - tolerated: Known conflict that coexists (cognitive flexibility)
    - resolved: Conflict resolved through belief update

    Attributes:
        edge_id: Unique identifier for this conflict edge
        from_belief_id: First belief node in the conflict
        to_belief_id: Second belief node in the conflict
        conflict_type: contradiction or tension
        status: unresolved, tolerated, resolved
        evidence_occurrence_ids: UUIDs of occurrences that evidence the conflict
        created_at: When this conflict was detected
        updated_at: When this conflict was last modified
    """

    __tablename__ = "conflict_edges"
    __table_args__ = (
        UniqueConstraint(
            "from_belief_id",
            "to_belief_id",
            "conflict_type",
            name="uq_conflict_edge_pair_type"
        ),
    )

    edge_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for this conflict edge"
    )

    from_belief_id: uuid.UUID = Field(
        foreign_key="belief_nodes.belief_id",
        index=True,
        description="First belief node in the conflict"
    )

    to_belief_id: uuid.UUID = Field(
        foreign_key="belief_nodes.belief_id",
        index=True,
        description="Second belief node in the conflict"
    )

    conflict_type: str = Field(
        description="Type of conflict: contradiction, tension"
    )

    status: str = Field(
        default="tolerated",
        description="Conflict status: unresolved, tolerated, resolved"
    )

    # JSON field: list of occurrence UUIDs as strings
    evidence_occurrence_ids: list = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="UUIDs of occurrences that evidence the conflict"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this conflict was detected"
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this conflict was last modified"
    )

    class Config:
        """SQLModel configuration."""
        arbitrary_types_allowed = True


# Valid conflict types
CONFLICT_TYPES = ["contradiction", "tension"]

# Valid conflict statuses
CONFLICT_STATUSES = ["unresolved", "tolerated", "resolved"]
