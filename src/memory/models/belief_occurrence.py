"""
BeliefOccurrence model - evidence events linking beliefs to source experiences.

Each occurrence represents a single extraction event where a belief was found
in a source experience. The same belief can have multiple occurrences from
different experiences or even multiple occurrences from the same experience
if extracted by different extractor versions.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel


class BeliefOccurrence(SQLModel, table=True):
    """
    Evidence event tying a BeliefNode to a source Experience.

    The unique constraint on (belief_id, source_experience_id, extractor_version)
    ensures we only store one occurrence per belief per source per extractor,
    allowing re-extraction with new extractor versions without duplicates.

    Attributes:
        occurrence_id: Unique identifier for this occurrence
        belief_id: Reference to the BeliefNode this occurrence supports
        source_experience_id: ID of the Experience this was extracted from
        extractor_version: Version hash of the extractor that created this
        raw_text: Original text as extracted (before canonicalization)
        raw_span: Character spans in source text, as {start, end} or list
        source_weight: Weight based on source context (mode, arousal, etc.)
        atom_confidence: LLM confidence in the extraction
        epistemic_frame: Temporal/modal qualifiers (JSON)
        epistemic_confidence: Confidence in epistemic frame extraction
        match_confidence: Confidence in resolving to this belief node
        context_id: Conversation context identifier
        created_at: When this occurrence was created
        deleted_at: Soft delete timestamp for rollback support
    """

    __tablename__ = "belief_occurrences"
    __table_args__ = (
        UniqueConstraint(
            "belief_id",
            "source_experience_id",
            "extractor_version",
            name="uq_belief_occurrence_source_version"
        ),
    )

    occurrence_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for this occurrence"
    )

    belief_id: uuid.UUID = Field(
        foreign_key="belief_nodes.belief_id",
        index=True,
        description="Reference to the BeliefNode this occurrence supports"
    )

    source_experience_id: str = Field(
        index=True,
        description="ID of the Experience this was extracted from"
    )

    extractor_version: str = Field(
        index=True,
        description="Version hash of the extractor that created this"
    )

    raw_text: str = Field(
        description="Original text as extracted (before canonicalization)"
    )

    # JSON field: {start: int, end: int} or [{start: int, end: int}, ...]
    raw_span: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Character spans in source text"
    )

    source_weight: float = Field(
        description="Weight based on source context (mode, arousal, etc.)"
    )

    atom_confidence: float = Field(
        description="LLM confidence in the extraction"
    )

    # JSON field: {temporal_scope, modality, degree, conditional}
    epistemic_frame: dict = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Temporal/modal qualifiers"
    )

    epistemic_confidence: float = Field(
        description="Confidence in epistemic frame extraction"
    )

    match_confidence: float = Field(
        description="Confidence in resolving to this belief node"
    )

    context_id: str = Field(
        index=True,
        description="Conversation context identifier"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this occurrence was created"
    )

    deleted_at: Optional[datetime] = Field(
        default=None,
        description="Soft delete timestamp for rollback support"
    )

    class Config:
        """SQLModel configuration."""
        arbitrary_types_allowed = True


# Valid temporal_scope values (NOT "current" - use "state")
TEMPORAL_SCOPES = ["state", "ongoing", "habitual", "transitional", "past", "unknown"]

# Valid modality values
MODALITIES = ["certain", "likely", "possible", "unsure"]
