"""Data models for experience memory system.

This module defines the core data structures for:
- Experience records (immutable memory entries)
- Signature embeddings (vector representations)
- Affect snapshots (emotional state tracking)

MVP scope focuses on 'occurrence' type experiences with semantic embeddings
and minimal affect tracking (user valence only).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import Column, Field as SQLField, SQLModel
from sqlalchemy import JSON


# Enums


class ExperienceType(str, Enum):
    """Experience type lattice: occurrence ⊑ observation ⊑ inference ⊑ reconciliation."""

    OCCURRENCE = "occurrence"  # MVP focus
    OBSERVATION = "observation"
    INFERENCE = "inference"
    RECONCILIATION = "reconciliation"


class Actor(str, Enum):
    """Actor in the system."""

    USER = "user"
    AGENT = "agent"
    TOOL = "tool"


class CaptureMethod(str, Enum):
    """Method used to capture experience."""

    CAPTURE = "capture"
    SCRAPE = "scrape"
    MODEL_INFER = "model_infer"
    RECONCILE = "reconcile"


class EmbeddingRole(str, Enum):
    """Role of an embedding in the experience signature."""

    PROMPT_SEMANTIC = "prompt_semantic"
    RESPONSE_SEMANTIC = "response_semantic"
    TEMPORAL_PROFILE = "temporal_profile"
    CAUSAL_PROFILE = "causal_profile"
    AFFECT_PROFILE = "affect_profile"


class Stage(str, Enum):
    """Processing stage for affect tracking."""

    INPUT = "input"
    DRAFT = "draft"
    AUGMENTED = "augmented"


class SessionStatus(str, Enum):
    """Status of a session."""

    ACTIVE = "active"
    ENDED = "ended"
    CONSOLIDATED = "consolidated"


# Pydantic models (for validation and API)


class ContentModel(BaseModel):
    """Content of an experience."""

    text: str
    media: list[str] = Field(default_factory=list)
    structured: dict[str, Any] = Field(default_factory=dict)


class ProvenanceSource(BaseModel):
    """Source reference for provenance tracking."""

    uri: str
    hash: Optional[str] = None


class ProvenanceModel(BaseModel):
    """Provenance metadata for experience."""

    sources: list[ProvenanceSource] = Field(default_factory=list)
    actor: Actor
    method: CaptureMethod


class ConfidenceModel(BaseModel):
    """Confidence score with method attribution."""

    p: float = Field(default=0.5, ge=0.0, le=1.0)
    method: str = "default"


class EmbeddingPointers(BaseModel):
    """Pointers to embedding vectors in vector store."""

    semantic: Optional[str] = None  # e.g., "vec://sem/exp_..."
    temporal: Optional[str] = None
    causal: Optional[str] = None
    symbolic: list[dict[str, str]] = Field(default_factory=list)  # Optional symbolic triples


class VAD(BaseModel):
    """Valence-Arousal-Dominance affect representation."""

    v: float = Field(default=0.0, ge=-1.0, le=1.0)  # Valence
    a: float = Field(default=0.0, ge=0.0, le=1.0)  # Arousal
    d: float = Field(default=0.0, ge=-1.0, le=1.0)  # Dominance


class AffectModel(BaseModel):
    """Affect metadata (MVP: minimal, user valence only)."""

    vad: VAD = Field(default_factory=VAD)
    labels: list[str] = Field(default_factory=list)
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExperienceModel(BaseModel):
    """Pydantic model for Experience validation."""

    id: str  # e.g., "exp_2025-10-02T17:01:05Z_9f3c"
    type: ExperienceType = ExperienceType.OCCURRENCE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: ContentModel
    provenance: ProvenanceModel
    evidence_ptrs: list[str] = Field(default_factory=list)
    confidence: ConfidenceModel = Field(default_factory=ConfidenceModel)
    embeddings: EmbeddingPointers = Field(default_factory=EmbeddingPointers)
    affect: AffectModel = Field(default_factory=AffectModel)
    parents: list[str] = Field(default_factory=list)
    sign: Optional[str] = None  # Cryptographic signature
    ownership: Actor = Actor.USER  # Who "owns" this experience's emotional content
    session_id: Optional[str] = None  # Session tracking
    consolidated: bool = False  # True if part of consolidated narrative

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "exp_2025-10-02T17:01:05Z_9f3c",
                "type": "occurrence",
                "created_at": "2025-10-02T17:01:05Z",
                "content": {"text": "User asked about Python imports", "media": []},
                "provenance": {"sources": [], "actor": "user", "method": "capture"},
                "confidence": {"p": 0.9, "method": "user_input"},
            }
        }
    )


class SignatureEmbeddingModel(BaseModel):
    """Pydantic model for signature embedding."""

    id: UUID = Field(default_factory=uuid4)
    experience_id: str
    role: EmbeddingRole
    embedding_model: str
    # vector: stored separately in vector DB
    label: Optional[str] = None
    hash: Optional[str] = None  # LSH for deduplication
    salience: float = Field(default=1.0, ge=0.0, le=1.0)
    emb_metadata: dict[str, Any] = Field(default_factory=dict, alias="metadata")


class AffectSnapshotModel(BaseModel):
    """Pydantic model for affect snapshot."""

    id: UUID = Field(default_factory=uuid4)
    experience_id: str
    actor: Actor
    stage: Stage
    primary_emotion: str = "neutral"
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    dominance: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: Optional[str] = None
    signals: dict[str, Any] = Field(default_factory=dict)


# SQLModel models (for database persistence)


class Experience(SQLModel, table=True):
    """SQLModel for experience table (raw store)."""

    __tablename__ = "experience"

    id: str = SQLField(primary_key=True)
    type: str = SQLField(default=ExperienceType.OCCURRENCE.value)
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc), index=True)
    # Store complex fields as JSON
    content: dict = SQLField(sa_column=Column(JSON))
    provenance: dict = SQLField(sa_column=Column(JSON))
    evidence_ptrs: list[str] = SQLField(sa_column=Column(JSON), default_factory=list)
    confidence: dict = SQLField(
        sa_column=Column(JSON), default_factory=lambda: {"p": 0.5, "method": "default"}
    )
    embeddings: dict = SQLField(sa_column=Column(JSON), default_factory=dict)
    affect: dict = SQLField(
        sa_column=Column(JSON),
        default_factory=lambda: {
            "vad": {"v": 0.0, "a": 0.0, "d": 0.0},
            "labels": [],
            "intensity": 0.0,
            "confidence": 0.0,
        },
    )
    parents: list[str] = SQLField(sa_column=Column(JSON), default_factory=list)
    sign: Optional[str] = SQLField(default=None)
    ownership: str = SQLField(default=Actor.USER.value)  # Who "owns" this experience's emotional content
    session_id: Optional[str] = SQLField(default=None, index=True)  # Session tracking
    consolidated: bool = SQLField(default=False)  # True if part of consolidated narrative


class SignatureEmbedding(SQLModel, table=True):
    """SQLModel for signature_embedding table."""

    __tablename__ = "signature_embedding"

    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    experience_id: str = SQLField(foreign_key="experience.id", index=True)
    role: str
    embedding_model: str
    label: Optional[str] = None
    hash: Optional[str] = None
    salience: float = 1.0
    emb_metadata: dict = SQLField(sa_column=Column(JSON), default_factory=dict)


class AffectSnapshot(SQLModel, table=True):
    """SQLModel for affect_snapshot table."""

    __tablename__ = "affect_snapshot"

    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    experience_id: str = SQLField(foreign_key="experience.id", index=True)
    actor: str
    stage: str
    primary_emotion: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    confidence: float = 0.0
    notes: Optional[str] = None
    signals: dict = SQLField(sa_column=Column(JSON), default_factory=dict)


class Session(SQLModel, table=True):
    """SQLModel for session table (tracking conversation sessions)."""

    __tablename__ = "session"

    id: str = SQLField(primary_key=True)
    user_id: str = SQLField(default="default_user", index=True)
    start_time: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc), index=True)
    end_time: Optional[datetime] = SQLField(default=None)
    last_activity: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))
    status: str = SQLField(default=SessionStatus.ACTIVE.value)
    experience_ids: list[str] = SQLField(sa_column=Column(JSON), default_factory=list)
    consolidated_narrative_id: Optional[str] = SQLField(default=None)
    session_metadata: dict = SQLField(sa_column=Column(JSON), default_factory=dict)


class MemoryDecayMetrics(SQLModel, table=True):
    """SQLModel for memory_decay_metrics table (tracking decay factors)."""

    __tablename__ = "memory_decay_metrics"

    experience_id: str = SQLField(primary_key=True, foreign_key="experience.id")
    access_count: int = SQLField(default=0)
    last_accessed: Optional[datetime] = SQLField(default=None)
    entropy_score: float = SQLField(default=0.5)  # 0-1, higher = more unique
    emotional_salience: float = SQLField(default=0.0)  # abs(valence)
    decay_factor: float = SQLField(default=1.0)  # 1.0 = no decay, 0.0 = fully decayed
    last_calculated: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))


# Serialization helpers


def experience_to_model(exp: Experience) -> ExperienceModel:
    """Convert SQLModel Experience to Pydantic ExperienceModel."""
    # SQLite doesn't preserve timezone info, so ensure created_at is UTC-aware
    created_at = exp.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    return ExperienceModel(
        id=exp.id,
        type=ExperienceType(exp.type),
        created_at=created_at,
        content=ContentModel(**exp.content),
        provenance=ProvenanceModel(**exp.provenance),
        evidence_ptrs=exp.evidence_ptrs,
        confidence=ConfidenceModel(**exp.confidence),
        embeddings=EmbeddingPointers(**exp.embeddings),
        affect=AffectModel(**exp.affect),
        parents=exp.parents,
        sign=exp.sign,
        ownership=Actor(exp.ownership),
        session_id=exp.session_id,
        consolidated=exp.consolidated,
    )


def model_to_experience(model: ExperienceModel) -> Experience:
    """Convert Pydantic ExperienceModel to SQLModel Experience."""
    return Experience(
        id=model.id,
        type=model.type.value,
        created_at=model.created_at,
        content=model.content.model_dump(),
        provenance=model.provenance.model_dump(),
        evidence_ptrs=model.evidence_ptrs,
        confidence=model.confidence.model_dump(),
        embeddings=model.embeddings.model_dump(),
        affect=model.affect.model_dump(),
        parents=model.parents,
        sign=model.sign,
        ownership=model.ownership.value,
        session_id=model.session_id,
        consolidated=model.consolidated,
    )


def signature_embedding_to_model(emb: SignatureEmbedding) -> SignatureEmbeddingModel:
    """Convert SQLModel SignatureEmbedding to Pydantic model."""
    return SignatureEmbeddingModel(
        id=emb.id,
        experience_id=emb.experience_id,
        role=EmbeddingRole(emb.role),
        embedding_model=emb.embedding_model,
        label=emb.label,
        hash=emb.hash,
        salience=emb.salience,
        emb_metadata=emb.emb_metadata,
    )


def model_to_signature_embedding(model: SignatureEmbeddingModel) -> SignatureEmbedding:
    """Convert Pydantic model to SQLModel SignatureEmbedding."""
    return SignatureEmbedding(
        id=model.id,
        experience_id=model.experience_id,
        role=model.role.value,
        embedding_model=model.embedding_model,
        label=model.label,
        hash=model.hash,
        salience=model.salience,
        emb_metadata=model.emb_metadata,
    )


def affect_snapshot_to_model(snap: AffectSnapshot) -> AffectSnapshotModel:
    """Convert SQLModel AffectSnapshot to Pydantic model."""
    return AffectSnapshotModel(
        id=snap.id,
        experience_id=snap.experience_id,
        actor=Actor(snap.actor),
        stage=Stage(snap.stage),
        primary_emotion=snap.primary_emotion,
        valence=snap.valence,
        arousal=snap.arousal,
        dominance=snap.dominance,
        confidence=snap.confidence,
        notes=snap.notes,
        signals=snap.signals,
    )


def model_to_affect_snapshot(model: AffectSnapshotModel) -> AffectSnapshot:
    """Convert Pydantic model to SQLModel AffectSnapshot."""
    return AffectSnapshot(
        id=model.id,
        experience_id=model.experience_id,
        actor=model.actor.value,
        stage=model.stage.value,
        primary_emotion=model.primary_emotion,
        valence=model.valence,
        arousal=model.arousal,
        dominance=model.dominance,
        confidence=model.confidence,
        notes=model.notes,
        signals=model.signals,
    )
