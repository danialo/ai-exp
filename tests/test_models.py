"""Tests for data models."""

from datetime import datetime, timezone
from uuid import UUID

from src.memory.models import (
    Actor,
    AffectModel,
    AffectSnapshot,
    AffectSnapshotModel,
    CaptureMethod,
    ConfidenceModel,
    ContentModel,
    EmbeddingPointers,
    EmbeddingRole,
    Experience,
    ExperienceModel,
    ExperienceType,
    ProvenanceModel,
    ProvenanceSource,
    SignatureEmbedding,
    SignatureEmbeddingModel,
    Stage,
    VAD,
    affect_snapshot_to_model,
    experience_to_model,
    model_to_affect_snapshot,
    model_to_experience,
    model_to_signature_embedding,
    signature_embedding_to_model,
)


class TestExperienceModel:
    """Test ExperienceModel validation and defaults."""

    def test_minimal_experience(self):
        """Test creating minimal experience with defaults."""
        exp = ExperienceModel(
            id="exp_test_001",
            content=ContentModel(text="Test experience"),
            provenance=ProvenanceModel(actor=Actor.USER, method=CaptureMethod.CAPTURE),
        )

        assert exp.id == "exp_test_001"
        assert exp.type == ExperienceType.OCCURRENCE
        assert exp.content.text == "Test experience"
        assert exp.confidence.p == 0.5  # Default
        assert exp.affect.vad.v == 0.0  # Default valence

    def test_full_experience(self):
        """Test creating experience with all fields populated."""
        exp = ExperienceModel(
            id="exp_2025-10-02T17:01:05Z_9f3c",
            type=ExperienceType.OCCURRENCE,
            created_at=datetime(2025, 10, 2, 17, 1, 5, tzinfo=timezone.utc),
            content=ContentModel(
                text="User asked about Python imports",
                media=[],
                structured={"query_type": "technical"},
            ),
            provenance=ProvenanceModel(
                sources=[ProvenanceSource(uri="https://example.com", hash="sha256:abc123")],
                actor=Actor.USER,
                method=CaptureMethod.CAPTURE,
            ),
            evidence_ptrs=["exp_001", "uri:https://example.com"],
            confidence=ConfidenceModel(p=0.78, method="calibrated_logit"),
            embeddings=EmbeddingPointers(
                semantic="vec://sem/exp_2025-10-02T17:01:05Z_9f3c",
                temporal="vec://temp/exp_2025-10-02T17:01:05Z_9f3c",
                symbolic=[{"subject": "X", "rel": "causes", "object": "Y"}],
            ),
            affect=AffectModel(
                vad=VAD(v=0.15, a=0.62, d=0.40),
                labels=["frustration"],
                intensity=0.7,
                confidence=0.66,
            ),
            parents=["exp_000"],
            sign="ed25519:signature_here",
        )

        assert exp.type == ExperienceType.OCCURRENCE
        assert exp.content.structured["query_type"] == "technical"
        assert exp.confidence.p == 0.78
        assert exp.affect.vad.v == 0.15
        assert len(exp.evidence_ptrs) == 2

    def test_vad_validation(self):
        """Test VAD value constraints."""
        # Valid values
        vad = VAD(v=0.5, a=0.5, d=0.5)
        assert vad.v == 0.5

        # Test boundaries
        vad_min = VAD(v=-1.0, a=0.0, d=-1.0)
        assert vad_min.v == -1.0

        vad_max = VAD(v=1.0, a=1.0, d=1.0)
        assert vad_max.a == 1.0

    def test_confidence_validation(self):
        """Test confidence probability constraints."""
        conf = ConfidenceModel(p=0.75)
        assert conf.p == 0.75

        # Default method
        assert conf.method == "default"


class TestSignatureEmbeddingModel:
    """Test SignatureEmbeddingModel."""

    def test_create_embedding(self):
        """Test creating signature embedding."""
        emb = SignatureEmbeddingModel(
            experience_id="exp_001",
            role=EmbeddingRole.PROMPT_SEMANTIC,
            embedding_model="all-MiniLM-L6-v2",
            salience=0.8,
            emb_metadata={"entity": "Python", "domain": "programming"},
        )

        assert emb.experience_id == "exp_001"
        assert emb.role == EmbeddingRole.PROMPT_SEMANTIC
        assert emb.salience == 0.8
        assert isinstance(emb.id, UUID)


class TestAffectSnapshotModel:
    """Test AffectSnapshotModel."""

    def test_create_snapshot(self):
        """Test creating affect snapshot."""
        snapshot = AffectSnapshotModel(
            experience_id="exp_001",
            actor=Actor.USER,
            stage=Stage.INPUT,
            primary_emotion="joy",
            valence=0.8,
            arousal=0.6,
            dominance=0.5,
            confidence=0.75,
            notes="User seems excited",
            signals={"punctuation": "!!!", "caps_ratio": 0.2},
        )

        assert snapshot.actor == Actor.USER
        assert snapshot.stage == Stage.INPUT
        assert snapshot.valence == 0.8
        assert snapshot.signals["punctuation"] == "!!!"


class TestSerialization:
    """Test round-trip serialization between Pydantic and SQLModel."""

    def test_experience_roundtrip(self):
        """Test Experience -> ExperienceModel -> Experience."""
        # Create Pydantic model
        model = ExperienceModel(
            id="exp_roundtrip_001",
            content=ContentModel(text="Roundtrip test", media=["image.png"]),
            provenance=ProvenanceModel(actor=Actor.AGENT, method=CaptureMethod.MODEL_INFER),
            confidence=ConfidenceModel(p=0.9, method="neural"),
            affect=AffectModel(vad=VAD(v=0.5, a=0.3, d=0.7)),
        )

        # Convert to SQLModel
        sql_exp = model_to_experience(model)
        assert sql_exp.id == "exp_roundtrip_001"
        assert sql_exp.type == ExperienceType.OCCURRENCE.value
        assert sql_exp.content["text"] == "Roundtrip test"
        assert sql_exp.confidence["p"] == 0.9

        # Convert back to Pydantic
        model_back = experience_to_model(sql_exp)
        assert model_back.id == model.id
        assert model_back.content.text == model.content.text
        assert model_back.content.media == model.content.media
        assert model_back.confidence.p == model.confidence.p
        assert model_back.affect.vad.v == model.affect.vad.v

    def test_signature_embedding_roundtrip(self):
        """Test SignatureEmbedding roundtrip."""
        model = SignatureEmbeddingModel(
            experience_id="exp_001",
            role=EmbeddingRole.RESPONSE_SEMANTIC,
            embedding_model="sentence-transformers",
            label="technical_response",
            salience=0.95,
        )

        sql_emb = model_to_signature_embedding(model)
        assert sql_emb.role == EmbeddingRole.RESPONSE_SEMANTIC.value

        model_back = signature_embedding_to_model(sql_emb)
        assert model_back.experience_id == model.experience_id
        assert model_back.role == model.role
        assert model_back.salience == model.salience

    def test_affect_snapshot_roundtrip(self):
        """Test AffectSnapshot roundtrip."""
        model = AffectSnapshotModel(
            experience_id="exp_001",
            actor=Actor.USER,
            stage=Stage.AUGMENTED,
            primary_emotion="satisfaction",
            valence=0.7,
            confidence=0.8,
        )

        sql_snap = model_to_affect_snapshot(model)
        assert sql_snap.actor == Actor.USER.value
        assert sql_snap.valence == 0.7

        model_back = affect_snapshot_to_model(sql_snap)
        assert model_back.experience_id == model.experience_id
        assert model_back.primary_emotion == model.primary_emotion
        assert model_back.valence == model.valence


class TestSQLModels:
    """Test SQLModel table definitions."""

    def test_experience_table_creation(self):
        """Test Experience SQL model can be instantiated."""
        exp = Experience(
            id="exp_sql_001",
            type=ExperienceType.OCCURRENCE.value,
            content={"text": "SQL test"},
            provenance={"actor": "user", "method": "capture", "sources": []},
        )

        assert exp.id == "exp_sql_001"
        assert exp.confidence["p"] == 0.5  # Default from factory

    def test_signature_embedding_table_creation(self):
        """Test SignatureEmbedding SQL model."""
        emb = SignatureEmbedding(
            experience_id="exp_001",
            role=EmbeddingRole.PROMPT_SEMANTIC.value,
            embedding_model="test-model",
        )

        assert isinstance(emb.id, UUID)
        assert emb.salience == 1.0  # Default

    def test_affect_snapshot_table_creation(self):
        """Test AffectSnapshot SQL model."""
        snapshot = AffectSnapshot(
            experience_id="exp_001",
            actor=Actor.USER.value,
            stage=Stage.INPUT.value,
        )

        assert snapshot.primary_emotion == "neutral"  # Default
        assert snapshot.valence == 0.0  # Default
