"""Tests for raw store persistence layer."""

import tempfile
from pathlib import Path

import pytest

from src.memory.models import (
    Actor,
    CaptureMethod,
    ContentModel,
    ExperienceModel,
    ExperienceType,
    ProvenanceModel,
)
from src.memory.raw_store import (
    ImmutabilityViolation,
    RawStore,
    RawStoreError,
    create_raw_store,
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_raw_store.db"
        yield db_path


@pytest.fixture
def raw_store(temp_db):
    """Create RawStore instance for testing."""
    store = RawStore(temp_db)
    yield store
    store.close()


@pytest.fixture
def sample_experience():
    """Create sample experience for testing."""
    return ExperienceModel(
        id="exp_test_001",
        type=ExperienceType.OCCURRENCE,
        content=ContentModel(text="Sample experience for testing"),
        provenance=ProvenanceModel(actor=Actor.USER, method=CaptureMethod.CAPTURE),
    )


class TestRawStoreBasics:
    """Test basic raw store operations."""

    def test_create_store(self, temp_db):
        """Test creating raw store instance."""
        store = RawStore(temp_db)
        assert store.db_path == temp_db
        assert temp_db.parent.exists()
        store.close()

    def test_create_with_factory(self):
        """Test factory function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = create_raw_store(db_path)
            assert isinstance(store, RawStore)
            store.close()

    def test_wal_mode_enabled(self, raw_store):
        """Test that WAL mode is enabled."""
        # This just checks that the store initializes without error
        # Actual WAL check would require inspecting SQLite pragmas
        assert raw_store.engine is not None


class TestAppendExperience:
    """Test appending experiences to raw store."""

    def test_append_single_experience(self, raw_store, sample_experience):
        """Test appending a single experience."""
        exp_id = raw_store.append_experience(sample_experience)
        assert exp_id == sample_experience.id

        # Verify it was stored
        retrieved = raw_store.get_experience(exp_id)
        assert retrieved is not None
        assert retrieved.id == sample_experience.id
        assert retrieved.content.text == sample_experience.content.text

    def test_append_multiple_experiences(self, raw_store):
        """Test appending multiple experiences."""
        experiences = [
            ExperienceModel(
                id=f"exp_test_{i:03d}",
                content=ContentModel(text=f"Experience {i}"),
                provenance=ProvenanceModel(actor=Actor.USER, method=CaptureMethod.CAPTURE),
            )
            for i in range(5)
        ]

        for exp in experiences:
            raw_store.append_experience(exp)

        assert raw_store.count_experiences() == 5

    def test_append_duplicate_id_fails(self, raw_store, sample_experience):
        """Test that appending duplicate ID raises error."""
        raw_store.append_experience(sample_experience)

        with pytest.raises(RawStoreError, match="already exists"):
            raw_store.append_experience(sample_experience)


class TestGetExperience:
    """Test retrieving experiences."""

    def test_get_existing_experience(self, raw_store, sample_experience):
        """Test retrieving existing experience."""
        raw_store.append_experience(sample_experience)
        retrieved = raw_store.get_experience(sample_experience.id)

        assert retrieved is not None
        assert retrieved.id == sample_experience.id
        assert retrieved.type == sample_experience.type

    def test_get_nonexistent_experience(self, raw_store):
        """Test retrieving non-existent experience returns None."""
        retrieved = raw_store.get_experience("exp_does_not_exist")
        assert retrieved is None

    def test_roundtrip_preserves_data(self, raw_store):
        """Test that round-trip preserves all data."""
        exp = ExperienceModel(
            id="exp_roundtrip",
            content=ContentModel(
                text="Roundtrip test",
                media=["image.png"],
                structured={"key": "value"},
            ),
            provenance=ProvenanceModel(
                actor=Actor.AGENT, method=CaptureMethod.MODEL_INFER, sources=[]
            ),
            evidence_ptrs=["exp_001"],
            parents=["exp_000"],
        )

        raw_store.append_experience(exp)
        retrieved = raw_store.get_experience(exp.id)

        assert retrieved is not None
        assert retrieved.content.media == exp.content.media
        assert retrieved.content.structured == exp.content.structured
        assert retrieved.evidence_ptrs == exp.evidence_ptrs
        assert retrieved.parents == exp.parents


class TestListRecent:
    """Test listing recent experiences."""

    def test_list_recent_empty(self, raw_store):
        """Test list_recent on empty store."""
        recent = raw_store.list_recent()
        assert recent == []

    def test_list_recent_returns_newest_first(self, raw_store):
        """Test that list_recent returns most recent first."""
        import time

        for i in range(5):
            exp = ExperienceModel(
                id=f"exp_seq_{i:03d}",
                content=ContentModel(text=f"Experience {i}"),
                provenance=ProvenanceModel(actor=Actor.USER, method=CaptureMethod.CAPTURE),
            )
            raw_store.append_experience(exp)
            time.sleep(0.01)  # Small delay to ensure different timestamps

        recent = raw_store.list_recent(limit=3)
        assert len(recent) == 3
        assert recent[0].id == "exp_seq_004"  # Most recent
        assert recent[1].id == "exp_seq_003"
        assert recent[2].id == "exp_seq_002"

    def test_list_recent_with_type_filter(self, raw_store):
        """Test filtering by experience type."""
        # Add occurrences
        for i in range(3):
            exp = ExperienceModel(
                id=f"exp_occ_{i}",
                type=ExperienceType.OCCURRENCE,
                content=ContentModel(text=f"Occurrence {i}"),
                provenance=ProvenanceModel(actor=Actor.USER, method=CaptureMethod.CAPTURE),
            )
            raw_store.append_experience(exp)

        # Add observations
        for i in range(2):
            exp = ExperienceModel(
                id=f"exp_obs_{i}",
                type=ExperienceType.OBSERVATION,
                content=ContentModel(text=f"Observation {i}"),
                provenance=ProvenanceModel(actor=Actor.AGENT, method=CaptureMethod.MODEL_INFER),
            )
            raw_store.append_experience(exp)

        # Filter by occurrence
        occurrences = raw_store.list_recent(experience_type=ExperienceType.OCCURRENCE)
        assert len(occurrences) == 3
        assert all(e.type == ExperienceType.OCCURRENCE for e in occurrences)

        # Filter by observation
        observations = raw_store.list_recent(experience_type=ExperienceType.OBSERVATION)
        assert len(observations) == 2
        assert all(e.type == ExperienceType.OBSERVATION for e in observations)


class TestAppendObservation:
    """Test observation helper method."""

    def test_append_observation_creates_observation(self, raw_store):
        """Test that append_observation creates observation-type experience."""
        obs_id = raw_store.append_observation(
            content_text="This is a reflection",
            parent_ids=["exp_001", "exp_002"],
        )

        retrieved = raw_store.get_experience(obs_id)
        assert retrieved is not None
        assert retrieved.type == ExperienceType.OBSERVATION
        assert retrieved.content.text == "This is a reflection"
        assert retrieved.parents == ["exp_001", "exp_002"]
        assert retrieved.provenance.actor == Actor.AGENT

    def test_append_observation_with_custom_id(self, raw_store):
        """Test append_observation with custom ID."""
        custom_id = "obs_custom_123"
        obs_id = raw_store.append_observation(
            content_text="Custom ID observation",
            parent_ids=[],
            experience_id=custom_id,
        )

        assert obs_id == custom_id
        retrieved = raw_store.get_experience(custom_id)
        assert retrieved is not None


class TestImmutability:
    """Test immutability constraints."""

    def test_update_raises_error(self, raw_store, sample_experience):
        """Test that update attempts raise ImmutabilityViolation."""
        raw_store.append_experience(sample_experience)

        with pytest.raises(ImmutabilityViolation, match="Cannot update"):
            raw_store.update_experience(sample_experience.id, text="Modified")

    def test_delete_raises_error(self, raw_store, sample_experience):
        """Test that delete attempts raise ImmutabilityViolation."""
        raw_store.append_experience(sample_experience)

        with pytest.raises(ImmutabilityViolation, match="Cannot delete"):
            raw_store.delete_experience(sample_experience.id)

    def test_tombstone_stub(self, raw_store, sample_experience):
        """Test tombstone stub (MVP implementation)."""
        raw_store.append_experience(sample_experience)

        # Tombstone should succeed for existing record
        result = raw_store.tombstone(sample_experience.id, reason="gdpr_erasure")
        assert result is True

        # Should return False for non-existent record
        result = raw_store.tombstone("exp_nonexistent", reason="test")
        assert result is False


class TestCountExperiences:
    """Test experience counting."""

    def test_count_empty_store(self, raw_store):
        """Test counting empty store."""
        assert raw_store.count_experiences() == 0

    def test_count_all_experiences(self, raw_store):
        """Test counting all experiences."""
        for i in range(7):
            exp = ExperienceModel(
                id=f"exp_{i}",
                content=ContentModel(text=f"Experience {i}"),
                provenance=ProvenanceModel(actor=Actor.USER, method=CaptureMethod.CAPTURE),
            )
            raw_store.append_experience(exp)

        assert raw_store.count_experiences() == 7

    def test_count_by_type(self, raw_store):
        """Test counting by type."""
        # Add 3 occurrences
        for i in range(3):
            exp = ExperienceModel(
                id=f"exp_occ_{i}",
                type=ExperienceType.OCCURRENCE,
                content=ContentModel(text="Occurrence"),
                provenance=ProvenanceModel(actor=Actor.USER, method=CaptureMethod.CAPTURE),
            )
            raw_store.append_experience(exp)

        # Add 2 observations
        for i in range(2):
            exp = ExperienceModel(
                id=f"exp_obs_{i}",
                type=ExperienceType.OBSERVATION,
                content=ContentModel(text="Observation"),
                provenance=ProvenanceModel(actor=Actor.AGENT, method=CaptureMethod.MODEL_INFER),
            )
            raw_store.append_experience(exp)

        assert raw_store.count_experiences(ExperienceType.OCCURRENCE) == 3
        assert raw_store.count_experiences(ExperienceType.OBSERVATION) == 2
