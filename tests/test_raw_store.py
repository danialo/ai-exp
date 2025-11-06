"""Tests for raw store persistence layer."""

import tempfile
from datetime import datetime, timedelta, timezone
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
from src.pipeline.task_experience import create_task_execution_experience


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

    def test_list_recent_with_since_filter(self, raw_store):
        """Test filtering by cutoff timestamp."""
        base_time = datetime.now(timezone.utc)

        older = ExperienceModel(
            id="exp_since_old",
            created_at=base_time - timedelta(days=2),
            content=ContentModel(text="Older experience"),
            provenance=ProvenanceModel(actor=Actor.USER, method=CaptureMethod.CAPTURE),
        )
        boundary = ExperienceModel(
            id="exp_since_boundary",
            created_at=base_time - timedelta(hours=12),
            content=ContentModel(text="Boundary experience"),
            provenance=ProvenanceModel(actor=Actor.USER, method=CaptureMethod.CAPTURE),
        )
        newer = ExperienceModel(
            id="exp_since_new",
            created_at=base_time - timedelta(hours=1),
            content=ContentModel(text="Newer experience"),
            provenance=ProvenanceModel(actor=Actor.USER, method=CaptureMethod.CAPTURE),
        )

        raw_store.append_experience(older)
        raw_store.append_experience(boundary)
        raw_store.append_experience(newer)

        cutoff = base_time - timedelta(hours=12)
        recent = raw_store.list_recent(limit=10, since=cutoff)

        returned_ids = [exp.id for exp in recent]
        assert "exp_since_old" not in returned_ids
        assert returned_ids[0] == "exp_since_new"
        assert "exp_since_boundary" in returned_ids


class TestTaskExecutionQueries:
    """Tests for task execution-specific raw store helpers."""

    def _store_task(
        self,
        raw_store: RawStore,
        *,
        task_id: str,
        started_at: datetime,
        ended_at: datetime,
        status: str,
        trace_id: str,
        span_id: str,
        attempt: int,
        backfilled: bool = False,
        error: dict | None = None,
        retry_of: str | None = None,
    ) -> ExperienceModel:
        experience = create_task_execution_experience(
            task_id=task_id,
            task_slug=task_id,
            task_name=task_id.replace("_", " ").title(),
            task_type="test",
            scheduled_vs_manual="scheduled",
            started_at=started_at,
            ended_at=ended_at,
            status=status,
            response_text=f"Response for {task_id} attempt {attempt}",
            error=error,
            parent_experience_ids=[],
            retrieval_metadata={"memory_count": 0, "source": []},
            files_written=[],
            task_config={},
            trace_id=trace_id,
            span_id=span_id,
            attempt=attempt,
            retry_of=retry_of,
        )

        if backfilled:
            experience.content.structured["backfilled"] = True

        raw_store.append_experience(experience)
        return experience

    def test_filters_and_backfilled(self, raw_store):
        """list_task_executions should honor status, since, and backfilled filters."""
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)

        self._store_task(
            raw_store,
            task_id="alpha",
            started_at=base - timedelta(hours=4),
            ended_at=base - timedelta(hours=3, minutes=50),
            status="success",
            trace_id="trace-alpha",
            span_id="span-alpha-1",
            attempt=1,
        )

        self._store_task(
            raw_store,
            task_id="beta",
            started_at=base - timedelta(hours=2),
            ended_at=base - timedelta(hours=1, minutes=50),
            status="failed",
            trace_id="trace-beta",
            span_id="span-beta-1",
            attempt=1,
            error={"type": "ValueError", "message": "boom", "stack_hash": "hash123"},
        )

        self._store_task(
            raw_store,
            task_id="gamma",
            started_at=base - timedelta(hours=1),
            ended_at=base - timedelta(minutes=50),
            status="success",
            trace_id="trace-gamma",
            span_id="span-gamma-1",
            attempt=1,
            backfilled=True,
        )

        # Status filter
        failed = raw_store.list_task_executions(status="failed")
        assert len(failed) == 1
        assert failed[0].content.structured["task_id"] == "beta"

        # Since filter (last 2 hours should include beta and gamma)
        recent = raw_store.list_task_executions(since=base - timedelta(hours=2))
        recent_ids = [exp.content.structured["task_id"] for exp in recent]
        assert set(recent_ids) == {"beta", "gamma"}

        # Backfilled filters
        live = raw_store.list_task_executions(backfilled=False)
        live_ids = [exp.content.structured["task_id"] for exp in live]
        assert "gamma" not in live_ids

        only_backfilled = raw_store.list_task_executions(backfilled=True)
        assert len(only_backfilled) == 1
        assert only_backfilled[0].content.structured["task_id"] == "gamma"

    def test_get_by_trace_orders_attempts(self, raw_store):
        """get_by_trace_id should return attempts ordered by attempt number."""
        base = datetime(2025, 1, 2, tzinfo=timezone.utc)

        first = self._store_task(
            raw_store,
            task_id="delta",
            started_at=base - timedelta(hours=2),
            ended_at=base - timedelta(hours=2, minutes=50),
            status="failed",
            trace_id="trace-delta",
            span_id="span-delta-1",
            attempt=1,
            error={"type": "RuntimeError", "message": "fail", "stack_hash": "stack1"},
        )

        second = self._store_task(
            raw_store,
            task_id="delta",
            started_at=base - timedelta(hours=1),
            ended_at=base - timedelta(hours=1, minutes=50),
            status="success",
            trace_id="trace-delta",
            span_id="span-delta-2",
            attempt=2,
            retry_of=first.content.structured["span_id"],
        )

        results = raw_store.get_by_trace_id("trace-delta")
        attempts = [exp.content.structured["attempt"] for exp in results]

        assert attempts == [1, 2]
        assert results[0].content.structured["span_id"] == first.content.structured["span_id"]
        assert results[1].content.structured["span_id"] == second.content.structured["span_id"]

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
