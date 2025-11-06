"""Raw store persistence layer for immutable experience records.

The raw store provides append-only storage for experiences with:
- WAL (Write-Ahead Logging) mode for better concurrency
- Immutability guarantees (no updates/deletes except tombstones)
- Simple query interface (by ID, recent experiences)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, event
from sqlmodel import Session, SQLModel, select

from src.memory.models import (
    Experience,
    ExperienceModel,
    ExperienceType,
    experience_to_model,
    model_to_experience,
)


class RawStoreError(Exception):
    """Base exception for raw store errors."""

    pass


class ImmutabilityViolation(RawStoreError):
    """Raised when attempting to modify immutable records."""

    pass


class RawStore:
    """Repository for immutable experience storage.

    Provides append-only interface to the experience raw store with
    automatic schema management and WAL mode for concurrency.
    """

    def __init__(self, db_path: str | Path, enable_wal: bool = True):
        """Initialize raw store connection.

        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable WAL mode for better concurrency (default True)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine with check_same_thread=False for SQLite
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})

        # Enable WAL mode
        if enable_wal:

            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()

        # Create all tables
        SQLModel.metadata.create_all(self.engine)

        # Create task execution indexes
        self._create_task_indexes()

    def _create_task_indexes(self):
        """Create SQLite indexes for efficient task execution queries.

        These indexes use json_extract to index fields within the structured content,
        enabling fast queries by task_id, task_slug, trace_id, and idempotency_key.
        """
        with Session(self.engine) as session:
            from sqlalchemy import text

            # Index for querying by experience type and creation time
            session.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_experiences_type_ts
                    ON experience(type, created_at DESC)
                    """
                )
            )

            # Index for querying by task_id (extracted from JSON)
            session.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_experiences_task
                    ON experience(json_extract(content, '$.structured.task_id'), created_at DESC)
                    """
                )
            )

            # Index for querying by task_slug (extracted from JSON)
            session.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_experiences_task_slug
                    ON experience(json_extract(content, '$.structured.task_slug'), created_at DESC)
                    """
                )
            )

            # Index for querying by trace_id (for correlation)
            session.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_experiences_trace
                    ON experience(json_extract(content, '$.structured.trace_id'))
                    """
                )
            )

            # Index for idempotency key lookup
            session.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_experiences_idempotency
                    ON experience(json_extract(content, '$.structured.idempotency_key'))
                    """
                )
            )

            session.commit()

    def append_experience(self, experience: ExperienceModel) -> str:
        """Append new experience to raw store.

        Args:
            experience: Experience model to persist

        Returns:
            Experience ID

        Raises:
            RawStoreError: If experience with same ID already exists
        """
        with Session(self.engine) as session:
            # Check if experience already exists
            existing = session.get(Experience, experience.id)
            if existing:
                raise RawStoreError(f"Experience {experience.id} already exists (immutable store)")

            # Convert to SQL model and add
            sql_exp = model_to_experience(experience)
            session.add(sql_exp)
            session.commit()
            session.refresh(sql_exp)

            return sql_exp.id

    def get_experience(self, experience_id: str) -> Optional[ExperienceModel]:
        """Retrieve experience by ID.

        Args:
            experience_id: Experience ID to retrieve

        Returns:
            ExperienceModel if found, None otherwise
        """
        with Session(self.engine) as session:
            exp = session.get(Experience, experience_id)
            if exp is None:
                return None
            return experience_to_model(exp)

    def list_recent(
        self,
        limit: int = 10,
        experience_type: Optional[ExperienceType] = None,
        since: Optional[datetime] = None,
    ) -> list[ExperienceModel]:
        """List recent experiences ordered by creation time.

        Args:
            limit: Maximum number of experiences to return
            experience_type: Filter by experience type (optional)
            since: Only include experiences created at or after this timestamp

        Returns:
            List of ExperienceModels, most recent first
        """
        with Session(self.engine) as session:
            statement = select(Experience).order_by(Experience.created_at.desc()).limit(limit)

            if experience_type:
                statement = statement.where(Experience.type == experience_type.value)

            if since is not None:
                if since.tzinfo is None:
                    since = since.replace(tzinfo=timezone.utc)
                statement = statement.where(Experience.created_at >= since)

            experiences = session.exec(statement).all()
            return [experience_to_model(exp) for exp in experiences]

    def append_observation(
        self,
        content_text: str,
        parent_ids: list[str],
        experience_id: Optional[str] = None,
    ) -> str:
        """Helper to append an observation-type experience (e.g., reflection shard).

        Args:
            content_text: Text content of the observation
            parent_ids: IDs of parent experiences this observation references
            experience_id: Optional custom ID (auto-generated if not provided)

        Returns:
            Experience ID of created observation
        """
        from src.memory.models import Actor, CaptureMethod, ContentModel, ProvenanceModel

        if experience_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            experience_id = f"obs_{timestamp}_{hash(content_text) % 10000:04x}"

        observation = ExperienceModel(
            id=experience_id,
            type=ExperienceType.OBSERVATION,
            content=ContentModel(text=content_text),
            provenance=ProvenanceModel(actor=Actor.AGENT, method=CaptureMethod.MODEL_INFER),
            parents=parent_ids,
        )

        return self.append_experience(observation)

    def update_experience(self, experience_id: str, **kwargs) -> None:
        """Updates are not allowed in immutable store.

        Raises:
            ImmutabilityViolation: Always
        """
        raise ImmutabilityViolation(
            "Cannot update experiences in immutable store. "
            "Create a new derived experience or use tombstone()."
        )

    def delete_experience(self, experience_id: str) -> None:
        """Deletes are not allowed in immutable store.

        Raises:
            ImmutabilityViolation: Always
        """
        raise ImmutabilityViolation(
            "Cannot delete experiences in immutable store. "
            "Use tombstone() for GDPR/erasure requests."
        )

    def append_experience_idempotent(self, experience: ExperienceModel, idempotency_key: str) -> str:
        """Append experience only if idempotency_key not seen before.

        This enables safe re-execution of task scheduler without creating duplicates.

        Args:
            experience: Experience model to persist
            idempotency_key: Unique key for idempotency check (usually hash of task_id + timestamp)

        Returns:
            Experience ID (existing if duplicate, newly created otherwise)
        """
        with Session(self.engine) as session:
            # Query for existing experience with this idempotency_key
            # Use json_extract to search structured content
            from sqlalchemy import text

            query = text(
                """
                SELECT id FROM experience
                WHERE json_extract(content, '$.structured.idempotency_key') = :key
                LIMIT 1
                """
            )
            result = session.execute(query, {"key": idempotency_key}).first()

            if result:
                # Experience already exists, return existing ID
                return result[0]

            # Not found, create new experience
            return self.append_experience(experience)

    def list_task_executions(
        self, task_id: Optional[str] = None, limit: int = 20
    ) -> list[ExperienceModel]:
        """List TASK_EXECUTION experiences, optionally filtered by task_id.

        Args:
            task_id: Filter by specific task_id (optional)
            limit: Maximum number of experiences to return

        Returns:
            List of ExperienceModels, most recent first
        """
        with Session(self.engine) as session:
            if task_id:
                # Filter by both type and task_id
                from sqlalchemy import text

                query = text(
                    """
                    SELECT * FROM experience
                    WHERE type = :exp_type
                    AND json_extract(content, '$.structured.task_id') = :task_id
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                )
                experiences = session.execute(
                    query,
                    {
                        "exp_type": ExperienceType.TASK_EXECUTION.value,
                        "task_id": task_id,
                        "limit": limit,
                    },
                ).all()
            else:
                # Just filter by type
                statement = (
                    select(Experience)
                    .where(Experience.type == ExperienceType.TASK_EXECUTION.value)
                    .order_by(Experience.created_at.desc())
                    .limit(limit)
                )
                experiences = session.exec(statement).all()

            # Convert SQL rows to ExperienceModel
            if task_id:
                # Manual conversion from Row objects
                result = []
                for row in experiences:
                    exp = Experience(
                        id=row.id,
                        type=row.type,
                        created_at=row.created_at,
                        content=row.content,
                        provenance=row.provenance,
                        evidence_ptrs=row.evidence_ptrs,
                        confidence=row.confidence,
                        embeddings=row.embeddings,
                        affect=row.affect,
                        parents=row.parents,
                        causes=row.causes,
                        sign=row.sign,
                        ownership=row.ownership,
                        session_id=row.session_id,
                        consolidated=row.consolidated,
                    )
                    result.append(experience_to_model(exp))
                return result
            else:
                return [experience_to_model(exp) for exp in experiences]

    def get_by_trace_id(self, trace_id: str) -> Optional[ExperienceModel]:
        """Get experience by trace_id.

        Args:
            trace_id: Correlation/trace ID to search for

        Returns:
            ExperienceModel if found, None otherwise
        """
        with Session(self.engine) as session:
            from sqlalchemy import text

            query = text(
                """
                SELECT * FROM experience
                WHERE json_extract(content, '$.structured.trace_id') = :trace_id
                LIMIT 1
                """
            )
            row = session.execute(query, {"trace_id": trace_id}).first()

            if row is None:
                return None

            # Manual conversion from Row object
            exp = Experience(
                id=row.id,
                type=row.type,
                created_at=row.created_at,
                content=row.content,
                provenance=row.provenance,
                evidence_ptrs=row.evidence_ptrs,
                confidence=row.confidence,
                embeddings=row.embeddings,
                affect=row.affect,
                parents=row.parents,
                causes=row.causes,
                sign=row.sign,
                ownership=row.ownership,
                session_id=row.session_id,
                consolidated=row.consolidated,
            )
            return experience_to_model(exp)

    def tombstone(self, experience_id: str, reason: str) -> bool:
        """Mark experience as tombstoned (soft delete for GDPR compliance).

        MVP: Stubbed implementation. Full implementation would mark the
        record as deleted while preserving provenance chain.

        Args:
            experience_id: Experience ID to tombstone
            reason: Reason for tombstoning (e.g., "gdpr_erasure")

        Returns:
            True if tombstoned, False if not found

        Note:
            In production, this would create a WAL entry and preserve
            cryptographic proof of deletion without removing the record.
        """
        # MVP stub: just verify experience exists
        with Session(self.engine) as session:
            exp = session.get(Experience, experience_id)
            return exp is not None
        # TODO: Implement full tombstone logic with WAL entry

    def count_experiences(self, experience_type: Optional[ExperienceType] = None) -> int:
        """Count total experiences in store.

        Args:
            experience_type: Filter by type (optional)

        Returns:
            Count of experiences
        """
        with Session(self.engine) as session:
            statement = select(Experience)
            if experience_type:
                statement = statement.where(Experience.type == experience_type.value)
            experiences = session.exec(statement).all()
            return len(experiences)

    def close(self):
        """Close database connection."""
        self.engine.dispose()


def create_raw_store(db_path: Optional[str | Path] = None) -> RawStore:
    """Factory function to create RawStore instance.

    Args:
        db_path: Database path (defaults to data/raw_store.db)

    Returns:
        Initialized RawStore instance
    """
    if db_path is None:
        db_path = Path("data/raw_store.db")
    return RawStore(db_path)
