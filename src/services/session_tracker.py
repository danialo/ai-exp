"""Session tracking service for managing conversation sessions.

Provides hybrid session tracking with both explicit markers and timeout-based
automatic session detection.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from pathlib import Path

from sqlalchemy import create_engine
from sqlmodel import Session as DBSession, select

from src.memory.models import Session, SessionStatus


class SessionTracker:
    """Service for tracking and managing conversation sessions."""

    def __init__(self, db_path: str | Path, timeout_minutes: int = 30):
        """Initialize session tracker.

        Args:
            db_path: Path to SQLite database
            timeout_minutes: Inactivity timeout in minutes
        """
        self.db_path = Path(db_path)
        self.timeout_minutes = timeout_minutes
        self.timeout_delta = timedelta(minutes=timeout_minutes)

        # Create engine with check_same_thread=False for SQLite
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})

    def start_session(self, user_id: str = "default_user", session_id: Optional[str] = None) -> Session:
        """Start a new session explicitly.

        Args:
            user_id: User identifier
            session_id: Optional custom session ID

        Returns:
            Created Session object
        """
        if session_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            session_id = f"sess_{user_id}_{timestamp}"

        session = Session(
            id=session_id,
            user_id=user_id,
            start_time=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            status=SessionStatus.ACTIVE.value,
        )

        with DBSession(self.engine) as db:
            db.add(session)
            db.commit()
            db.refresh(session)

        return session

    def get_or_create_active_session(self, user_id: str = "default_user") -> Session:
        """Get active session or create new one if timeout exceeded.

        This implements the timeout-based session detection.

        Args:
            user_id: User identifier

        Returns:
            Active Session object (existing or newly created)
        """
        with DBSession(self.engine) as db:
            # Find most recent active session for user
            statement = (
                select(Session)
                .where(Session.user_id == user_id)
                .where(Session.status == SessionStatus.ACTIVE.value)
                .order_by(Session.last_activity.desc())
            )
            active_session = db.exec(statement).first()

            # Check if session timed out
            now = datetime.now(timezone.utc)
            if active_session:
                # Ensure timezone-aware comparison
                last_activity = active_session.last_activity
                if last_activity.tzinfo is None:
                    last_activity = last_activity.replace(tzinfo=timezone.utc)

                time_since_activity = now - last_activity

                if time_since_activity < self.timeout_delta:
                    # Session still active, update last activity
                    active_session.last_activity = now
                    db.add(active_session)
                    db.commit()
                    db.refresh(active_session)
                    return active_session
                else:
                    # Session timed out, end it
                    active_session.status = SessionStatus.ENDED.value
                    active_session.end_time = now
                    db.add(active_session)
                    db.commit()

        # Create new session
        return self.start_session(user_id=user_id)

    def update_activity(self, session_id: str) -> bool:
        """Update last activity timestamp for a session.

        Args:
            session_id: Session ID

        Returns:
            True if updated, False if session not found
        """
        with DBSession(self.engine) as db:
            session = db.get(Session, session_id)
            if not session:
                return False

            session.last_activity = datetime.now(timezone.utc)
            db.add(session)
            db.commit()
            return True

    def add_experience(self, session_id: str, experience_id: str) -> bool:
        """Add an experience ID to a session's experience list.

        Args:
            session_id: Session ID
            experience_id: Experience ID to add

        Returns:
            True if added, False if session not found
        """
        with DBSession(self.engine) as db:
            session = db.get(Session, session_id)
            if not session:
                return False

            # Ensure experience_ids is a list (handle JSON deserialization)
            if not isinstance(session.experience_ids, list):
                session.experience_ids = []

            if experience_id not in session.experience_ids:
                session.experience_ids = session.experience_ids + [experience_id]
                session.last_activity = datetime.now(timezone.utc)
                db.add(session)
                db.commit()

            return True

    def end_session(self, session_id: str) -> Optional[Session]:
        """End a session explicitly.

        Args:
            session_id: Session ID to end

        Returns:
            Updated Session object, or None if not found
        """
        with DBSession(self.engine) as db:
            session = db.get(Session, session_id)
            if not session:
                return None

            session.status = SessionStatus.ENDED.value
            session.end_time = datetime.now(timezone.utc)
            db.add(session)
            db.commit()
            db.refresh(session)
            return session

    def mark_consolidated(self, session_id: str, narrative_id: str) -> bool:
        """Mark a session as consolidated with narrative ID.

        Args:
            session_id: Session ID
            narrative_id: ID of consolidated narrative experience

        Returns:
            True if updated, False if session not found
        """
        with DBSession(self.engine) as db:
            session = db.get(Session, session_id)
            if not session:
                return False

            session.status = SessionStatus.CONSOLIDATED.value
            session.consolidated_narrative_id = narrative_id
            db.add(session)
            db.commit()
            return True

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session object or None if not found
        """
        with DBSession(self.engine) as db:
            return db.get(Session, session_id)

    def list_ended_sessions(self, limit: int = 10) -> list[Session]:
        """List ended sessions that haven't been consolidated yet.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of Session objects
        """
        with DBSession(self.engine) as db:
            statement = (
                select(Session)
                .where(Session.status == SessionStatus.ENDED.value)
                .order_by(Session.end_time.desc())
                .limit(limit)
            )
            return list(db.exec(statement).all())

    def close(self):
        """Close database connection."""
        self.engine.dispose()


def create_session_tracker(
    db_path: Optional[str | Path] = None,
    timeout_minutes: int = 30,
) -> SessionTracker:
    """Factory function to create SessionTracker instance.

    Args:
        db_path: Database path (defaults to data/raw_store.db)
        timeout_minutes: Inactivity timeout in minutes

    Returns:
        Initialized SessionTracker instance
    """
    if db_path is None:
        db_path = Path("data/raw_store.db")
    return SessionTracker(db_path, timeout_minutes=timeout_minutes)
