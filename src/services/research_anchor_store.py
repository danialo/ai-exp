"""Lightweight anchor system for research session reuse.

Stores tiny summaries of completed research sessions to avoid re-investigating
the same topics repeatedly.
"""

import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ResearchAnchor:
    """Lightweight pointer to a completed research session."""

    session_id: str
    topic: str  # Normalized root question
    one_sentence_summary: str
    created_at: float
    days_valid: int = 7  # How long this anchor is considered fresh


class ResearchAnchorStore:
    """Persist and query research session anchors."""

    def __init__(self, db_path: str = "data/core.db"):
        self.db_path = db_path
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Create research_anchors table if not exists."""
        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cx.execute("""
                CREATE TABLE IF NOT EXISTS research_anchors (
                    session_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    one_sentence_summary TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    days_valid INTEGER DEFAULT 7,
                    FOREIGN KEY (session_id) REFERENCES research_sessions(id)
                )
            """)
            cx.execute("""
                CREATE INDEX IF NOT EXISTS idx_research_anchors_topic
                ON research_anchors(topic)
            """)
            cx.execute("""
                CREATE INDEX IF NOT EXISTS idx_research_anchors_created
                ON research_anchors(created_at DESC)
            """)
            cx.execute("COMMIT")

    def create_anchor(
        self,
        session_id: str,
        topic: str,
        one_sentence_summary: str,
        days_valid: int = 7
    ) -> None:
        """Create a new research anchor.

        Args:
            session_id: Session UUID
            topic: Normalized root question (lowercase, stripped)
            one_sentence_summary: One-line summary of findings
            days_valid: How many days this anchor stays fresh
        """
        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cx.execute("""
                INSERT OR REPLACE INTO research_anchors
                (session_id, topic, one_sentence_summary, created_at, days_valid)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, topic.lower().strip(), one_sentence_summary, time.time(), days_valid))
            cx.execute("COMMIT")

    def find_recent_anchor(
        self,
        topic: str,
        max_age_days: int = 7
    ) -> Optional[ResearchAnchor]:
        """Find most recent anchor for a topic within max age.

        Args:
            topic: Topic to search for (will be normalized)
            max_age_days: Maximum age in days (default: 7)

        Returns:
            ResearchAnchor if found and fresh, else None
        """
        normalized_topic = topic.lower().strip()
        cutoff_time = time.time() - (max_age_days * 86400)

        with self._conn() as cx:
            row = cx.execute("""
                SELECT * FROM research_anchors
                WHERE topic = ?
                  AND created_at > ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (normalized_topic, cutoff_time)).fetchone()

        if not row:
            return None

        return ResearchAnchor(
            session_id=row["session_id"],
            topic=row["topic"],
            one_sentence_summary=row["one_sentence_summary"],
            created_at=float(row["created_at"]),
            days_valid=int(row["days_valid"] or 7)
        )

    def list_recent_anchors(self, max_age_days: int = 7, limit: int = 20) -> List[ResearchAnchor]:
        """List all recent anchors, sorted by creation time descending.

        Args:
            max_age_days: Maximum age in days
            limit: Max number to return

        Returns:
            List of ResearchAnchor objects
        """
        cutoff_time = time.time() - (max_age_days * 86400)

        with self._conn() as cx:
            rows = cx.execute("""
                SELECT * FROM research_anchors
                WHERE created_at > ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (cutoff_time, limit)).fetchall()

        anchors = []
        for row in rows:
            anchors.append(ResearchAnchor(
                session_id=row["session_id"],
                topic=row["topic"],
                one_sentence_summary=row["one_sentence_summary"],
                created_at=float(row["created_at"]),
                days_valid=int(row["days_valid"] or 7)
            ))
        return anchors

    def cleanup_old_anchors(self, max_age_days: int = 30) -> int:
        """Delete anchors older than max_age_days.

        Args:
            max_age_days: Delete anchors older than this

        Returns:
            Number of anchors deleted
        """
        cutoff_time = time.time() - (max_age_days * 86400)

        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cursor = cx.execute("""
                DELETE FROM research_anchors
                WHERE created_at < ?
            """, (cutoff_time,))
            deleted_count = cursor.rowcount
            cx.execute("COMMIT")

        return deleted_count


def create_anchor_from_session(session_id: str, summary_obj: dict) -> None:
    """
    Create an anchor after research session completes.

    Args:
        session_id: Research session UUID
        summary_obj: Synthesis summary from research_and_summarize
    """
    from src.services.research_session import ResearchSessionStore
    import logging

    logger = logging.getLogger(__name__)

    # Load session to get root_question
    session_store = ResearchSessionStore()
    session = session_store.get_session(session_id)
    if not session:
        return

    # Guard rail: Don't create anchors for sessions with zero docs
    doc_count = summary_obj.get("coverage_stats", {}).get("total_docs", 0)
    if doc_count == 0:
        logger.info(f"Skipping anchor creation for session {session_id} (zero docs)")
        return

    # Build one-sentence summary from key events
    key_events = summary_obj.get("key_events", [])
    if key_events:
        # Use first key event as summary (handle if it's a dict)
        first_event = key_events[0]
        if isinstance(first_event, dict):
            # Try common keys: description, event, summary, then stringify
            one_sentence = first_event.get("description",
                           first_event.get("event",
                           first_event.get("summary", str(first_event))))
            # Ensure it's a string
            if not isinstance(one_sentence, str):
                one_sentence = str(one_sentence)
        else:
            one_sentence = str(first_event)
    else:
        # Fallback to narrative
        narrative = summary_obj.get("narrative_summary", "")
        one_sentence = narrative[:200] if narrative else "Research session completed"

    # Create anchor
    anchor_store = ResearchAnchorStore()
    anchor_store.create_anchor(
        session_id=session_id,
        topic=session.root_question,
        one_sentence_summary=one_sentence,
        days_valid=7
    )
