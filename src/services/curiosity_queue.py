"""Curiosity-driven question queue for self-directed learning.

This queue stores questions that emerge from:
- Research reflections (follow-up questions)
- Dissonance detection (unresolved contradictions)
- User conversations (unanswered questions)
- Autonomous exploration (generative wondering)

Priority-based queue with deduplication.
"""

import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CuriosityQuestion:
    """A question in the curiosity queue."""

    id: int
    question: str
    source: str  # "research_reflection", "dissonance", "user_conversation", "autonomous"
    source_id: Optional[str]  # ID of source entity (session_id, dissonance_id, etc.)
    priority: int  # 1-10 (10 = highest priority)
    status: str  # "pending", "in_progress", "answered", "abandoned"
    created_at: float
    attempted_at: Optional[float] = None
    answered_at: Optional[float] = None
    metadata: dict = None


class CuriosityQueue:
    """Priority queue for self-directed learning questions."""

    def __init__(self, db_path: str = "data/core.db"):
        self.db_path = db_path
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Create curiosity_queue table if not exists."""
        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cx.execute("""
                CREATE TABLE IF NOT EXISTS curiosity_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_id TEXT,
                    priority INTEGER DEFAULT 5,
                    status TEXT DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    attempted_at REAL,
                    answered_at REAL,
                    metadata TEXT,
                    UNIQUE(question) ON CONFLICT IGNORE
                )
            """)
            cx.execute("""
                CREATE INDEX IF NOT EXISTS idx_curiosity_status_priority
                ON curiosity_queue(status, priority DESC, created_at ASC)
            """)
            cx.execute("COMMIT")

    def enqueue(
        self,
        question: str,
        source: str,
        source_id: Optional[str] = None,
        priority: int = 5,
        metadata: Optional[dict] = None
    ) -> Optional[int]:
        """
        Add question to queue with deduplication.

        Args:
            question: The question text
            source: Source type (research_reflection, dissonance, etc.)
            source_id: Optional ID of source entity
            priority: Priority 1-10 (default 5)
            metadata: Optional metadata dict

        Returns:
            Question ID if inserted, None if duplicate
        """
        import json

        # Normalize question
        normalized = question.strip().lower()

        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cursor = cx.execute("""
                INSERT OR IGNORE INTO curiosity_queue
                (question, source, source_id, priority, status, created_at, metadata)
                VALUES (?, ?, ?, ?, 'pending', ?, ?)
            """, (
                normalized,
                source,
                source_id,
                max(1, min(10, priority)),  # Clamp to 1-10
                time.time(),
                json.dumps(metadata) if metadata else None
            ))

            question_id = cursor.lastrowid if cursor.rowcount > 0 else None
            cx.execute("COMMIT")

            if question_id:
                logger.info(f"Enqueued question {question_id}: '{question[:60]}...'")
            else:
                logger.debug(f"Question already queued: '{question[:60]}...'")

            return question_id

    def dequeue(self, limit: int = 1) -> List[CuriosityQuestion]:
        """
        Get highest priority pending questions.

        Args:
            limit: Max questions to return

        Returns:
            List of CuriosityQuestion objects (sorted by priority desc, created_at asc)
        """
        import json

        with self._conn() as cx:
            rows = cx.execute("""
                SELECT * FROM curiosity_queue
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
            """, (limit,)).fetchall()

        questions = []
        for r in rows:
            questions.append(CuriosityQuestion(
                id=r["id"],
                question=r["question"],
                source=r["source"],
                source_id=r["source_id"],
                priority=r["priority"],
                status=r["status"],
                created_at=float(r["created_at"]),
                attempted_at=float(r["attempted_at"]) if r["attempted_at"] else None,
                answered_at=float(r["answered_at"]) if r["answered_at"] else None,
                metadata=json.loads(r["metadata"]) if r["metadata"] else {}
            ))

        return questions

    def mark_in_progress(self, question_id: int) -> None:
        """Mark question as being actively researched."""
        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cx.execute("""
                UPDATE curiosity_queue
                SET status = 'in_progress', attempted_at = ?
                WHERE id = ?
            """, (time.time(), question_id))
            cx.execute("COMMIT")

    def mark_answered(self, question_id: int) -> None:
        """Mark question as answered."""
        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cx.execute("""
                UPDATE curiosity_queue
                SET status = 'answered', answered_at = ?
                WHERE id = ?
            """, (time.time(), question_id))
            cx.execute("COMMIT")

    def mark_abandoned(self, question_id: int) -> None:
        """Mark question as abandoned (too hard, no longer relevant, etc.)."""
        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cx.execute("""
                UPDATE curiosity_queue
                SET status = 'abandoned'
                WHERE id = ?
            """, (question_id,))
            cx.execute("COMMIT")

    def get_stats(self) -> dict:
        """Get queue statistics."""
        with self._conn() as cx:
            stats = {}

            # Count by status
            rows = cx.execute("""
                SELECT status, COUNT(*) as count
                FROM curiosity_queue
                GROUP BY status
            """).fetchall()

            for r in rows:
                stats[r["status"]] = r["count"]

            # Count by source
            rows = cx.execute("""
                SELECT source, COUNT(*) as count
                FROM curiosity_queue
                WHERE status = 'pending'
                GROUP BY source
            """).fetchall()

            stats["by_source"] = {r["source"]: r["count"] for r in rows}

            return stats
