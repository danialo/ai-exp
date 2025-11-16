"""Research session management with HTN task decomposition."""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import uuid4


@dataclass
class ResearchSession:
    """Session for multi-step research with task budgets."""

    id: str = field(default_factory=lambda: str(uuid4()))
    root_question: str = ""
    status: str = "active"  # active, completed, aborted
    max_tasks: int = 50
    max_children_per_task: int = 5
    max_depth: int = 4
    tasks_created: int = 0
    session_summary: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def can_create_tasks(self, count: int = 1) -> bool:
        """Check if we can create more tasks within budget."""
        return (self.tasks_created + count) <= self.max_tasks

    def increment_task_count(self, count: int = 1):
        """Increment tasks_created counter."""
        self.tasks_created += count
        if self.tasks_created >= self.max_tasks:
            self.status = "completed"
            self.completed_at = datetime.now(timezone.utc)


@dataclass
class SourceDoc:
    """Source document with claims and provenance."""

    id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    url: Optional[str] = None
    title: Optional[str] = None
    published_at: Optional[datetime] = None
    claims: List[Dict[str, Any]] = field(default_factory=list)
    content_summary: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchSessionStore:
    """Persist research sessions to SQLite."""

    def __init__(self, db_path: str = "data/core.db"):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_sessions (
                    id TEXT PRIMARY KEY,
                    root_question TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    max_tasks INTEGER DEFAULT 50,
                    max_children_per_task INTEGER DEFAULT 5,
                    max_depth INTEGER DEFAULT 4,
                    tasks_created INTEGER DEFAULT 0,
                    session_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_docs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    url TEXT,
                    title TEXT,
                    published_at TIMESTAMP,
                    claims TEXT,
                    content_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES research_sessions(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_docs_session
                ON source_docs(session_id)
            """)

    def create_session(self, session: ResearchSession) -> ResearchSession:
        """Create new research session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO research_sessions
                (id, root_question, status, max_tasks, max_children_per_task, max_depth,
                 tasks_created, session_summary, created_at, completed_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id,
                session.root_question,
                session.status,
                session.max_tasks,
                session.max_children_per_task,
                session.max_depth,
                session.tasks_created,
                json.dumps(session.session_summary) if session.session_summary else None,
                session.created_at.isoformat(),
                session.completed_at.isoformat() if session.completed_at else None,
                json.dumps(session.metadata)
            ))
        return session

    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Retrieve session by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM research_sessions WHERE id = ?", (session_id,)
            ).fetchone()

            if not row:
                return None

            return ResearchSession(
                id=row["id"],
                root_question=row["root_question"],
                status=row["status"],
                max_tasks=row["max_tasks"],
                max_children_per_task=row["max_children_per_task"],
                max_depth=row["max_depth"],
                tasks_created=row["tasks_created"],
                session_summary=json.loads(row["session_summary"]) if row["session_summary"] else None,
                created_at=datetime.fromisoformat(row["created_at"]),
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )

    def update_session(self, session: ResearchSession):
        """Update existing session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE research_sessions
                SET status = ?, tasks_created = ?, session_summary = ?,
                    completed_at = ?, metadata = ?
                WHERE id = ?
            """, (
                session.status,
                session.tasks_created,
                json.dumps(session.session_summary) if session.session_summary else None,
                session.completed_at.isoformat() if session.completed_at else None,
                json.dumps(session.metadata),
                session.id
            ))

    def create_source_doc(self, doc: SourceDoc) -> SourceDoc:
        """Create source document."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO source_docs
                (id, session_id, url, title, published_at, claims, content_summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id,
                doc.session_id,
                doc.url,
                doc.title,
                doc.published_at.isoformat() if doc.published_at else None,
                json.dumps(doc.claims),
                doc.content_summary,
                doc.created_at.isoformat()
            ))
        return doc

    def list_source_docs(self, session_id: str) -> List[SourceDoc]:
        """List all source docs for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM source_docs WHERE session_id = ? ORDER BY created_at",
                (session_id,)
            ).fetchall()

            return [
                SourceDoc(
                    id=row["id"],
                    session_id=row["session_id"],
                    url=row["url"],
                    title=row["title"],
                    published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
                    claims=json.loads(row["claims"]),
                    content_summary=row["content_summary"],
                    created_at=datetime.fromisoformat(row["created_at"])
                )
                for row in rows
            ]

    def increment_tasks_created(self, session_id: str, n: int) -> None:
        """Increment tasks_created counter."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE research_sessions
                SET tasks_created = COALESCE(tasks_created, 0) + ?
                WHERE id = ?
            """, (n, session_id))

    def mark_complete(self, session_id: str) -> None:
        """Mark session as completed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE research_sessions
                SET status = 'completed', completed_at = ?
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), session_id))

    def save_session_summary(self, session_id: str, summary: Dict[str, Any]) -> None:
        """Save synthesis summary to session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE research_sessions
                SET session_summary = ?
                WHERE id = ?
            """, (json.dumps(summary), session_id))

    def load_source_docs_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all source docs for synthesis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM source_docs
                WHERE session_id = ?
                ORDER BY created_at ASC
            """, (session_id,)).fetchall()

        docs = []
        for r in rows:
            docs.append({
                "id": r["id"],
                "url": r["url"],
                "title": r["title"],
                "published_at": r["published_at"],
                "claims": json.loads(r["claims"]) if r["claims"] else [],
                "content_summary": r["content_summary"],
            })
        return docs
