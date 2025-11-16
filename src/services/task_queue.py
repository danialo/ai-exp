"""Task queue with HTN task type dispatch."""

from __future__ import annotations
import json
import sqlite3
import uuid
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DB_PATH = "data/core.db"


@dataclass
class Task:
    """Task in the HTN execution queue."""
    id: str
    session_id: str
    htn_task_type: str
    args: Dict[str, Any]
    status: str = "queued"  # queued | running | done | error
    depth: int = 0
    parent_id: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.updated_at:
            self.updated_at = time.time()


class TaskStore:
    """Persist tasks to SQLite with atomic pop operations."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Create tasks table if not exists."""
        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cx.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    htn_task_type TEXT NOT NULL,
                    args TEXT,
                    status TEXT NOT NULL DEFAULT 'queued',
                    depth INTEGER DEFAULT 0,
                    parent_id TEXT,
                    created_at REAL,
                    updated_at REAL,
                    FOREIGN KEY(session_id) REFERENCES research_sessions(id)
                )
            """)
            cx.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
            cx.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)")
            cx.execute("COMMIT")

    def create_many(self, tasks: List[Task]) -> None:
        """Bulk insert tasks."""
        if not tasks:
            return
        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cx.executemany("""
                INSERT INTO tasks (id, session_id, htn_task_type, args, status, depth, parent_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    t.id, t.session_id, t.htn_task_type, json.dumps(t.args or {}),
                    t.status, t.depth, t.parent_id, t.created_at, t.updated_at
                ) for t in tasks
            ])
            cx.execute("COMMIT")

    def create_one(self, task: Task) -> None:
        """Insert single task."""
        self.create_many([task])

    def pop_next_queued(self) -> Optional[Task]:
        """Atomically pop next queued task (FIFO by created_at)."""
        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            row = cx.execute("""
                SELECT id FROM tasks
                WHERE status='queued'
                ORDER BY created_at ASC
                LIMIT 1
            """).fetchone()
            if not row:
                cx.execute("COMMIT")
                return None
            task_id = row["id"]
            cx.execute("""
                UPDATE tasks
                SET status='running', updated_at=?
                WHERE id=? AND status='queued'
            """, (time.time(), task_id))
            row2 = cx.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            cx.execute("COMMIT")
        return self._row_to_task(row2) if row2 else None

    def mark_done(self, task_id: str) -> None:
        """Mark task as completed."""
        with self._conn() as cx:
            cx.execute("UPDATE tasks SET status='done', updated_at=? WHERE id=?", (time.time(), task_id))

    def mark_error(self, task_id: str, reason: str = "") -> None:
        """Mark task as failed."""
        with self._conn() as cx:
            cx.execute("UPDATE tasks SET status='error', updated_at=? WHERE id=?", (time.time(), task_id))

    def queued_count(self, session_id: str) -> int:
        """Count queued tasks for a session."""
        with self._conn() as cx:
            row = cx.execute("""
                SELECT COUNT(*) c FROM tasks WHERE session_id=? AND status='queued'
            """, (session_id,)).fetchone()
            return int(row["c"])

    def list_tasks_for_session(self, session_id: str) -> List[Task]:
        """List all tasks for a session (for synthesis)."""
        with self._conn() as cx:
            rows = cx.execute("""
                SELECT * FROM tasks
                WHERE session_id = ?
                ORDER BY created_at ASC
            """, (session_id,)).fetchall()
        return [self._row_to_task(r) for r in rows]

    def _row_to_task(self, r: sqlite3.Row) -> Task:
        """Convert SQLite row to Task dataclass."""
        return Task(
            id=r["id"],
            session_id=r["session_id"],
            htn_task_type=r["htn_task_type"],
            args=json.loads(r["args"] or "{}"),
            status=r["status"],
            depth=int(r["depth"] or 0),
            parent_id=r["parent_id"],
            created_at=float(r["created_at"] or time.time()),
            updated_at=float(r["updated_at"] or time.time()),
        )


def new_task(
    session_id: str,
    htn_task_type: str,
    args: Dict[str, Any],
    depth: int = 0,
    parent_id: Optional[str] = None
) -> Task:
    """Factory function for creating new tasks."""
    return Task(
        id=str(uuid.uuid4()),
        session_id=session_id,
        htn_task_type=htn_task_type,
        args=args or {},
        depth=depth,
        parent_id=parent_id,
    )
