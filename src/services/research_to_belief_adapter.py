"""Adapter from research session summaries to belief updates."""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from uuid import uuid4


@dataclass
class BeliefUpdate:
    """Proposed update to belief system from research findings."""

    id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    kind: str = "informational"  # new, reinforce, weaken, contest, informational
    summary: str = ""  # Human-readable summary
    payload: Dict[str, Any] = field(default_factory=dict)  # JSON details
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)


class BeliefUpdateStore:
    """Persist belief updates to SQLite."""

    def __init__(self, db_path: str = "data/core.db"):
        self.db_path = db_path
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Create belief_updates table if not exists."""
        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cx.execute("""
                CREATE TABLE IF NOT EXISTS belief_updates (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    payload TEXT,
                    confidence REAL,
                    created_at REAL,
                    FOREIGN KEY (session_id) REFERENCES research_sessions(id)
                )
            """)
            cx.execute("""
                CREATE INDEX IF NOT EXISTS idx_belief_updates_session
                ON belief_updates(session_id)
            """)
            cx.execute("COMMIT")

    def create_many(self, updates: List[BeliefUpdate]) -> None:
        """Bulk insert belief updates."""
        if not updates:
            return
        with self._conn() as cx:
            cx.execute("BEGIN IMMEDIATE")
            cx.executemany("""
                INSERT INTO belief_updates (id, session_id, kind, summary, payload, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    u.id, u.session_id, u.kind, u.summary,
                    json.dumps(u.payload), u.confidence, u.created_at
                ) for u in updates
            ])
            cx.execute("COMMIT")

    def list_for_session(self, session_id: str) -> List[BeliefUpdate]:
        """List all belief updates for a session."""
        with self._conn() as cx:
            rows = cx.execute("""
                SELECT * FROM belief_updates
                WHERE session_id = ?
                ORDER BY created_at ASC
            """, (session_id,)).fetchall()

        updates = []
        for r in rows:
            updates.append(BeliefUpdate(
                id=r["id"],
                session_id=r["session_id"],
                kind=r["kind"],
                summary=r["summary"],
                payload=json.loads(r["payload"]) if r["payload"] else {},
                confidence=float(r["confidence"] or 0.5),
                created_at=float(r["created_at"] or time.time())
            ))
        return updates


def _classify_update_kind(contested_claims: List[Dict], sources_count: int) -> str:
    """
    Classify belief update kind based on contested claims and source diversity.

    Rules:
    - 0 contested, multiple independent domains → "reinforce"
    - >0 contested but majority high-quality sources agree → "contest_minor"
    - >0 contested and split along low-quality vs high-quality → "informational"

    Args:
        contested_claims: List of contested claim dicts
        sources_count: Number of distinct sources investigated

    Returns:
        kind string: "reinforce", "contest_minor", or "informational"
    """
    contested_count = len(contested_claims)

    # No contested claims + multiple sources = reinforce
    if contested_count == 0 and sources_count >= 3:
        return "reinforce"

    # Has contested claims - check if minor or major
    if contested_count > 0:
        # For now, treat all contested as minor
        # Later: Could parse claim.reason to detect quality split
        return "contest_minor"

    # Default: informational
    return "informational"


def propose_updates(session, session_summary: Dict[str, Any]) -> List[BeliefUpdate]:
    """
    Generate candidate belief updates from research session summary.

    Automatically classifies kind based on contested claims and source diversity:
    - "reinforce": No contested claims, multiple independent sources
    - "contest_minor": Has contested claims but resolvable
    - "informational": Default for unclear or split findings

    Args:
        session: ResearchSession object
        session_summary: Synthesis output dict

    Returns:
        List of BeliefUpdate objects
    """
    updates = []

    key_events = session_summary.get("key_events", [])
    contested_claims = session_summary.get("contested_claims", [])
    open_questions = session_summary.get("open_questions", [])
    stats = session_summary.get("coverage_stats", {})
    sources_count = stats.get("sources_investigated", 0)

    # Classify kind automatically
    kind = _classify_update_kind(contested_claims, sources_count)

    # Build summary text
    summary_parts = []
    if key_events:
        summary_parts.append(f"Found {len(key_events)} key events")
    if contested_claims:
        summary_parts.append(f"{len(contested_claims)} contested claims")
    if open_questions:
        summary_parts.append(f"{len(open_questions)} open questions")

    summary_text = f"Research on '{session.root_question}': " + ", ".join(summary_parts)

    # Set confidence based on sources and contestation
    if kind == "reinforce":
        confidence = 0.8  # High confidence - multiple sources agree
    elif kind == "contest_minor":
        confidence = 0.5  # Medium confidence - some disagreement
    else:
        confidence = 0.6  # Default moderate confidence

    # Create classified update
    update = BeliefUpdate(
        session_id=session.id,
        kind=kind,
        summary=summary_text,
        payload={
            "root_question": session.root_question,
            "key_events": key_events,
            "contested_claims": contested_claims,
            "open_questions": open_questions,
            "coverage_stats": stats
        },
        confidence=confidence
    )
    updates.append(update)

    # TODO: Expand this to create specific updates for:
    # - Each key event (kind="new" or "reinforce")
    # - Each contested claim (kind="contest")
    # - High-confidence claims (kind="reinforce")

    return updates


def store_belief_updates(updates: List[BeliefUpdate]) -> None:
    """Persist belief updates to database."""
    if not updates:
        return
    store = BeliefUpdateStore()
    store.create_many(updates)
