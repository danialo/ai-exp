"""GoalStore service: persist, rank, and manage goals.

Implements CRUD with optimistic locking, soft delete, idempotent command ops,
prioritization scoring, and optional safety vetting against beliefs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.services.identity_ledger import append_event, LedgerEvent

logger = logging.getLogger(__name__)


class GoalCategory(str, Enum):
    INTROSPECTION = "introspection"
    EXPLORATION = "exploration"
    MAINTENANCE = "maintenance"
    USER_REQUESTED = "user_requested"


class GoalState(str, Enum):
    PROPOSED = "proposed"
    ADOPTED = "adopted"
    EXECUTING = "executing"
    SATISFIED = "satisfied"
    ABANDONED = "abandoned"


class GoalSource(str, Enum):
    """Source of goal creation."""
    USER = "user"
    SYSTEM = "system"
    COLLABORATIVE = "collaborative"


@dataclass
class GoalDefinition:
    id: str
    text: str
    category: GoalCategory
    value: float
    effort: float
    risk: float
    horizon_min_min: int
    horizon_max_min: Optional[int] = None
    aligns_with: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    state: GoalState = GoalState.PROPOSED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 0
    deleted_at: Optional[datetime] = None
    # Source tracking fields
    source: GoalSource = GoalSource.USER
    created_by: Optional[str] = None  # user_id or detector_name
    proposal_id: Optional[str] = None  # Link to original proposal
    auto_approved: bool = False  # True if created without user review


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GoalStore:
    """SQLite-backed store for goals.

    Uses a provided SQLite database path (shared with raw_store). Arrays and dicts are
    stored as JSON strings.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # Idempotent creation (run all migrations)
        from pathlib import Path

        migrations = [
            "scripts/migrate_001_goal_store.sql",
            "scripts/migrate_002_goal_source_tracking.sql",
        ]

        for migration_path in migrations:
            sql_path = Path(migration_path)
            if sql_path.exists():
                try:
                    with open(sql_path, "r") as f:
                        sql = f.read()
                    self.conn.executescript(sql)
                    self.conn.commit()
                except sqlite3.OperationalError as e:
                    # Ignore "duplicate column" errors (migration already applied)
                    if "duplicate column name" not in str(e).lower():
                        raise

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # --------- Persistence helpers ---------
    def _row_to_goal(self, row: sqlite3.Row) -> GoalDefinition:
        return GoalDefinition(
            id=row["id"],
            text=row["text"],
            category=GoalCategory(row["category"]),
            value=float(row["value"]),
            effort=float(row["effort"]),
            risk=float(row["risk"]),
            horizon_min_min=int(row["horizon_min_min"]),
            horizon_max_min=int(row["horizon_max_min"]) if row["horizon_max_min"] is not None else None,
            aligns_with=json.loads(row["aligns_with"]) if row["aligns_with"] else [],
            contradicts=json.loads(row["contradicts"]) if row["contradicts"] else [],
            success_metrics=json.loads(row["success_metrics"]) if row["success_metrics"] else {},
            state=GoalState(row["state"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            version=int(row["version"]),
            deleted_at=datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None,
            # Source tracking fields (with defaults for backward compatibility)
            source=GoalSource(row["source"]) if "source" in row.keys() else GoalSource.USER,
            created_by=row["created_by"] if "created_by" in row.keys() else None,
            proposal_id=row["proposal_id"] if "proposal_id" in row.keys() else None,
            auto_approved=bool(row["auto_approved"]) if "auto_approved" in row.keys() else False,
        )

    def _insert_idempotency(self, key: str, op: str, entity_id: str) -> bool:
        try:
            self.conn.execute(
                "INSERT INTO goal_idempotency(key, op, entity_id, created_at) VALUES (?,?,?,?)",
                (key, op, entity_id, _utcnow().isoformat()),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def _lookup_idempotency(self, key: str) -> Optional[Tuple[str, str]]:
        cur = self.conn.execute(
            "SELECT op, entity_id FROM goal_idempotency WHERE key=?",
            (key,),
        )
        row = cur.fetchone()
        if row:
            return row["op"], row["entity_id"]
        return None

    # --------- Public API ---------
    def create_goal(self, goal: GoalDefinition, idempotency_key: Optional[str] = None) -> GoalDefinition:
        if idempotency_key:
            existing = self._lookup_idempotency(idempotency_key)
            if existing:
                # Return existing entity if create was seen
                _, entity_id = existing
                g = self.get_goal(entity_id)
                if g:
                    return g

        dt = _utcnow().isoformat()
        goal.created_at = datetime.fromisoformat(dt)
        goal.updated_at = goal.created_at
        self.conn.execute(
            """
            INSERT INTO goals (id, text, category, value, effort, risk, horizon_min_min, horizon_max_min,
                               aligns_with, contradicts, success_metrics, state, created_at, updated_at, metadata,
                               version, deleted_at, source, created_by, proposal_id, auto_approved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?)
            """,
            (
                goal.id,
                goal.text,
                goal.category.value,
                goal.value,
                goal.effort,
                goal.risk,
                goal.horizon_min_min,
                goal.horizon_max_min,
                json.dumps(goal.aligns_with or []),
                json.dumps(goal.contradicts or []),
                json.dumps(goal.success_metrics or {}),
                goal.state.value,
                dt,
                dt,
                json.dumps(goal.metadata or {}),
                goal.source.value,
                goal.created_by,
                goal.proposal_id,
                int(goal.auto_approved),
            ),
        )
        self.conn.commit()

        if idempotency_key:
            self._insert_idempotency(idempotency_key, op="create", entity_id=goal.id)

        # Ledger
        append_event(
            LedgerEvent(
                ts=datetime.now(timezone.utc).timestamp(),
                schema=2,
                event="goal_created",
                meta={
                    "goal_id": goal.id,
                    "category": goal.category.value,
                    "source": goal.source.value,
                    "created_by": goal.created_by,
                },
            )
        )

        return goal

    def get_goal(self, goal_id: str) -> Optional[GoalDefinition]:
        cur = self.conn.execute("SELECT * FROM goals WHERE id=?", (goal_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_goal(row)

    def list_goals(
        self,
        state: Optional[GoalState] = None,
        category: Optional[GoalCategory] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[GoalDefinition]:
        q = "SELECT * FROM goals WHERE 1=1"
        params: List[Any] = []
        if state:
            q += " AND state=?"
            params.append(state.value)
        if category:
            q += " AND category=?"
            params.append(category.value)
        q += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cur = self.conn.execute(q, tuple(params))
        return [self._row_to_goal(r) for r in cur.fetchall()]

    def update_goal(self, goal_id: str, updates: Dict[str, Any], expected_version: int) -> Optional[GoalDefinition]:
        # Build dynamic update
        allowed = {
            "text",
            "category",
            "value",
            "effort",
            "risk",
            "horizon_min_min",
            "horizon_max_min",
            "aligns_with",
            "contradicts",
            "success_metrics",
            "state",
            "metadata",
        }
        sets = []
        params: List[Any] = []
        for k, v in updates.items():
            if k not in allowed:
                continue
            if k in {"aligns_with", "contradicts", "success_metrics", "metadata"}:
                v = json.dumps(v)
            elif k == "category" and isinstance(v, GoalCategory):
                v = v.value
            elif k == "state" and isinstance(v, GoalState):
                v = v.value
            sets.append(f"{k}=?")
            params.append(v)

        if not sets:
            return self.get_goal(goal_id)

        params.extend([_utcnow().isoformat(), goal_id, expected_version])
        sql = f"UPDATE goals SET {', '.join(sets)}, updated_at=?, version=version+1 WHERE id=? AND version=?"
        cur = self.conn.execute(sql, tuple(params))
        self.conn.commit()
        if cur.rowcount == 0:
            return None  # version conflict or not found
        return self.get_goal(goal_id)

    def abandon_goal(self, goal_id: str, idempotency_key: Optional[str] = None) -> Optional[GoalDefinition]:
        if idempotency_key and self._lookup_idempotency(idempotency_key):
            return self.get_goal(goal_id)

        now = _utcnow().isoformat()
        cur = self.conn.execute(
            "UPDATE goals SET state=?, deleted_at=?, updated_at=?, version=version+1 WHERE id=?",
            (GoalState.ABANDONED.value, now, now, goal_id),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            return None

        if idempotency_key:
            self._insert_idempotency(idempotency_key, op="abandon", entity_id=goal_id)

        append_event(
            LedgerEvent(
                ts=datetime.now(timezone.utc).timestamp(),
                schema=2,
                event="goal_abandoned",
                meta={"goal_id": goal_id},
            )
        )
        return self.get_goal(goal_id)

    # --------- Ranking ---------
    @staticmethod
    def compute_urgency(created_at: datetime, horizon_max_min: Optional[int], now: Optional[datetime] = None) -> float:
        now = now or _utcnow()
        if not horizon_max_min:
            return 0.0
        elapsed_min = (now - created_at).total_seconds() / 60
        remaining_min = horizon_max_min - elapsed_min
        if remaining_min > 0:
            hours_remaining = remaining_min / 60
            return 0.0 if hours_remaining > 24 else (1.0 - hours_remaining / 24)
        return -1.0

    @staticmethod
    def score_goal(
        goal: GoalDefinition,
        weights: Dict[str, float],
        active_beliefs: Optional[Iterable[str]] = None,
        now: Optional[datetime] = None,
    ) -> float:
        now = now or _utcnow()
        wv = float(weights.get("value_weight", 0.5))
        we = float(weights.get("effort_weight", 0.25))
        wr = float(weights.get("risk_weight", 0.15))
        wu = float(weights.get("urgency_weight", 0.05))
        wa = float(weights.get("alignment_weight", 0.05))

        effort_term = 1.0 - goal.effort
        risk_term = 1.0 - goal.risk
        urgency = GoalStore.compute_urgency(goal.created_at, goal.horizon_max_min, now)
        active = set(active_beliefs or [])
        if goal.aligns_with:
            alignment = len([b for b in goal.aligns_with if b in active]) / max(1, len(goal.aligns_with))
        else:
            alignment = 0.0
        contradict_active = any(b in active for b in goal.contradicts)
        penalty = 1.0 if contradict_active else 0.0
        raw = (wv * goal.value + we * effort_term + wr * risk_term + wu * urgency + wa * alignment) - penalty
        return max(0.0, min(1.0, raw))

    def prioritized(
        self,
        state: GoalState = GoalState.PROPOSED,
        limit: int = 50,
        weights: Optional[Dict[str, float]] = None,
        active_beliefs: Optional[Iterable[str]] = None,
    ) -> List[Tuple[GoalDefinition, float]]:
        weights = weights or {}
        goals = self.list_goals(state=state, limit=limit)
        scored = [
            (g, self.score_goal(g, weights=weights, active_beliefs=active_beliefs))
            for g in goals
        ]
        scored.sort(key=lambda t: (-t[1], t[0].id))
        return scored

    # --------- Adoption (safety) ---------
    def adopt_goal(
        self,
        goal_id: str,
        idempotency_key: Optional[str] = None,
        severity_threshold: float = 0.6,
        belief_checker=None,
        aligned_beliefs: Sequence[Any] = (),
        memories: Sequence[Any] = (),
        active_belief_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[bool, Optional[GoalDefinition], Dict[str, Any]]:
        """Attempt to adopt a goal with safety checks.

        Returns (adopted, goal, details)
        details: {blocked_by_belief: bool, belief_ids: List[str], reason: str}
        """
        goal = self.get_goal(goal_id)
        if not goal:
            return False, None, {"reason": "not_found"}

        if idempotency_key and self._lookup_idempotency(idempotency_key):
            return True, goal, {"reason": "idempotent"}

        # Contradiction against active beliefs is hard block
        active_ids = set(active_belief_ids or [])
        contradict_hits = [b for b in goal.contradicts if b in active_ids]
        if contradict_hits:
            self._emit_blocked(goal_id, contradict_hits)
            return False, goal, {
                "blocked_by_belief": True,
                "belief_ids": contradict_hits,
                "reason": "contradiction"
            }

        # Optional semantic check via checker
        if belief_checker:
            try:
                report = belief_checker.check_consistency(
                    query=goal.text,
                    beliefs=list(aligned_beliefs),
                    memories=list(memories),
                )
                max_sev = 0.0
                belief_ids = []
                for pat in report.dissonance_patterns:
                    max_sev = max(max_sev, float(getattr(pat, "severity", 0.0)))
                    # If the pattern references a belief, we try to collect it from aligned_beliefs
                if max_sev >= severity_threshold:
                    self._emit_blocked(goal_id, belief_ids)
                    return False, goal, {
                        "blocked_by_belief": True,
                        "belief_ids": belief_ids,
                        "reason": f"severity={max_sev:.2f}"
                    }
            except Exception as e:
                logger.error(f"Belief checker failed: {e}")

        # Passed checks → adopt
        now = _utcnow().isoformat()
        self.conn.execute(
            "UPDATE goals SET state=?, updated_at=?, version=version+1 WHERE id=?",
            (GoalState.ADOPTED.value, now, goal_id),
        )
        self.conn.commit()
        if idempotency_key:
            self._insert_idempotency(idempotency_key, op="adopt", entity_id=goal_id)

        append_event(
            LedgerEvent(
                ts=datetime.now(timezone.utc).timestamp(),
                schema=2,
                event="goal_adopted",
                meta={"goal_id": goal_id},
            )
        )
        return True, self.get_goal(goal_id), {}

    def _emit_blocked(self, goal_id: str, belief_ids: List[str]) -> None:
        append_event(
            LedgerEvent(
                ts=datetime.now(timezone.utc).timestamp(),
                schema=2,
                event="goal_blocked_by_belief",
                meta={"goal_id": goal_id, "belief_ids": belief_ids},
            )
        )

    async def execute_goal(
        self,
        goal_id: str,
        code_access_service,
        timeout_ms: int = 600000
    ):
        """Execute an adopted goal using GoalExecutionService.

        Args:
            goal_id: ID of goal to execute
            code_access_service: CodeAccessService instance for file operations
            timeout_ms: Maximum execution time in milliseconds

        Returns:
            GoalExecutionResult with task outcomes and artifacts

        Raises:
            ValueError: If goal doesn't exist or is not ADOPTED
        """
        from src.services.goal_execution_service import GoalExecutionService

        goal = self.get_goal(goal_id)
        if not goal:
            raise ValueError(f"Goal not found: {goal_id}")

        if goal.state != GoalState.ADOPTED:
            raise ValueError(f"Goal must be ADOPTED to execute. Current state: {goal.state}")

        # Update goal state to EXECUTING
        self.update_goal(goal_id, {"state": GoalState.EXECUTING}, expected_version=goal.version)

        # Initialize execution service
        exec_service = GoalExecutionService(
            code_access=code_access_service,
            identity_ledger=None,  # TODO: wire in Phase 4
            workdir=str(code_access_service.project_root),
            max_concurrent=3
        )

        try:
            # Execute goal
            result = await exec_service.execute_goal(
                goal_text=goal.text,
                context=goal.metadata,
                timeout_ms=timeout_ms
            )

            # Update goal state based on result
            final_state = GoalState.SATISFIED if result.success else GoalState.ADOPTED
            updated_goal = self.get_goal(goal_id)
            if updated_goal:
                self.update_goal(
                    goal_id,
                    {
                        "state": final_state,
                        "metadata": {
                            **updated_goal.metadata,
                            "execution_result": {
                                "success": result.success,
                                "completed_tasks": len(result.completed_tasks),
                                "failed_tasks": len(result.failed_tasks),
                                "execution_time_ms": result.execution_time_ms,
                            }
                        }
                    },
                    expected_version=updated_goal.version
                )

            # Emit ledger event
            append_event(
                LedgerEvent(
                    ts=datetime.now(timezone.utc).timestamp(),
                    schema=2,
                    event="goal_executed",
                    meta={
                        "goal_id": goal_id,
                        "success": result.success,
                        "total_tasks": result.total_tasks,
                        "completed_tasks": len(result.completed_tasks),
                        "failed_tasks": len(result.failed_tasks),
                    }
                )
            )

            return result

        except Exception as e:
            # Revert to ADOPTED on error
            updated_goal = self.get_goal(goal_id)
            if updated_goal and updated_goal.state == GoalState.EXECUTING:
                self.update_goal(
                    goal_id,
                    {"state": GoalState.ADOPTED},
                    expected_version=updated_goal.version
                )
            raise


def create_goal_store(db_path: str) -> GoalStore:
    return GoalStore(db_path)


def register_goal_selection_decision(decision_registry) -> None:
    """Register goal_selected as an adaptive decision point.

    This allows the DecisionFramework to learn optimal weights for
    value/effort/risk/urgency/alignment scoring.

    Args:
        decision_registry: DecisionRegistry instance to register with
    """
    from src.services.decision_framework import Parameter

    parameters = {
        "value_weight": Parameter(
            name="value_weight",
            current_value=0.5,
            min_value=0.0,
            max_value=1.0,
            step_size=0.05,
            adaptation_rate=0.1
        ),
        "effort_weight": Parameter(
            name="effort_weight",
            current_value=0.25,
            min_value=0.0,
            max_value=1.0,
            step_size=0.05,
            adaptation_rate=0.1
        ),
        "risk_weight": Parameter(
            name="risk_weight",
            current_value=0.15,
            min_value=0.0,
            max_value=1.0,
            step_size=0.05,
            adaptation_rate=0.1
        ),
        "urgency_weight": Parameter(
            name="urgency_weight",
            current_value=0.05,
            min_value=0.0,
            max_value=0.2,
            step_size=0.01,
            adaptation_rate=0.1
        ),
        "alignment_weight": Parameter(
            name="alignment_weight",
            current_value=0.05,
            min_value=0.0,
            max_value=0.3,
            step_size=0.01,
            adaptation_rate=0.1
        ),
    }

    decision_registry.register_decision(
        decision_id="goal_selected",
        subsystem="goal_store",
        description="Goal prioritization and selection for execution",
        parameters=parameters,
        success_metrics=["coherence_delta", "goal_completion", "satisfaction_score"],
        context_features=["goal_category", "goal_value", "goal_effort", "goal_risk", "active_beliefs"]
    )

    logger.info("Registered goal_selected decision point with adaptive weights")

