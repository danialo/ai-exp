"""
Adaptive Decision Framework - Meta-system for coordinating and learning decision parameters.

Provides:
- Decision point registry
- Decision outcome tracking
- Parameter adaptation based on outcomes
- Abort condition monitoring
- Success signal evaluation

This transforms hardcoded thresholds into adaptive, outcome-driven parameters.
"""

import logging
import time
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Literal, Tuple
from pathlib import Path
import sqlite3
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class Parameter:
    """A tunable parameter for a decision."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float = 0.05
    adaptation_rate: float = 0.1  # How quickly to adapt (0-1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Parameter":
        return cls(**data)


@dataclass
class DecisionPoint:
    """A registered decision point in the system."""
    decision_id: str
    subsystem: str
    description: str
    parameters: Dict[str, Parameter]
    success_metrics: List[str]
    context_features: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["parameters"] = {k: v.to_dict() for k, v in self.parameters.items()}
        data["created_at"] = self.created_at.isoformat()
        return data


@dataclass
class DecisionOutcome:
    """Outcome of a decision after evaluation period."""
    decision_record_id: str
    success_score: float  # [-1, 1]
    coherence_delta: float
    dissonance_delta: float
    satisfaction_delta: float
    aborted: bool
    abort_reason: Optional[str] = None
    evaluation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["evaluation_timestamp"] = self.evaluation_timestamp.isoformat()
        return data


@dataclass
class ConversationContext:
    """Context classification for parameter selection."""
    stakes: Literal["high", "medium", "low"] = "medium"
    engagement: Literal["deep", "moderate", "casual"] = "moderate"
    system_confidence: Literal["high", "medium", "low"] = "medium"
    exploration_mode: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionRegistry:
    """Registry of all decision points across Astra's subsystems."""

    def __init__(self, db_path: str = "data/decision_registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS decision_registry (
                    decision_id TEXT PRIMARY KEY,
                    subsystem TEXT NOT NULL,
                    description TEXT,
                    parameters JSON,
                    success_metrics JSON,
                    context_features JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS decision_history (
                    record_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    context JSON,
                    parameters_used JSON,
                    outcome_snapshot JSON,
                    evaluated BOOLEAN DEFAULT FALSE,
                    evaluation_timestamp TIMESTAMP,
                    success_score REAL,
                    outcome_details JSON,
                    FOREIGN KEY (decision_id) REFERENCES decision_registry(decision_id)
                );

                CREATE TABLE IF NOT EXISTS parameter_adaptations (
                    adaptation_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    param_name TEXT NOT NULL,
                    old_value REAL NOT NULL,
                    new_value REAL NOT NULL,
                    reason TEXT,
                    based_on_records JSON,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (decision_id) REFERENCES decision_registry(decision_id)
                );

                CREATE INDEX IF NOT EXISTS idx_decision_history_decision_id
                    ON decision_history(decision_id, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_decision_history_evaluated
                    ON decision_history(evaluated, timestamp);

                CREATE INDEX IF NOT EXISTS idx_parameter_adaptations_decision
                    ON parameter_adaptations(decision_id, timestamp DESC);
            """)

    def register_decision(
        self,
        decision_id: str,
        subsystem: str,
        description: str,
        parameters: Dict[str, Parameter],
        success_metrics: List[str],
        context_features: List[str] = None
    ) -> None:
        """Register a new decision point."""
        with self._lock:
            decision = DecisionPoint(
                decision_id=decision_id,
                subsystem=subsystem,
                description=description,
                parameters=parameters,
                success_metrics=success_metrics,
                context_features=context_features or []
            )

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO decision_registry
                    (decision_id, subsystem, description, parameters, success_metrics, context_features, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id,
                        subsystem,
                        description,
                        json.dumps({k: v.to_dict() for k, v in parameters.items()}),
                        json.dumps(success_metrics),
                        json.dumps(context_features or []),
                        datetime.now(timezone.utc).isoformat()
                    )
                )
            logger.info(f"Registered decision point: {decision_id} ({subsystem})")

    def get_parameter(self, decision_id: str, param_name: str) -> Optional[float]:
        """Get current value for a parameter."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT parameters FROM decision_registry WHERE decision_id = ?",
                (decision_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            params = json.loads(row[0])
            if param_name not in params:
                return None

            return params[param_name]["current_value"]

    def get_all_parameters(self, decision_id: str) -> Optional[Dict[str, float]]:
        """Get all current parameter values for a decision."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT parameters FROM decision_registry WHERE decision_id = ?",
                (decision_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            params = json.loads(row[0])
            return {name: p["current_value"] for name, p in params.items()}

    def record_decision(
        self,
        decision_id: str,
        context: Dict[str, Any],
        parameters_used: Dict[str, float],
        outcome_snapshot: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Record a decision for later evaluation.

        Returns:
            record_id for tracking this decision
        """
        with self._lock:
            timestamp = datetime.now(timezone.utc)
            record_id = self._generate_record_id(decision_id, timestamp)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO decision_history
                    (record_id, decision_id, timestamp, context, parameters_used, outcome_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id,
                        decision_id,
                        timestamp.isoformat(),
                        json.dumps(context),
                        json.dumps(parameters_used),
                        json.dumps(outcome_snapshot or {})
                    )
                )

            logger.debug(f"Recorded decision: {record_id} ({decision_id})")
            return record_id

    def update_decision_outcome(
        self,
        record_id: str,
        outcome: DecisionOutcome
    ) -> None:
        """Update a decision record with its evaluated outcome."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE decision_history
                    SET evaluated = TRUE,
                        evaluation_timestamp = ?,
                        success_score = ?,
                        outcome_details = ?
                    WHERE record_id = ?
                    """,
                    (
                        outcome.evaluation_timestamp.isoformat(),
                        outcome.success_score,
                        json.dumps(outcome.to_dict()),
                        record_id
                    )
                )

    def update_parameter(
        self,
        decision_id: str,
        param_name: str,
        new_value: float,
        reason: str = "manual_adjustment",
        based_on_records: List[str] = None
    ) -> bool:
        """
        Update a parameter value.

        Returns:
            True if updated, False if parameter doesn't exist or value out of bounds
        """
        with self._lock:
            # Get current parameters
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT parameters FROM decision_registry WHERE decision_id = ?",
                    (decision_id,)
                )
                row = cursor.fetchone()
                if not row:
                    logger.error(f"Decision {decision_id} not found")
                    return False

                params = json.loads(row[0])
                if param_name not in params:
                    logger.error(f"Parameter {param_name} not found in {decision_id}")
                    return False

                param = params[param_name]
                old_value = param["current_value"]

                # Validate bounds
                if not (param["min_value"] <= new_value <= param["max_value"]):
                    logger.warning(
                        f"Value {new_value} out of bounds [{param['min_value']}, {param['max_value']}] "
                        f"for {param_name}"
                    )
                    # Clamp to bounds
                    new_value = max(param["min_value"], min(param["max_value"], new_value))

                # Update parameter
                param["current_value"] = new_value
                params[param_name] = param

                # Save to registry
                conn.execute(
                    "UPDATE decision_registry SET parameters = ?, updated_at = ? WHERE decision_id = ?",
                    (json.dumps(params), datetime.now(timezone.utc).isoformat(), decision_id)
                )

                # Log adaptation
                adaptation_id = f"adapt_{decision_id}_{param_name}_{int(time.time())}"
                conn.execute(
                    """
                    INSERT INTO parameter_adaptations
                    (adaptation_id, decision_id, param_name, old_value, new_value, reason, based_on_records)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        adaptation_id,
                        decision_id,
                        param_name,
                        old_value,
                        new_value,
                        reason,
                        json.dumps(based_on_records or [])
                    )
                )

            logger.info(
                f"Updated {decision_id}.{param_name}: {old_value:.3f} → {new_value:.3f} ({reason})"
            )
            return True

    def get_recent_decisions(
        self,
        decision_id: str,
        limit: int = 20,
        evaluated_only: bool = False
    ) -> List[dict]:
        """Get recent decisions for a decision point."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT record_id, timestamp, context, parameters_used,
                       evaluated, success_score, outcome_details
                FROM decision_history
                WHERE decision_id = ?
            """
            if evaluated_only:
                query += " AND evaluated = TRUE"
            query += " ORDER BY timestamp DESC LIMIT ?"

            cursor = conn.execute(query, (decision_id, limit))
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "record_id": row[0],
                    "timestamp": row[1],
                    "context": json.loads(row[2]),
                    "parameters_used": json.loads(row[3]),
                    "evaluated": bool(row[4]),
                    "success_score": row[5],
                    "outcome_details": json.loads(row[6]) if row[6] else None
                })

            return results

    def get_unevaluated_decisions(
        self,
        decision_id: Optional[str] = None,
        older_than_hours: int = 24
    ) -> List[dict]:
        """Get decisions that need outcome evaluation."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)

        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT record_id, decision_id, timestamp, context, parameters_used, outcome_snapshot
                FROM decision_history
                WHERE evaluated = FALSE
                AND timestamp < ?
            """
            params = [cutoff.isoformat()]

            if decision_id:
                query += " AND decision_id = ?"
                params.append(decision_id)

            query += " ORDER BY timestamp"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "record_id": row[0],
                    "decision_id": row[1],
                    "timestamp": row[2],
                    "context": json.loads(row[3]),
                    "parameters_used": json.loads(row[4]),
                    "outcome_snapshot": json.loads(row[5]) if row[5] else {}
                })

            return results

    def _generate_record_id(self, decision_id: str, timestamp: datetime) -> str:
        """Generate unique record ID."""
        ts_str = timestamp.strftime("%Y%m%d%H%M%S%f")
        hash_input = f"{decision_id}{ts_str}".encode()
        short_hash = hashlib.blake2b(hash_input, digest_size=4).hexdigest()
        return f"dec_{decision_id}_{ts_str}_{short_hash}"

    def get_registry_stats(self) -> dict:
        """Get statistics about the decision registry."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Count registered decisions
            cursor = conn.execute("SELECT COUNT(*) FROM decision_registry")
            stats["total_decision_types"] = cursor.fetchone()[0]

            # Count total decisions made
            cursor = conn.execute("SELECT COUNT(*) FROM decision_history")
            stats["total_decisions_made"] = cursor.fetchone()[0]

            # Count evaluated decisions
            cursor = conn.execute("SELECT COUNT(*) FROM decision_history WHERE evaluated = TRUE")
            stats["evaluated_decisions"] = cursor.fetchone()[0]

            # Count adaptations
            cursor = conn.execute("SELECT COUNT(*) FROM parameter_adaptations")
            stats["total_adaptations"] = cursor.fetchone()[0]

            # Get decision types and their counts
            cursor = conn.execute("""
                SELECT decision_id, COUNT(*) as count
                FROM decision_history
                GROUP BY decision_id
                ORDER BY count DESC
            """)
            stats["decisions_by_type"] = {row[0]: row[1] for row in cursor.fetchall()}

            return stats


# Singleton instance
_decision_registry = None
_registry_lock = Lock()


def get_decision_registry() -> DecisionRegistry:
    """Get or create the global decision registry."""
    global _decision_registry
    if _decision_registry is None:
        with _registry_lock:
            if _decision_registry is None:
                _decision_registry = DecisionRegistry()
    return _decision_registry
