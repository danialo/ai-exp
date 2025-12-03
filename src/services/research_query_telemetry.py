"""Telemetry dataclasses for InvestigateTopic query attempts."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class QueryAttempt:
    """Represents a single search attempt for a topic."""

    query: str
    stage: str  # e.g., raw_llm, validated, sanitized, fallback_X
    result_count: int = 0
    token_overlap_score: float = 0.0
    entity_hit_score: float = 0.0
    authority_domain_score: float = 0.0
    composite_score: float = 0.0
    removed_tokens: List[str] = field(default_factory=list)
    suspect_tokens: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    is_winner: bool = False

    def to_dict(self) -> dict:
        """Convert to plain dict for logging/serialization."""
        return asdict(self)


@dataclass
class QueryTelemetry:
    """Telemetry captured across all search attempts for a task."""

    task_id: str
    session_id: str
    topic: str
    attempts: List[QueryAttempt] = field(default_factory=list)
    winner_index: Optional[int] = None
    total_latency_ms: float = 0.0
    current_event: bool = False

    def add_attempt(self, attempt: QueryAttempt) -> None:
        """Append attempt for later winner selection."""
        self.attempts.append(attempt)

    def to_dict(self) -> dict:
        """Convert to dict for logging."""
        data = asdict(self)
        data["attempts"] = [a.to_dict() for a in self.attempts]
        return data

    def mark_winner(self, index: int) -> None:
        """Mark a particular attempt as the winner."""
        if not (0 <= index < len(self.attempts)):
            return
        for att in self.attempts:
            att.is_winner = False
        self.attempts[index].is_winner = True
        self.winner_index = index
