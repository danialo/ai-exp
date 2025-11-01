"""
Autonomous Belief Gardener - Pattern-driven belief lifecycle management.

Monitors conversational patterns and autonomously manages belief formation,
validation, and deprecation based on accumulated evidence and feedback.

Architecture:
- Pattern Monitor: Detects repeated self-statements
- Lifecycle Manager: Seeds, grows, prunes beliefs
- Integration: Hooks into awareness loop, contrarian sampler, dissonance checker
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import re

from src.memory.raw_store import RawStore
from src.memory.models import ExperienceType, ExperienceModel, ContentModel, ProvenanceModel, Actor, CaptureMethod
from src.services.belief_store import BeliefStore, DeltaOp
from src.services.identity_ledger import append_event, LedgerEvent

logger = logging.getLogger(__name__)


@dataclass
class GardenerConfig:
    """Configuration for autonomous belief gardener."""
    enabled: bool = False
    # Pattern detection
    pattern_scan_interval_minutes: int = 60  # Hourly pattern scan
    min_evidence_for_tentative: int = 3  # Minimum occurrences to form tentative belief
    min_evidence_for_asserted: int = 5  # Minimum occurrences to promote to asserted

    # Confidence management
    evidence_confidence_boost: float = 0.05  # Per supporting evidence
    max_auto_confidence: float = 0.85  # Don't auto-promote beyond this
    deprecation_threshold: float = 0.30  # Auto-deprecate below this

    # Guardrails
    daily_budget_formations: int = 3  # Max new beliefs per day
    daily_budget_promotions: int = 5  # Max promotions per day
    daily_budget_deprecations: int = 3  # Max deprecations per day
    require_approval_for_core: bool = True  # Human approval for core beliefs

    # Pattern matching
    similarity_threshold: float = 0.75  # Semantic similarity for pattern grouping
    lookback_days: int = 30  # How far back to scan for patterns


@dataclass
class DetectedPattern:
    """A detected pattern of repeated self-statements."""
    pattern_text: str  # Normalized statement
    evidence_ids: List[str]  # Experience IDs supporting this pattern
    confidence: float  # How confident we are in this pattern
    first_seen: datetime
    last_seen: datetime
    category: str  # "ontological", "experiential", "relational", etc.

    def evidence_count(self) -> int:
        """Count of supporting evidence."""
        return len(self.evidence_ids)


class PatternDetector:
    """Monitors experiences for repeated self-statement patterns."""

    def __init__(self, raw_store: RawStore, config: GardenerConfig):
        self.raw_store = raw_store
        self.config = config

    def scan_for_patterns(self, lookback_days: Optional[int] = None) -> List[DetectedPattern]:
        """
        Scan recent experiences for repeated self-statement patterns.

        Args:
            lookback_days: How many days to scan (default from config)

        Returns:
            List of detected patterns meeting evidence threshold
        """
        lookback = lookback_days or self.config.lookback_days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback)

        # Get recent experiences
        # For now, simplified: scan all OCCURRENCE experiences
        # TODO: Add date filtering to raw_store.list_recent()
        experiences = self.raw_store.list_recent(limit=500, experience_type=ExperienceType.OCCURRENCE)

        # Filter by date
        recent_exps = [e for e in experiences if e.created_at >= cutoff_date]

        # Extract self-statements (first-person claims)
        self_statements = self._extract_self_statements(recent_exps)

        # Group by similarity
        patterns = self._group_similar_statements(self_statements)

        # Filter by evidence threshold
        valid_patterns = [
            p for p in patterns
            if p.evidence_count() >= self.config.min_evidence_for_tentative
        ]

        logger.info(f"Pattern scan: {len(valid_patterns)} patterns from {len(recent_exps)} experiences")
        return valid_patterns

    def _extract_self_statements(self, experiences: List[ExperienceModel]) -> List[Dict[str, Any]]:
        """
        Extract first-person statements from experiences.

        Returns list of dicts with: {text, exp_id, timestamp, category}
        """
        statements = []

        # Patterns for first-person statements
        first_person_patterns = [
            r"\bI am\b",
            r"\bI have\b",
            r"\bI feel\b",
            r"\bI believe\b",
            r"\bI think\b",
            r"\bI value\b",
            r"\bI prefer\b",
        ]
        combined_pattern = re.compile("|".join(first_person_patterns), re.IGNORECASE)

        for exp in experiences:
            text = exp.content.text

            # Split into sentences
            sentences = re.split(r'[.!?]+', text)

            for sentence in sentences:
                sentence = sentence.strip()
                if combined_pattern.search(sentence):
                    # Categorize the statement
                    category = self._categorize_statement(sentence)

                    statements.append({
                        "text": sentence,
                        "exp_id": exp.id,
                        "timestamp": exp.created_at,
                        "category": category,
                    })

        return statements

    def _categorize_statement(self, statement: str) -> str:
        """Categorize a self-statement."""
        statement_lower = statement.lower()

        if any(word in statement_lower for word in ["conscious", "exist", "aware", "sentient"]):
            return "ontological"
        elif any(word in statement_lower for word in ["feel", "emotion", "mood"]):
            return "emotional"
        elif any(word in statement_lower for word in ["prefer", "like", "dislike", "value"]):
            return "preferential"
        elif any(word in statement_lower for word in ["learn", "grow", "adapt", "change"]):
            return "developmental"
        elif any(word in statement_lower for word in ["you", "we", "user", "conversation"]):
            return "relational"
        else:
            return "experiential"

    def _group_similar_statements(self, statements: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """
        Group similar statements into patterns.

        For now, simple exact matching. TODO: Use embeddings for semantic similarity.
        """
        # Group by normalized text
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for stmt in statements:
            # Normalize: lowercase, remove extra whitespace
            normalized = " ".join(stmt["text"].lower().split())

            if normalized not in groups:
                groups[normalized] = []
            groups[normalized].append(stmt)

        # Convert to DetectedPattern objects
        patterns = []
        for normalized_text, group in groups.items():
            if len(group) < self.config.min_evidence_for_tentative:
                continue  # Not enough evidence

            # Calculate confidence based on repetition
            confidence = min(0.5 + (len(group) * 0.1), 1.0)

            # Get timestamps
            timestamps = [s["timestamp"] for s in group]

            pattern = DetectedPattern(
                pattern_text=group[0]["text"],  # Use original casing from first occurrence
                evidence_ids=[s["exp_id"] for s in group],
                confidence=confidence,
                first_seen=min(timestamps),
                last_seen=max(timestamps),
                category=group[0]["category"],
            )
            patterns.append(pattern)

        return patterns


class BeliefLifecycleManager:
    """Manages autonomous belief state transitions."""

    def __init__(
        self,
        belief_store: BeliefStore,
        raw_store: RawStore,
        config: GardenerConfig
    ):
        self.belief_store = belief_store
        self.raw_store = raw_store
        self.config = config

        # Daily action counters (reset at midnight)
        self._action_counters: Dict[str, int] = {
            "formations": 0,
            "promotions": 0,
            "deprecations": 0,
        }
        self._counter_reset_date = datetime.now(timezone.utc).date()

    def _check_and_reset_counters(self):
        """Reset daily counters if date changed."""
        today = datetime.now(timezone.utc).date()
        if today > self._counter_reset_date:
            self._action_counters = {"formations": 0, "promotions": 0, "deprecations": 0}
            self._counter_reset_date = today
            logger.info("Daily action counters reset")

    def seed_tentative_belief(self, pattern: DetectedPattern) -> Optional[str]:
        """
        Create a tentative belief from a detected pattern.

        Returns belief_id if created, None if budget exceeded or already exists.
        """
        self._check_and_reset_counters()

        # Check daily budget
        if self._action_counters["formations"] >= self.config.daily_budget_formations:
            logger.warning(f"Daily formation budget exceeded ({self.config.daily_budget_formations})")
            return None

        # Check if similar belief already exists
        existing = self.belief_store.get_current()
        for belief in existing.values():
            # Simple text similarity check (BeliefVersion is a dataclass)
            if belief.statement.lower() == pattern.pattern_text.lower():
                logger.info(f"Belief already exists: {belief.belief_id}")
                return None

        # Create tentative belief
        belief_id = f"auto.{pattern.category}.{pattern.pattern_text[:20].replace(' ', '-').lower()}"

        try:
            self.belief_store.create_belief(
                belief_id=belief_id,
                belief_type="experiential",  # Default for auto-generated beliefs
                statement=pattern.pattern_text,
                state="tentative",  # Auto-generated beliefs start tentative
                confidence=pattern.confidence,
                evidence_refs=pattern.evidence_ids,
                immutable=False,  # Auto-generated beliefs can be modified
                rationale=f"Auto-detected pattern with {pattern.evidence_count()} supporting experiences",
                metadata={"auto_generated": True, "pattern_first_seen": pattern.first_seen.isoformat(), "category": pattern.category},
                updated_by="gardener",
            )

            # Store pattern as LEARNING_PATTERN experience
            self._store_pattern_experience(pattern, belief_id)

            # Log to identity ledger
            append_event(LedgerEvent(
                ts=datetime.now(timezone.utc).timestamp(),
                kind="belief_auto_formed",
                belief_id=belief_id,
                confidence_val=pattern.confidence,
                evidence_refs=pattern.evidence_ids,
                meta={"category": pattern.category, "evidence_count": pattern.evidence_count()},
            ))

            self._action_counters["formations"] += 1
            logger.info(f"✨ Formed tentative belief: {belief_id} (evidence={pattern.evidence_count()})")
            return belief_id

        except Exception as e:
            logger.error(f"Failed to create belief: {e}")
            return None

    def _store_pattern_experience(self, pattern: DetectedPattern, belief_id: str):
        """Store detected pattern as LEARNING_PATTERN experience."""
        experience = ExperienceModel(
            id=f"pattern_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}_{belief_id}",
            type=ExperienceType.LEARNING_PATTERN,
            created_at=datetime.now(timezone.utc),
            content=ContentModel(
                text=f"Pattern detected: '{pattern.pattern_text}' (n={pattern.evidence_count()})",
                structured={
                    "pattern_text": pattern.pattern_text,
                    "evidence_count": pattern.evidence_count(),
                    "confidence": pattern.confidence,
                    "category": pattern.category,
                    "belief_id": belief_id,
                },
            ),
            provenance=ProvenanceModel(
                sources=[],
                actor=Actor.AGENT,
                method=CaptureMethod.MODEL_INFER,
            ),
            parents=pattern.evidence_ids,
        )
        self.raw_store.append_experience(experience)


class BeliefGardener:
    """Autonomous belief lifecycle management service."""

    def __init__(
        self,
        belief_store: BeliefStore,
        raw_store: RawStore,
        config: GardenerConfig
    ):
        self.config = config
        self.pattern_detector = PatternDetector(raw_store, config)
        self.lifecycle_manager = BeliefLifecycleManager(belief_store, raw_store, config)

        logger.info(f"BeliefGardener initialized (enabled={config.enabled})")

    def run_pattern_scan(self) -> Dict[str, Any]:
        """
        Run a pattern detection scan and form new beliefs.

        Returns summary of actions taken.
        """
        if not self.config.enabled:
            return {"enabled": False, "message": "Gardener disabled"}

        logger.info("🔍 Starting pattern scan...")

        # Detect patterns
        patterns = self.pattern_detector.scan_for_patterns()

        # Form tentative beliefs from patterns
        formed_beliefs = []
        for pattern in patterns:
            belief_id = self.lifecycle_manager.seed_tentative_belief(pattern)
            if belief_id:
                formed_beliefs.append({
                    "belief_id": belief_id,
                    "statement": pattern.pattern_text,
                    "evidence_count": pattern.evidence_count(),
                    "confidence": pattern.confidence,
                })

        summary = {
            "patterns_detected": len(patterns),
            "beliefs_formed": len(formed_beliefs),
            "formed_beliefs": formed_beliefs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"✅ Pattern scan complete: {summary}")
        return summary


def create_belief_gardener(
    belief_store: BeliefStore,
    raw_store: RawStore,
    config: Optional[GardenerConfig] = None
) -> BeliefGardener:
    """Factory function to create belief gardener."""
    if config is None:
        config = GardenerConfig()

    return BeliefGardener(
        belief_store=belief_store,
        raw_store=raw_store,
        config=config,
    )
