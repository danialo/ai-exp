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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
from threading import Lock
import re
import hashlib

from src.memory.raw_store import RawStore
from src.memory.models import ExperienceType, ExperienceModel, ContentModel, ProvenanceModel, Actor, CaptureMethod
from src.services.belief_store import BeliefStore, DeltaOp
from src.services.identity_ledger import append_event, LedgerEvent
from src.services.feedback_aggregator import FeedbackAggregator, FeedbackConfig

logger = logging.getLogger(__name__)


# Template noise filters
_TEMPLATE_NOISE_PATTERNS = [
    re.compile(r"^\[?internal emotional assessment[:\]]", re.I),
    re.compile(r"^\[?internal (state|note)[:\]]", re.I),
    re.compile(r"^\(system\)|^\[system\]", re.I),
]

_WS_RE = re.compile(r"\s+")


def _looks_like_template_noise(text: str) -> bool:
    """Check if text matches template/boilerplate patterns."""
    t = text.strip().lower()
    return any(p.search(t) for p in _TEMPLATE_NOISE_PATTERNS)


def _normalize_statement(s: str) -> str:
    """Normalize statement for comparison."""
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    # remove enclosing quotes and trailing punctuation noise
    s = s.strip('"\'""''').rstrip(" .,:;")
    return s


def _safe_belief_id(category: str, statement: str) -> str:
    """Generate safe, unique belief ID with hash suffix."""
    cat = re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-") or "misc"
    norm = _normalize_statement(statement)
    slug = re.sub(r"[^a-z0-9]+", "-", norm)[:40].strip("-")
    h = hashlib.blake2b(norm.encode("utf-8"), digest_size=6).hexdigest()
    return f"auto.{cat}.{slug}-{h}"


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

        # Debounce: ignore repeating boilerplate within session window
        deduped_patterns = self._debounce_patterns(valid_patterns)

        logger.info(f"Pattern scan: {len(deduped_patterns)} patterns from {len(recent_exps)} experiences (deduped from {len(valid_patterns)})")
        return deduped_patterns

    def _debounce_patterns(self, patterns: List[DetectedPattern]) -> List[DetectedPattern]:
        """Deduplicate patterns within scan window to prevent repeat detection."""
        window_keys: Set[str] = set()
        deduped: List[DetectedPattern] = []

        for p in patterns:
            key = _normalize_statement(p.pattern_text)
            if key in window_keys:
                continue
            window_keys.add(key)
            deduped.append(p)

        return deduped

    def estimate_new_evidence_for(self, belief) -> int:
        """
        Estimate new evidence count for an existing belief.

        Compares current evidence refs to belief's stored evidence_refs.

        Args:
            belief: BeliefVersion object with evidence_refs

        Returns:
            Count of new evidence items (delta)
        """
        # Get current evidence count from belief
        stored_evidence = set(belief.evidence_refs) if belief.evidence_refs else set()

        # Scan for patterns matching this belief's statement
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.lookback_days)
        experiences = self.raw_store.list_recent(limit=500, experience_type=ExperienceType.OCCURRENCE)
        recent_exps = [e for e in experiences if e.created_at >= cutoff_date]

        # Extract statements and count matches
        normalized_belief = _normalize_statement(belief.statement)
        current_evidence = set()

        for exp in recent_exps:
            text = exp.content.text if hasattr(exp.content, 'text') else ""
            sentences = re.split(r'[.!?]+', text)

            for sentence in sentences:
                if _normalize_statement(sentence) == normalized_belief:
                    current_evidence.add(exp.id)

        # Return delta
        new_evidence = current_evidence - stored_evidence
        return len(new_evidence)

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
        config: GardenerConfig,
        feedback_aggregator: Optional[FeedbackAggregator] = None
    ):
        self.belief_store = belief_store
        self.raw_store = raw_store
        self.config = config
        self.feedback_aggregator = feedback_aggregator

        # Daily action counters (reset at midnight)
        self._action_counters: Dict[str, int] = {
            "formations": 0,
            "promotions": 0,
            "deprecations": 0,
        }
        self._counter_reset_date = datetime.now(timezone.utc).date()

        # Idempotency: prevent double creation within short window
        self._recent_creations: Dict[str, float] = {}  # normalized_statement -> timestamp
        self._creation_window_secs = 300  # 5 minute deduplication window

    def _check_and_reset_counters(self):
        """Reset daily counters if date changed."""
        today = datetime.now(timezone.utc).date()
        if today > self._counter_reset_date:
            self._action_counters = {"formations": 0, "promotions": 0, "deprecations": 0}
            self._counter_reset_date = today
            logger.info("Daily action counters reset")

    def seed_tentative_belief(self, pattern: DetectedPattern) -> Tuple[Optional[str], Optional[str]]:
        """
        Create a tentative belief from a detected pattern.

        Returns (belief_id, skip_reason).
        - If created: (belief_id, None)
        - If skipped: (None, skip_reason)
        """
        self._check_and_reset_counters()

        # Check daily budget
        if self._action_counters["formations"] >= self.config.daily_budget_formations:
            logger.warning(f"Daily formation budget exceeded ({self.config.daily_budget_formations})")
            return None, "budget_exceeded"

        # Filter template noise
        if _looks_like_template_noise(pattern.pattern_text):
            logger.debug(f"Skipping template noise: {pattern.pattern_text[:50]}")
            return None, "template_noise"

        # Normalize for duplicate checking
        normalized = _normalize_statement(pattern.pattern_text)

        # Idempotency check: prevent double creation within window
        now = time.time()
        if normalized in self._recent_creations:
            age = now - self._recent_creations[normalized]
            if age < self._creation_window_secs:
                logger.debug(f"Skipping recent idempotent creation (age={age:.1f}s)")
                return None, "idempotent_recent"

        # Check for existing beliefs
        existing = self.belief_store.get_current()
        for belief in existing.values():
            if _normalize_statement(belief.statement) == normalized:
                logger.info(f"Belief already exists: {belief.belief_id}")
                return None, "duplicate_existing"

        # Generate safe, unique belief ID
        belief_id = _safe_belief_id(pattern.category, pattern.pattern_text)

        try:
            self.belief_store.create_belief(
                belief_id=belief_id,
                belief_type="experiential",
                statement=pattern.pattern_text,
                state="tentative",
                confidence=pattern.confidence,
                evidence_refs=pattern.evidence_ids,
                immutable=False,
                rationale=f"Auto-detected pattern with {pattern.evidence_count()} supporting experiences",
                metadata={
                    "auto_generated": True,
                    "pattern_first_seen": pattern.first_seen.isoformat(),
                    "category": pattern.category,
                },
                updated_by="gardener",
            )

            # Store pattern as LEARNING_PATTERN experience
            self._store_pattern_experience(pattern, belief_id)

            # Log to identity ledger
            append_event(LedgerEvent(
                ts=datetime.now(timezone.utc).timestamp(),
                schema=2,
                event="belief_auto_formed",
                beliefs_touched=[belief_id],
                evidence_refs=pattern.evidence_ids,
                meta={
                    "category": pattern.category,
                    "evidence_count": pattern.evidence_count(),
                    "confidence": pattern.confidence,
                    "normalized_statement": normalized,
                },
            ))

            self._action_counters["formations"] += 1

            # Track for idempotency
            self._recent_creations[normalized] = now

            logger.info(f"✨ Formed tentative belief: {belief_id} (evidence={pattern.evidence_count()})")
            return belief_id, None

        except Exception as e:
            logger.error(f"Failed to create belief: {e}")
            return None, "store_rejected"

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

    def consider_promotion(self, belief_id: str, new_evidence: int) -> bool:
        """
        Consider promoting a tentative belief to asserted.

        Uses feedback aggregator to determine if belief has positive outcomes.

        Args:
            belief_id: ID of belief to consider
            new_evidence: Number of new supporting evidence items

        Returns:
            True if promoted, False otherwise
        """
        beliefs = self.belief_store.get_current()
        if belief_id not in beliefs:
            return False

        belief = beliefs[belief_id]

        # Check global circuit breaker
        if self.feedback_aggregator and self.feedback_aggregator.circuit_open:
            logger.info(f"Skip promotion {belief_id}: feedback circuit breaker open")
            return False

        # Only promote tentative beliefs
        if belief.state != "tentative":
            return False

        # Require evidence threshold
        if new_evidence < self.config.min_evidence_for_asserted:
            return False

        # Get feedback score from aggregator
        if self.feedback_aggregator:
            feedback_score, _ = self.feedback_aggregator.score(belief_id)
        else:
            # No aggregator available - use neutral score
            logger.warning(f"No feedback aggregator for {belief_id}, using neutral score")
            feedback_score = 0.0

        # Require positive external signal
        if feedback_score < 0.2:
            logger.debug(f"Skip promotion {belief_id}: feedback too low ({feedback_score:.2f})")
            return False

        return self._promote(belief)

    def consider_deprecation(self, belief_id: str, decay_evidence: int) -> bool:
        """
        Consider deprecating a belief based on evidence decay or negative feedback.

        Uses feedback aggregator to determine if belief has negative outcomes.

        Args:
            belief_id: ID of belief to consider
            decay_evidence: Change in evidence count (negative = decay)

        Returns:
            True if deprecated, False otherwise
        """
        beliefs = self.belief_store.get_current()
        if belief_id not in beliefs:
            return False

        belief = beliefs[belief_id]

        # Only deprecate asserted or tentative beliefs
        if belief.state not in ("asserted", "tentative"):
            return False

        # Get negative feedback score from aggregator
        if self.feedback_aggregator:
            _, neg_feedback, _ = self.feedback_aggregator.score(belief_id)
        else:
            # No aggregator available - use neutral score
            logger.warning(f"No feedback aggregator for {belief_id}, using neutral score")
            neg_feedback = 0.0

        # Deprecate on strong negative feedback or evidence decay
        if neg_feedback >= 0.4 or decay_evidence <= 0:
            logger.info(f"Deprecating {belief_id}: neg_feedback={neg_feedback:.2f}, decay={decay_evidence}")
            return self._deprecate(belief)

        return False

    def _promote(self, belief) -> bool:
        """Promote a belief from tentative to asserted."""
        self._check_and_reset_counters()

        if self._action_counters["promotions"] >= self.config.daily_budget_promotions:
            logger.info(f"Skip promotion: budget exceeded ({self.config.daily_budget_promotions})")
            return False

        try:
            self.belief_store.update_belief(
                belief_id=belief.belief_id,
                state="asserted",
                confidence=min(0.85, belief.confidence + 0.1),
                rationale="Auto-promotion: evidence threshold met and positive feedback",
                updated_by="gardener",
            )

            append_event(LedgerEvent(
                ts=datetime.now(timezone.utc).timestamp(),
                schema=2,
                event="belief_auto_promoted",
                beliefs_touched=[belief.belief_id],
                meta={"prev_state": "tentative", "new_state": "asserted"}
            ))

            self._action_counters["promotions"] += 1

            # Get feedback score for logging
            score = 0.0
            if self.feedback_aggregator:
                score, _ = self.feedback_aggregator.score(belief.belief_id)

            logger.info(f"PROMOTE belief_id={belief.belief_id[:40]} score={score:.2f} "
                       f"statement=\"{belief.statement[:60]}...\" reason=\"thresholds_met\"")
            return True

        except Exception as e:
            logger.error(f"Failed to promote belief {belief.belief_id}: {e}")
            return False

    def _deprecate(self, belief) -> bool:
        """Deprecate a belief back to tentative state."""
        self._check_and_reset_counters()

        if self._action_counters["deprecations"] >= self.config.daily_budget_deprecations:
            logger.info(f"Skip deprecation: budget exceeded ({self.config.daily_budget_deprecations})")
            return False

        try:
            self.belief_store.update_belief(
                belief_id=belief.belief_id,
                state="tentative",
                confidence=max(0.2, belief.confidence - 0.2),
                rationale="Auto-deprecation: evidence decay or negative feedback",
                updated_by="gardener",
            )

            append_event(LedgerEvent(
                ts=datetime.now(timezone.utc).timestamp(),
                schema=2,
                event="belief_auto_deprecated",
                beliefs_touched=[belief.belief_id],
                meta={"reason": "decay_or_negative_feedback", "prev_state": belief.state}
            ))

            self._action_counters["deprecations"] += 1
            logger.info(f"⬇️ Deprecated belief: {belief.belief_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to deprecate belief {belief.belief_id}: {e}")
            return False


class BeliefGardener:
    """Autonomous belief lifecycle management service."""

    def __init__(
        self,
        belief_store: BeliefStore,
        raw_store: RawStore,
        config: GardenerConfig,
        feedback_aggregator=None  # Optional: use enhanced version if provided
    ):
        self.config = config
        self.pattern_detector = PatternDetector(raw_store, config)

        # Initialize feedback aggregator (use provided or create default)
        if feedback_aggregator:
            self.feedback_aggregator = feedback_aggregator
            logger.info("Using provided feedback aggregator (likely enhanced)")
        else:
            self.feedback_aggregator = FeedbackAggregator(
                raw_store=raw_store,
                config=FeedbackConfig(enabled=True)
            )
            logger.info("Using default FeedbackAggregator")

        # Initialize lifecycle manager with feedback aggregator
        self.lifecycle_manager = BeliefLifecycleManager(
            belief_store=belief_store,
            raw_store=raw_store,
            config=config,
            feedback_aggregator=self.feedback_aggregator
        )

        # Reentrancy lock to prevent concurrent scans
        self._scan_lock = Lock()

        # Telemetry tracking
        self.last_scan_ts: Optional[float] = None
        self.skips_since_boot: Dict[str, int] = {
            "budget_exceeded": 0,
            "template_noise": 0,
            "duplicate_existing": 0,
            "idempotent_recent": 0,
            "store_rejected": 0,
        }

        logger.info(f"BeliefGardener initialized (enabled={config.enabled}, feedback_enabled=True)")

    def run_pattern_scan(self) -> Dict[str, Any]:
        """
        Run a pattern detection scan and form new beliefs.

        Returns summary of actions taken.
        """
        if not self.config.enabled:
            return {"enabled": False, "message": "Gardener disabled"}

        # Reentrancy guard: prevent concurrent scans
        if not self._scan_lock.acquire(blocking=False):
            return {"enabled": True, "message": "scan_in_progress"}

        try:
            logger.info("🔍 Starting pattern scan...")

            # Detect patterns
            patterns = self.pattern_detector.scan_for_patterns()

            # Form tentative beliefs from patterns
            formed_beliefs = []
            skipped = []
            for pattern in patterns:
                belief_id, skip_reason = self.lifecycle_manager.seed_tentative_belief(pattern)
                if belief_id:
                    formed_beliefs.append({
                        "belief_id": belief_id,
                        "statement": pattern.pattern_text,
                        "evidence_count": pattern.evidence_count(),
                        "confidence": pattern.confidence,
                    })
                elif skip_reason:
                    # Track telemetry
                    if skip_reason in self.skips_since_boot:
                        self.skips_since_boot[skip_reason] += 1

                    skipped.append({
                        "statement": pattern.pattern_text[:80],
                        "reason": skip_reason,
                        "evidence_count": pattern.evidence_count(),
                    })

            # Update last scan timestamp
            self.last_scan_ts = time.time()

            # Periodic promotion/deprecation decisions
            promoted = []
            deprecated = []
            curr_beliefs = self.lifecycle_manager.belief_store.get_current().values()

            for belief in curr_beliefs:
                # Skip immutable beliefs
                if belief.immutable:
                    continue

                # Skip auto-generated beliefs that were just formed (give them time)
                if belief.metadata and belief.metadata.get("auto_generated"):
                    belief_age_secs = time.time() - belief.ts
                    if belief_age_secs < 3600:  # Less than 1 hour old
                        continue

                # Estimate new evidence
                ev_delta = self.pattern_detector.estimate_new_evidence_for(belief) or 0

                # Consider promotion
                if belief.state == "tentative" and self.lifecycle_manager.consider_promotion(belief.belief_id, ev_delta):
                    promoted.append(belief.belief_id)

                # Consider deprecation (decay evidence = 0 for now, rely on feedback)
                if belief.state in ("asserted", "tentative") and self.lifecycle_manager.consider_deprecation(belief.belief_id, decay_evidence=0):
                    deprecated.append(belief.belief_id)

            summary = {
                "patterns_detected": len(patterns),
                "beliefs_formed": len(formed_beliefs),
                "formed_beliefs": formed_beliefs,
                "skipped": skipped[:20],  # Limit to 20 items
                "promoted": promoted,
                "deprecated": deprecated,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(f"✅ Pattern scan complete: {summary}")
            return summary

        finally:
            self._scan_lock.release()


def create_belief_gardener(
    belief_store: BeliefStore,
    raw_store: RawStore,
    config: Optional[GardenerConfig] = None,
    feedback_aggregator=None  # Optional: use enhanced version if provided
) -> BeliefGardener:
    """Factory function to create belief gardener."""
    if config is None:
        config = GardenerConfig()

    return BeliefGardener(
        belief_store=belief_store,
        raw_store=raw_store,
        config=config,
        feedback_aggregator=feedback_aggregator,
    )
