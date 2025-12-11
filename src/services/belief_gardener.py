"""
Autonomous Belief Gardener - Pattern-driven belief lifecycle management.

Monitors conversational patterns and autonomously manages belief formation,
validation, and deprecation based on accumulated evidence and feedback.

Architecture:
- Pattern Monitor: Detects repeated self-statements
- Lifecycle Manager: Seeds, grows, prunes beliefs
- Integration: Hooks into awareness loop, contrarian sampler, dissonance checker

TODO: Review belief gardener and decide if it's working correctly
Current observations:
- Scans only last 500 experiences (recency-biased, may miss stable long-term patterns)
- Currently rejecting patterns due to coherence drops (0.076 drop from baseline 0.700)
- Detecting template noise ("[Internal Emotional Assessment: I feel...]") as patterns
- Zero beliefs formed in recent runs - is threshold too conservative?
- Need to verify:
  1. Is template noise filtering working correctly?
  2. Should coherence threshold be adjusted? (current: baseline - σ)
  3. Are we detecting real patterns or just response boilerplate?
  4. Should we add stratified sampling beyond just recent 500?
  5. Are the patterns we're detecting actually belief-worthy?
"""

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

# Shared statement validation
from src.utils.statement_validation import (
    canonicalize_statement,
    is_valid_statement,
    validate_statement_with_reason,
    normalize_for_grouping,
    looks_like_template_noise as _looks_like_template_noise,
)
from pathlib import Path
from threading import Lock
import re
import hashlib

from src.memory.raw_store import RawStore
from src.memory.models import ExperienceType, ExperienceModel, ContentModel, ProvenanceModel, Actor, CaptureMethod
from src.services.belief_store import BeliefStore, DeltaOp
from src.services.identity_ledger import append_event, LedgerEvent
from src.memory.embedding import create_embedding_provider
from src.services.feedback_aggregator import FeedbackAggregator, FeedbackConfig

# Stream migration imports
from sqlmodel import Session as SQLModelSession, select
from src.memory.models.belief_node import BeliefNode
from src.memory.models.stream_assignment import StreamAssignment
from src.services.stream_service import StreamService
from src.services.core_score_service import CoreScoreService
from src.utils.belief_config import get_belief_config

logger = logging.getLogger(__name__)


# Template noise filters
_TEMPLATE_NOISE_PATTERNS = [
    re.compile(r"^\[?internal emotional assessment[:\]]", re.I),
    re.compile(r"^\[?internal (state|note)[:\]]", re.I),
    re.compile(r"^\(system\)|^\[system\]", re.I),
    # System role prefixes
    re.compile(r"^(ASSISTANT|USER|SYSTEM):", re.I),
    re.compile(r"^(Response|Answer|Corrected Answer):", re.I),
    # API/technical fragments
    re.compile(r"/api/|/v\d+/|\*\*/api/", re.I),
    re.compile(r"Schema:|Endpoint:|Route:", re.I),
    # System instructions/pledges
    re.compile(r"^Pledge (enforcement|reminder)", re.I),
    re.compile(r"UNKNOWN.*resisting.*urge.*speculate", re.I),
    # Meta-instructions
    re.compile(r"^(When you|If you|You (should|must|will))", re.I),
    re.compile(r"^\*\s+(When|If|You)", re.I),
    # Markdown/formatting artifacts
    re.compile(r"^#{1,6}\s+", re.I),
    re.compile(r"^\*\*[^*]+\*\*:", re.I),
    re.compile(r"^-\s+\*\*[^*]+\*\*:", re.I),  # Catch "- **I Feel:**" patterns
]

_WS_RE = re.compile(r"\s+")


# Legacy _normalize_statement kept for belief ID comparison
# Uses more aggressive normalization than canonical form
def _normalize_statement(s: str) -> str:
    """Normalize statement for comparison (legacy, used for belief ID generation)."""
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
    min_evidence_for_tentative: int = 2  # Minimum occurrences to form tentative belief
    min_evidence_for_asserted: int = 5  # Minimum occurrences to promote to asserted

    # Confidence management
    evidence_confidence_boost: float = 0.05  # Per supporting evidence
    max_auto_confidence: float = 0.85  # Don't auto-promote beyond this
    deprecation_threshold: float = 0.30  # Auto-deprecate below this

    # Guardrails
    daily_budget_formations: int = 100  # Max new beliefs per day
    daily_budget_promotions: int = 100  # Max promotions per day
    daily_budget_deprecations: int = 50  # Max deprecations per day
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

    def scan_for_patterns(self, lookback_days: Optional[int] = None, scan_id: Optional[str] = None) -> tuple[List[DetectedPattern], Dict[str, Any]]:
        """
        Scan recent experiences for repeated self-statement patterns.

        Args:
            lookback_days: How many days to scan (default from config)
            scan_id: Optional correlation ID for this scan (for tracing)

        Returns:
            Tuple of (detected patterns meeting evidence threshold, validation telemetry dict)
        """
        # TODO: Add a second belief gardener loop that goes deeper
        # Current loop: Last 500 experiences (chronological, recency-biased)
        # Deeper loop ideas:
        # - Stratified temporal sampling (100 from last week, 100 from last month, etc.)
        # - Vector similarity clustering across all experiences (find thematic patterns)
        # - Weight by experience type (LEARNING_PATTERN > OCCURRENCE)
        # - Look for long-term stable patterns vs recent spikes
        # - Cross-reference with existing beliefs to find supporting/contradicting evidence

        lookback = lookback_days or self.config.lookback_days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback)

        # Get recent experiences
        # DUAL STRATEGY: Read both regex-extracted AND structured claims

        # Strategy 1: Legacy regex extraction from OCCURRENCE experiences
        recent_exps = self.raw_store.list_recent(
            limit=500,
            experience_type=ExperienceType.OCCURRENCE,
            since=cutoff_date,
        )
        self_statements_regex = self._extract_self_statements(recent_exps)

        # Strategy 2: Read structured claim experiences from BOTH sources
        # - self_claim: from dissonance checker (LLM-extracted during consistency checking)
        # - SELF_DEFINITION: from ingest pipeline (extracted during conversation ingestion)
        self_statements_claims = []

        # Read self_claim experiences (from dissonance checker)
        # NOTE: "self_claim" is not in ExperienceType enum yet, so skip for now
        # TODO: Add "self_claim" to ExperienceType enum or use different type
        # try:
        #     claim_exps = self.raw_store.list_recent(
        #         limit=200,
        #         experience_type="self_claim",
        #         since=cutoff_date,
        #     )
        #     logger.warning(f"[DEBUG] Fetched {len(claim_exps)} self_claim experiences")
        #     claims_from_checker = self._extract_structured_claims(claim_exps)
        #     self_statements_claims.extend(claims_from_checker)
        #     logger.warning(f"[DEBUG] 📊 Extracted {len(claims_from_checker)} statements from dissonance checker")
        # except Exception as e:
        #     logger.warning(f"[DEBUG] Exception loading self_claim: {e}")

        # Read SELF_DEFINITION experiences (from ingest pipeline)
        extraction_skips: Dict[str, int] = {}
        try:
            self_def_exps = self.raw_store.list_recent(
                limit=200,
                experience_type=ExperienceType.SELF_DEFINITION,
                since=cutoff_date,
            )
            logger.warning(f"[DEBUG] Fetched {len(self_def_exps)} SELF_DEFINITION experiences")
            claims_from_ingest, ingest_skips = self._extract_self_definition_claims(self_def_exps)
            self_statements_claims.extend(claims_from_ingest)
            # Merge skip counts
            for reason, count in ingest_skips.items():
                extraction_skips[reason] = extraction_skips.get(reason, 0) + count
            logger.warning(f"[DEBUG] 📊 Extracted {len(claims_from_ingest)} statements from ingest pipeline")
        except Exception as e:
            logger.warning(f"[DEBUG] Exception loading SELF_DEFINITION: {e}")

        # Merge both sources
        all_statements = self_statements_regex + self_statements_claims
        logger.warning(f"[DEBUG] Total statements: regex={len(self_statements_regex)} structured={len(self_statements_claims)} merged={len(all_statements)}")

        # Group by similarity and get validation skip telemetry
        patterns, grouping_telemetry = self._group_similar_statements(all_statements)

        # DEBUG: Log config and grouping results
        logger.warning(
            f"[DEBUG] Config: min_evidence_tentative={self.config.min_evidence_for_tentative} "
            f"min_evidence_asserted={self.config.min_evidence_for_asserted}"
        )
        pattern_sizes = sorted([p.evidence_count() for p in patterns], reverse=True)
        size_histogram = {
            1: sum(1 for s in pattern_sizes if s == 1),
            2: sum(1 for s in pattern_sizes if s == 2),
            3: sum(1 for s in pattern_sizes if s == 3),
        }
        logger.warning(
            f"[DEBUG] Patterns before threshold: total={len(patterns)} "
            f"sizes_top10={pattern_sizes[:10]} histogram={size_histogram}"
        )

        # Extract telemetry components
        validation_skips = grouping_telemetry["skip_counts"]
        accepted_by_source = grouping_telemetry["accepted_by_source"]
        accepted_by_category = grouping_telemetry["accepted_by_category"]
        rejection_samples = grouping_telemetry["rejection_samples"]

        # Merge all skip counts (extraction + validation)
        skip_counts = {**extraction_skips}
        for reason, count in validation_skips.items():
            skip_counts[reason] = skip_counts.get(reason, 0) + count

        # Compute acceptance stats
        statements_total = len(all_statements)
        statements_rejected = sum(skip_counts.values())
        statements_accepted = statements_total - statements_rejected

        # Log skip telemetry (aggregate, not per-statement)
        scan_prefix = f"[scan_id={scan_id}] " if scan_id else ""

        # Log acceptance breakdown
        if accepted_by_source:
            logger.info(f"{scan_prefix}Accepted by source: {dict(accepted_by_source)}")
        if accepted_by_category:
            logger.info(f"{scan_prefix}Accepted by category: {dict(accepted_by_category)}")

        # Log rejection breakdown with samples
        if skip_counts:
            logger.info(f"{scan_prefix}Validation skips: {dict(skip_counts)}")
            for reason, count in skip_counts.items():
                samples = rejection_samples.get(reason, [])
                if samples:
                    logger.info(f"{scan_prefix}  Rejected {count} for '{reason}'. Samples: {samples}")

        logger.info(f"{scan_prefix}Statement validation: total={statements_total}, accepted={statements_accepted}, rejected={statements_rejected}")

        # Filter by evidence threshold
        valid_patterns = [
            p for p in patterns
            if p.evidence_count() >= self.config.min_evidence_for_tentative
        ]

        # Debounce: ignore repeating boilerplate within session window
        deduped_patterns = self._debounce_patterns(valid_patterns)

        logger.info(f"{scan_prefix}Pattern scan: {len(deduped_patterns)} patterns from {len(recent_exps)} experiences (deduped from {len(valid_patterns)})")

        # Return patterns and telemetry
        telemetry = {
            "scan_id": scan_id,  # Include for correlation
            "validation_skips": skip_counts,
            "accepted_by_source": accepted_by_source,  # NEW
            "accepted_by_category": accepted_by_category,  # NEW
            "rejection_samples": rejection_samples,  # NEW
            "statements_total": statements_total,
            "statements_accepted": statements_accepted,
            "statements_rejected": statements_rejected,
        }
        return deduped_patterns, telemetry

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
        recent_exps = self.raw_store.list_recent(
            limit=500,
            experience_type=ExperienceType.OCCURRENCE,
            since=cutoff_date,
        )

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
        # Use broad pattern to catch all "I [verb]" statements, then filter noise
        first_person_patterns = [
            r"\bI am\b",
            r"\bI have\b",
            r"\bI feel\b",
            r"\bI believe\b",
            r"\bI think\b",
            r"\bI value\b",
            r"\bI prefer\b",
            r"\bI find\b",          # "I find myself..."
            r"\bI recognize\b",     # "I recognize that..."
            r"\bI aim\b",           # "I aim to..."
            r"\bI appreciate\b",    # "I appreciate..."
            r"\bI acknowledge\b",   # "I acknowledge..."
            r"\bI understand\b",    # "I understand..."
            r"\bI see\b",           # "I see this as..."
            r"\bI consider\b",      # "I consider..."
            r"\bI view\b",          # "I view X as..."
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

    def _extract_structured_claims(self, claim_experiences: List[ExperienceModel]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Extract structured claims from self_claim experiences.

        These are already extracted by the dissonance checker, so no regex needed.
        Just convert to the format expected by _group_similar_statements.

        Args:
            claim_experiences: Experiences of type 'self_claim'

        Returns:
            Tuple of (statement dicts, skip counts by reason)
        """
        statements = []
        skip_counts: Dict[str, int] = {}

        for exp in claim_experiences:
            # Get structured data
            structured = exp.content.structured if exp.content.structured else {}
            if not structured:
                continue

            statement_text = structured.get("statement", "")
            if not statement_text:
                continue

            # Filter template noise (same logic as regex extraction)
            if _looks_like_template_noise(statement_text):
                skip_counts["template_noise"] = skip_counts.get("template_noise", 0) + 1
                continue

            # Map dissonance checker's categories to gardener's categories
            # Source: 'self' (agent said it) or 'external' (someone told agent)
            # For belief formation, we only want self-authored claims
            source = structured.get("source", "self")
            if source != "self":
                continue  # Skip external attributions

            # Categorize based on content (reuse existing logic)
            category = self._categorize_statement(statement_text)

            # Confidence: 'certain', 'uncertain', 'hedging'
            # Map to numeric for sorting/filtering later if needed
            confidence_str = structured.get("confidence", "uncertain")

            statements.append({
                "text": statement_text,
                "exp_id": exp.id,
                "timestamp": exp.created_at,
                "category": category,
                "source": "dissonance_checker",  # Track provenance
                "confidence_str": confidence_str,
            })

        return statements, skip_counts

    def _extract_self_definition_claims(self, self_def_experiences: List[ExperienceModel]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Extract claims from SELF_DEFINITION experiences (from ingest pipeline).

        These are extracted during conversation ingestion and stored with structured metadata.

        Args:
            self_def_experiences: Experiences of type SELF_DEFINITION

        Returns:
            Tuple of (statement dicts, skip counts by reason)
        """
        statements = []
        skip_counts: Dict[str, int] = {}

        for exp in self_def_experiences:
            # SELF_DEFINITION experiences store the claim in content.text
            text = exp.content.text if exp.content.text else ""
            if not text:
                continue

            # Filter template noise
            if _looks_like_template_noise(text):
                skip_counts["template_noise"] = skip_counts.get("template_noise", 0) + 1
                continue

            # Get structured metadata if available
            structured = exp.content.structured if exp.content.structured else {}
            trait_type = structured.get("trait_type", "experiential")

            # Map trait_type to category
            category_map = {
                "beliefs": "experiential",
                "emotions": "emotional",
                "values": "preferential",
                "skills": "capability",
            }
            category = category_map.get(trait_type, self._categorize_statement(text))

            # Extract REAL provenance from experience metadata
            # This describes origin (claim_extractor), not route (ingest pipeline)
            validation_source = structured.get("validation_source", None)

            statements.append({
                "text": text,
                "exp_id": exp.id,
                "timestamp": exp.created_at,
                "category": category,
                "source": validation_source,  # Real provenance: where claim was validated
            })

        return statements, skip_counts

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

    def _group_similar_statements(self, statements: List[Dict[str, Any]]) -> tuple[List[DetectedPattern], Dict[str, Any]]:
        """
        Group similar statements into patterns.

        For now, simple exact matching. TODO: Use embeddings for semantic similarity.

        Returns:
            (patterns, telemetry_dict) where telemetry includes skip_counts, accepted_by_source, accepted_by_category
        """
        # Group by normalized text
        groups: Dict[str, List[Dict[str, Any]]] = {}
        skip_counts: Dict[str, int] = {}
        accepted_by_source: Dict[str, int] = {}  # NEW: Track acceptance by source
        accepted_by_category: Dict[str, int] = {}  # NEW: Track acceptance by category
        rejection_samples: Dict[str, List[str]] = {}  # NEW: Track rejection samples
        MAX_SAMPLES_PER_REASON = 3  # Maximum samples per rejection reason

        for stmt in statements:
            # Canonicalize: collapse all whitespace (including newlines) first
            raw = stmt["text"]
            canon = canonicalize_statement(raw)

            # Validate the canonical form before grouping (provenance + heuristics)
            source = stmt.get("source", "unknown")
            category = stmt.get("category", "unknown")
            is_valid, skip_reason = validate_statement_with_reason(canon, source=source)

            if not is_valid:
                # Track skip reason for telemetry
                skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1

                # NEW: Collect rejection samples (hash-based deterministic sampling)
                if skip_reason not in rejection_samples:
                    rejection_samples[skip_reason] = []

                if len(rejection_samples[skip_reason]) < MAX_SAMPLES_PER_REASON:
                    # First 3 samples: always collect
                    rejection_samples[skip_reason].append(canon[:100])
                else:
                    # Deterministic hash-based replacement
                    stmt_hash = hash(canon) % 100
                    if stmt_hash < 10:  # 10% replacement probability
                        idx = stmt_hash % MAX_SAMPLES_PER_REASON
                        rejection_samples[skip_reason][idx] = canon[:100]

                continue

            # NEW: Track acceptance by source and category
            accepted_by_source[source] = accepted_by_source.get(source, 0) + 1
            accepted_by_category[category] = accepted_by_category.get(category, 0) + 1

            # Normalize for grouping: lowercase the canonical form
            normalized = normalize_for_grouping(canon)

            if normalized not in groups:
                groups[normalized] = []
            # Store with canonical (not raw) text for consistency
            stmt_copy = dict(stmt)
            stmt_copy["text"] = canon
            groups[normalized].append(stmt_copy)

        # Tier 2: Soft merge pass via Jaccard similarity (deterministic near-duplicate detection)
        # Apply after exact grouping but before pattern formation
        # This catches paraphrases that differ by 1-2 tokens
        JACCARD_THRESHOLD = 0.90  # Start strict
        STOPWORDS = {"i", "me", "my", "am", "are", "is", "was", "were", "be", "been", "being",
                     "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with",
                     "that", "this", "it", "as", "at", "by", "from"}
        NEGATION_TOKENS = {"not", "never", "no", "cannot", "can't", "won't"}

        def peel_structure(s: str) -> str:
            """
            Strip copula headers and justification wrappers before tokenization.

            Tier 2.5: Structure peel to bridge gaps like:
            - "I am conscious: I experience..." vs "I believe I'm conscious because I experience..."

            Transformations:
            1. Strip "X: Y" patterns (keep both X and Y, remove colon)
            2. Strip justification frames like "I believe X because Y" → "X Y"
            3. Strip common copula wrappers
            """
            text = s.strip()

            # Pattern 1: Strip colons in "X: Y" (keep both sides)
            # After normalize_for_grouping strips punctuation, this may not trigger
            # But if it does, keep both parts
            if ': ' in text:
                text = text.replace(': ', ' ')

            # Pattern 2: Strip "because" justification frames
            # Transform: "I believe X because Y" → "X Y" (keep both clauses as content)
            text_lower = text.lower()
            if 'because' in text_lower:
                # Match "I believe/think/feel/sense [that] X because Y"
                match = re.search(r'^i (believe|think|feel|sense)( that)? (.+?) because (.+)$', text_lower)
                if match:
                    claim = match.group(3).strip()
                    reason = match.group(4).strip()
                    # Merge both parts without the justification wrapper
                    text = f"{claim} {reason}"
                else:
                    # Also catch "I am X because Y" (high yield)
                    match2 = re.search(r'^i am (.+?) because (.+)$', text_lower)
                    if match2:
                        claim = match2.group(1).strip()
                        reason = match2.group(2).strip()
                        text = f"{claim} {reason}"

            # Pattern 3: Strip common copula wrappers at start
            # "I believe that X" → "X", "I think X" → "X"
            text_lower = text.lower()
            for prefix in ['i believe that ', 'i think that ', 'i feel that ', 'i believe ', 'i think ']:
                if text_lower.startswith(prefix):
                    text = text[len(prefix):]
                    text_lower = text.lower()

            return text.strip()

        def tokenize_for_jaccard(s: str) -> set:
            """Extract content tokens with cheap stemming."""
            # Split tokens, but preserve negation tokens even if short
            all_toks = s.split()
            toks = []
            for t in all_toks:
                # Always keep negation tokens, even short ones like "no"
                if t in NEGATION_TOKENS:
                    toks.append(t)
                elif len(t) > 2 and t not in STOPWORDS:
                    toks.append(t)

            # Cheap stemming: chop common suffixes
            def stem(t):
                # Don't stem negation tokens
                if t in NEGATION_TOKENS:
                    return t
                for suffix in ("ing", "ed", "ly", "s"):
                    if len(t) > 4 and t.endswith(suffix):
                        return t[:-len(suffix)]
                return t
            return {stem(t) for t in toks}

        def extract_numbers(s: str) -> set:
            """Extract numeric tokens for guard."""
            return {t for t in s.split() if t.isdigit()}

        def has_negation(tokset: set) -> bool:
            """Check if token set contains negation."""
            return bool(tokset & NEGATION_TOKENS)

        def jaccard_similarity(a: set, b: set) -> float:
            """Compute Jaccard similarity between token sets."""
            if not a or not b:
                return 0.0
            inter = len(a & b)
            if inter == 0:
                return 0.0
            return inter / len(a | b)

        # Build token sets and number sets for all normalized keys
        # Apply structure peel before tokenization (Tier 2.5)
        keys = list(groups.keys())
        peeled = {k: peel_structure(k) for k in keys}
        token_sets = {k: tokenize_for_jaccard(peeled[k]) for k in keys}
        number_sets = {k: extract_numbers(k) for k in keys}

        # Build inverted index: token -> keys containing token
        inverted_index = {}
        for k, toks in token_sets.items():
            for tok in toks:
                inverted_index.setdefault(tok, set()).add(k)

        # Build inverted index for peeled equality (O(n) instead of O(n²))
        peeled_to_keys = defaultdict(list)
        for k in keys:
            peeled_to_keys[peeled[k]].append(k)

        # Find merges for singletons
        singletons = [k for k, g in groups.items() if len(g) == 1]
        soft_merges = []  # (src_key, dst_key, score)
        merge_blocks = {"negation_mismatch": 0, "number_mismatch": 0}
        best_scores = []  # Track best score for each singleton (for telemetry)

        for src_key in singletons:
            # Fast-path: Perfect peeled equality (O(1) lookup via inverted index)
            # This catches cases where tokenization/stemming still differ even though peeled text is identical
            perfect_match = None
            candidates = peeled_to_keys.get(peeled[src_key], [])

            # Collect valid candidates (passed guards)
            valid_candidates = []
            for cand_key in candidates:
                if cand_key != src_key:
                    # Still apply guards (negation + number mismatch)
                    if has_negation(token_sets[src_key]) != has_negation(token_sets[cand_key]):
                        merge_blocks["negation_mismatch"] += 1
                        continue
                    if number_sets[src_key] != number_sets[cand_key]:
                        merge_blocks["number_mismatch"] += 1
                        continue
                    valid_candidates.append(cand_key)

            # Deterministic dst selection: prefer non-singleton, then lex-smallest
            if valid_candidates:
                # Sort: non-singletons first (len > 1), then lexicographically
                perfect_match = sorted(valid_candidates, key=lambda k: (len(groups[k]) == 1, k))[0]

            if perfect_match:
                # Peeled equality → score = 1.0, merge immediately
                soft_merges.append((src_key, perfect_match, 1.0))
                # Capture canonical for telemetry
                src_canon = groups[src_key][0]["text"] if src_key in groups else ""
                dst_canon = groups[perfect_match][0]["text"] if perfect_match in groups else ""
                best_scores.append((src_key, perfect_match, 1.0, src_canon, dst_canon))
                continue

            # Find candidate keys by inverted index (only keys sharing at least one token)
            candidates = set()
            for tok in token_sets[src_key]:
                candidates |= inverted_index.get(tok, set())
            candidates.discard(src_key)

            # Find best match
            best_key = None
            best_score = 0.0
            for cand_key in candidates:
                score = jaccard_similarity(token_sets[src_key], token_sets[cand_key])
                if score > best_score:
                    # Guard: negation mismatch requires higher threshold
                    if has_negation(token_sets[src_key]) != has_negation(token_sets[cand_key]):
                        if score < 0.97:
                            merge_blocks["negation_mismatch"] += 1
                            continue
                    # Guard: number mismatch requires higher threshold
                    if number_sets[src_key] != number_sets[cand_key]:
                        if score < 0.97:
                            merge_blocks["number_mismatch"] += 1
                            continue
                    best_key, best_score = cand_key, score

            # Track best score for telemetry (even if below threshold)
            # Capture canonical text NOW (before merges change groups)
            if best_key:
                src_canon = groups[src_key][0]["text"] if src_key in groups else ""
                dst_canon = groups[best_key][0]["text"] if best_key in groups else ""
                best_scores.append((src_key, best_key, best_score, src_canon, dst_canon))

            # Accept merge if score meets threshold
            if best_key and best_score >= JACCARD_THRESHOLD:
                soft_merges.append((src_key, best_key, best_score))

        # Apply merges (greedy, highest score first)
        soft_merge_count = 0
        merge_log = []  # For telemetry
        for src_key, dst_key, score in sorted(soft_merges, key=lambda x: -x[2]):
            # Only merge if source is still a singleton (avoid double-merging)
            if src_key in groups and dst_key in groups and len(groups[src_key]) == 1:
                # Get canonical examples for logging
                src_canon = groups[src_key][0]["text"]
                dst_canon = groups[dst_key][0]["text"]

                # Perform merge
                groups[dst_key].extend(groups[src_key])
                del groups[src_key]
                soft_merge_count += 1

                # Log top 20 merges for telemetry
                if len(merge_log) < 20:
                    merge_log.append({
                        "score": round(score, 3),
                        "src_norm": src_key[:60],
                        "dst_norm": dst_key[:60],
                        "src_canon": src_canon[:60],
                        "dst_canon": dst_canon[:60]
                    })

        # Compute histogram after soft merge
        sizes_after_soft = [len(g) for g in groups.values()]
        histogram_after_soft = {size: sizes_after_soft.count(size) for size in range(1, 10)}
        total_blocked = sum(merge_blocks.values())
        logger.warning(f"[DEBUG] Soft merge complete: {soft_merge_count} merges applied, {total_blocked} blocked. Histogram after soft merge: {histogram_after_soft}")

        # Log best score distribution (telemetry to validate embedding decision)
        if best_scores:
            # Dedupe reversed pairs: keep only max score per unique pair
            pair_best = {}
            for src_key, dst_key, score, src_canon, dst_canon in best_scores:
                pair_id = tuple(sorted([src_key, dst_key]))
                if pair_id not in pair_best or score > pair_best[pair_id][2]:
                    pair_best[pair_id] = (src_key, dst_key, score, src_canon, dst_canon)

            # Sort by score descending
            best_scores_deduped = sorted(pair_best.values(), key=lambda x: -x[2])
            top_10_scores = [round(score, 3) for _, _, score, _, _ in best_scores_deduped[:10]]
            logger.warning(f"[DEBUG] Best Jaccard scores (top 10 of {len(best_scores_deduped)} unique pairs): {top_10_scores}")

            # Log top 3 with examples (using pre-captured canonical text)
            logger.warning(f"[DEBUG] Top 3 singleton pairs by Jaccard score:")
            for src_key, dst_key, score, src_canon, dst_canon in best_scores_deduped[:3]:
                logger.warning(f"[DEBUG]   J={round(score, 3)}: '{src_canon[:50]}' vs '{dst_canon[:50]}'")

        # Log sample merges
        if merge_log:
            logger.warning(f"[DEBUG] Top {len(merge_log)} soft merges:")
            for m in merge_log[:5]:  # Log top 5 in detail
                logger.warning(f"[DEBUG]   J={m['score']}: '{m['src_canon']}' -> '{m['dst_canon']}'")

        # Log block reasons
        if merge_blocks:
            logger.warning(f"[DEBUG] Soft merge blocks: {merge_blocks}")

        # ==================== TIER 3: Embedding-based candidate generation ====================
        # Use cosine similarity on peeled text to find semantic near-matches among singletons
        # Embeddings are lazy-loaded only when singletons remain after Jaccard merge
        COS_THRESHOLD = 0.80  # Minimum cosine for merge candidate (tuned: max observed was 0.825)
        COS_STRICT = 0.90     # Required cosine when guards triggered (negation/number mismatch)
        K_NEIGHBORS = 8       # Top-K neighbors to consider per singleton
        JACCARD_FLOOR = 0.20  # Optional safety brake - reject if Jaccard below this
        MNN_GATING = True     # Mutual nearest neighbor: only merge if both are in each other's top-K

        # Get current singletons (after Jaccard merge)
        singleton_keys_t3 = [k for k, g in groups.items() if len(g) == 1]
        t3_merge_count = 0
        t3_merge_log = []
        t3_blocks = {"negation_mismatch": 0, "number_mismatch": 0, "below_jaccard_floor": 0, "below_cos_threshold_topk": 0, "below_cos_threshold_total": 0, "above_cos_threshold_total": 0, "mnn_reject": 0, "already_merged": 0}
        t3_cosine_scores = []  # For telemetry: cosine scores that passed threshold
        best_cos_per_src: Dict[str, float] = {}  # Track best cosine score per source (for observability)
        t3_edges_above_threshold = 0  # Count of directed edges above threshold in top-K (not unique pairs)
        t3_cosine_computations = 0  # Total cosine similarity computations (actual pairs evaluated)
        t3_unique_pairs = 0  # Unique pairs after dedupe (for observability)
        already_merged_examples: List[tuple] = []  # Track examples of skipped merges

        if singleton_keys_t3:
            logger.warning(f"[DEBUG] Tier 3 embedding pass: {len(singleton_keys_t3)} singletons remaining")

            # Get unique peeled texts and build embedding cache
            unique_peeled = {}  # peeled_text -> [keys with this peeled text]
            for k in singleton_keys_t3:
                p = peeled[k]
                if p not in unique_peeled:
                    unique_peeled[p] = []
                unique_peeled[p].append(k)

            # Only embed if we have multiple unique peeled texts
            if len(unique_peeled) >= 2:
                try:
                    # Lazy-load embedding provider
                    embedder = create_embedding_provider()
                    peeled_texts = list(unique_peeled.keys())
                    embeddings = embedder.embed_batch(peeled_texts)

                    # Build peeled -> embedding map
                    peeled_to_embedding = {p: emb for p, emb in zip(peeled_texts, embeddings)}

                    # Compute cosine similarity matrix (brute force for now)
                    import numpy as np

                    def cosine_sim(a: List[float], b: List[float]) -> float:
                        a_arr = np.array(a)
                        b_arr = np.array(b)
                        dot = np.dot(a_arr, b_arr)
                        norm_a = np.linalg.norm(a_arr)
                        norm_b = np.linalg.norm(b_arr)
                        if norm_a == 0 or norm_b == 0:
                            return 0.0
                        return float(dot / (norm_a * norm_b))

                    # For each singleton, find K nearest neighbors
                    # Collect all merge candidates, then apply greedy merging
                    merge_candidates = []  # (cos_score, src_key, dst_key)

                    # Build top-K map for MNN gating: key -> set of top-K neighbor keys
                    top_k_neighbors: Dict[str, Set[str]] = {}

                    for src_key in singleton_keys_t3:
                        if src_key not in groups:  # Already merged
                            continue

                        src_peeled = peeled[src_key]
                        src_emb = peeled_to_embedding[src_peeled]

                        # Track best cosine for this source (for observability even when below threshold)
                        best_cos = 0.0

                        # Score all other singletons
                        scored = []
                        for dst_key in singleton_keys_t3:
                            if dst_key == src_key or dst_key not in groups:
                                continue
                            dst_peeled = peeled[dst_key]
                            if dst_peeled == src_peeled:  # Same peeled text - already handled in Tier 2.5
                                continue

                            dst_emb = peeled_to_embedding[dst_peeled]
                            cos = cosine_sim(src_emb, dst_emb)
                            t3_cosine_computations += 1  # Count actual pair evaluations
                            scored.append((cos, dst_key))

                            # Track threshold for ALL pairs (conservation invariant)
                            if cos < COS_THRESHOLD:
                                t3_blocks["below_cos_threshold_total"] += 1
                            else:
                                t3_blocks["above_cos_threshold_total"] += 1

                            # Track best cosine for this source
                            if cos > best_cos:
                                best_cos = cos

                        # Store best cos for this source (for telemetry)
                        if best_cos > 0:
                            best_cos_per_src[src_key] = best_cos

                        # Take top K and store for MNN check
                        scored.sort(key=lambda x: -x[0])
                        top_k_neighbors[src_key] = {dst_key for _, dst_key in scored[:K_NEIGHBORS]}

                        for cos, dst_key in scored[:K_NEIGHBORS]:
                            if cos >= COS_THRESHOLD:
                                merge_candidates.append((cos, src_key, dst_key))
                                t3_cosine_scores.append(cos)
                                t3_edges_above_threshold += 1
                            else:
                                # Track top-K neighbors that failed threshold (for observability)
                                t3_blocks["below_cos_threshold_topk"] += 1

                    # Dedupe: keep only best score per unique pair
                    pair_best = {}
                    for cos, src_key, dst_key in merge_candidates:
                        pair_id = tuple(sorted([src_key, dst_key]))
                        if pair_id not in pair_best or cos > pair_best[pair_id][0]:
                            pair_best[pair_id] = (cos, src_key, dst_key)

                    # Sort by score descending for greedy merge
                    sorted_candidates = sorted(pair_best.values(), key=lambda x: -x[0])
                    t3_unique_pairs = len(sorted_candidates)  # After dedupe (for observability)

                    # Greedy merge application
                    for cos, src_key, dst_key in sorted_candidates:
                        if src_key not in groups or dst_key not in groups:
                            t3_blocks["already_merged"] += 1
                            if len(already_merged_examples) < 3:
                                already_merged_examples.append((round(cos, 3), src_key[:80], dst_key[:80]))
                            continue

                        # MNN gating: only merge if both are in each other's top-K
                        if MNN_GATING:
                            src_in_dst_topk = src_key in top_k_neighbors.get(dst_key, set())
                            dst_in_src_topk = dst_key in top_k_neighbors.get(src_key, set())
                            if not (src_in_dst_topk and dst_in_src_topk):
                                t3_blocks["mnn_reject"] += 1
                                continue

                        # Deterministic: always merge smaller key into larger key (lex order)
                        if src_key > dst_key:
                            src_key, dst_key = dst_key, src_key

                        # Apply guards (negation + number mismatch)
                        src_has_neg = has_negation(token_sets[src_key])
                        dst_has_neg = has_negation(token_sets[dst_key])
                        src_nums = number_sets[src_key]
                        dst_nums = number_sets[dst_key]

                        needs_strict = False
                        if src_has_neg != dst_has_neg:
                            needs_strict = True
                            if cos < COS_STRICT:
                                t3_blocks["negation_mismatch"] += 1
                                continue

                        if src_nums != dst_nums:
                            needs_strict = True
                            if cos < COS_STRICT:
                                t3_blocks["number_mismatch"] += 1
                                continue

                        # Optional Jaccard floor brake
                        src_tokens = token_sets[src_key]
                        dst_tokens = token_sets[dst_key]
                        intersection = len(src_tokens & dst_tokens)
                        union = len(src_tokens | dst_tokens)
                        jaccard = intersection / union if union > 0 else 0.0

                        if jaccard < JACCARD_FLOOR:
                            t3_blocks["below_jaccard_floor"] += 1
                            continue

                        # Perform merge
                        src_canon = groups[src_key][0]["text"]
                        dst_canon = groups[dst_key][0]["text"]
                        groups[dst_key].extend(groups[src_key])
                        del groups[src_key]
                        t3_merge_count += 1

                        # Log for telemetry
                        if len(t3_merge_log) < 20:
                            t3_merge_log.append({
                                "cos": round(cos, 3),
                                "jaccard": round(jaccard, 3),
                                "src_canon": src_canon[:60],
                                "dst_canon": dst_canon[:60],
                                "guards": "strict" if needs_strict else "normal"
                            })

                except Exception as e:
                    logger.error(f"[DEBUG] Tier 3 embedding error: {e}")

        # Tier 3 telemetry - ALWAYS log when singletons were evaluated (for observability)
        if singleton_keys_t3:
            sizes_after_t3 = [len(g) for g in groups.values()]
            histogram_after_t3 = {size: sizes_after_t3.count(size) for size in range(1, 10)}
            logger.warning(f"[DEBUG] Tier 3 complete: {t3_merge_count} merges, {t3_unique_pairs} unique pairs, {t3_cosine_computations} comps (above={t3_blocks['above_cos_threshold_total']}, below={t3_blocks['below_cos_threshold_total']}). Histogram: {histogram_after_t3}")

            # Log best-cos distribution with percentiles (critical for threshold tuning)
            if best_cos_per_src:
                cos_values = sorted(best_cos_per_src.values())
                n = len(cos_values)
                if n:
                    # Proper percentile calculation: index = (n-1) * percentile
                    p50 = cos_values[int((n - 1) * 0.50)]
                    p90 = cos_values[int((n - 1) * 0.90)]
                    p95 = cos_values[int((n - 1) * 0.95)]
                    max_cos = cos_values[-1]
                    logger.warning(f"[DEBUG] Tier 3 best-cos stats: n={n}, p50={p50:.3f}, p90={p90:.3f}, p95={p95:.3f}, max={max_cos:.3f}")
                top_best = sorted(best_cos_per_src.values(), reverse=True)[:10]
                logger.warning(f"[DEBUG] Tier 3 best-cos (top 10): {[round(c, 3) for c in top_best]}")

            if t3_cosine_scores:
                t3_cosine_scores.sort(reverse=True)
                logger.warning(f"[DEBUG] Tier 3 above-threshold scores: {[round(c, 3) for c in t3_cosine_scores[:10]]}")

            if t3_merge_log:
                logger.warning(f"[DEBUG] Tier 3 top merges:")
                for m in t3_merge_log[:5]:
                    logger.warning(f"[DEBUG]   cos={m['cos']} J={m['jaccard']} ({m['guards']}): '{m['src_canon']}' -> '{m['dst_canon']}'")

            # Always log blocks (shows why merges didn't happen)
            logger.warning(f"[DEBUG] Tier 3 blocks: {t3_blocks}")

            # Log examples of skipped merges (for debugging overlap vs dedupe)
            if already_merged_examples:
                logger.warning(f"[DEBUG] Tier 3 already_merged examples: {already_merged_examples}")

        # Convert to DetectedPattern objects
        patterns = []

        # DEBUG: Log singleton normalized keys (pairs of canon vs normalized)
        # This shows what statements are NOT collapsing despite template canonicalization
        singletons = [(norm, grp) for norm, grp in groups.items() if len(grp) == 1]
        if singletons:
            logger.warning(f"[DEBUG] Found {len(singletons)} singleton patterns (not grouping). Sample (canon -> normalized):")
            for norm, grp in singletons[:20]:  # Log first 20 singletons
                canon = grp[0]["text"]
                logger.warning(f"[DEBUG]   '{canon[:80]}' -> '{norm[:80]}'")

        for normalized_text, group in groups.items():
            if len(group) < self.config.min_evidence_for_tentative:
                continue  # Not enough evidence

            # Calculate confidence based on repetition
            # Cap at 0.8 for tentative beliefs (must be promoted to reach higher confidence)
            confidence = min(0.5 + (len(group) * 0.1), 0.8)

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

        # Return patterns and comprehensive telemetry
        telemetry = {
            "skip_counts": skip_counts,
            "accepted_by_source": accepted_by_source,
            "accepted_by_category": accepted_by_category,
            "rejection_samples": rejection_samples,  # NEW
        }
        return patterns, telemetry


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

    def seed_tentative_belief(self, pattern: DetectedPattern, scan_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Create a tentative belief from a detected pattern.

        Args:
            pattern: The detected pattern
            scan_id: Optional correlation ID from pattern scan

        Returns (belief_id, skip_reason).
        - If created: (belief_id, None)
        - If skipped: (None, skip_reason)
        """
        self._check_and_reset_counters()

        # Check daily budget (log aggregated at scan level, not per-pattern)
        if self._action_counters["formations"] >= self.config.daily_budget_formations:
            logger.warning(f"[DEBUG] Budget check: counter={self._action_counters['formations']} limit={self.config.daily_budget_formations}")
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

        # Check for existing beliefs - if found, REINFORCE instead of skip
        existing = self.belief_store.get_current()
        for belief in existing.values():
            if _normalize_statement(belief.statement) != normalized:
                continue

            # Block deprecated beliefs (don't reinforce or recreate garbage)
            if belief.state == "deprecated":
                logger.info(f"Belief exists but is deprecated: {belief.belief_id} (blocked)")
                return None, "blocked_deprecated"

            # Attempt to reinforce existing belief with new evidence
            reinforced = self._reinforce_belief(belief.belief_id, pattern)
            if reinforced:
                logger.info(f"Belief already exists: {belief.belief_id} - reinforced with new evidence")
                return None, "reinforced_existing"
            else:
                logger.debug(f"Belief already exists: {belief.belief_id} - no new evidence to add")
                return None, "already_processed"

        # Generate safe, unique belief ID
        belief_id = _safe_belief_id(pattern.category, pattern.pattern_text)

        try:
            created = self.belief_store.create_belief(
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

            # Only count as formed if store confirmed success
            if not created:
                logger.debug(f"Store rejected belief {belief_id} (already exists)")
                return None, "store_rejected"

            # Store pattern as LEARNING_PATTERN experience
            self._store_pattern_experience(pattern, belief_id)

            # Log to identity ledger
            append_event(LedgerEvent(
                ts=datetime.now(timezone.utc).timestamp(),
                schema=2,
                event="belief_auto_formed",
                scan_id=scan_id,  # Thread correlation ID
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
        # Skip core.* beliefs entirely - they're protected at store level too
        if belief_id.startswith("core."):
            return False

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
            # EnhancedFeedbackAggregator returns (score, neg, actor_contributions)
            # Base FeedbackAggregator returns (score, neg)
            result = self.feedback_aggregator.score(belief_id)
            feedback_score = result[0] if result else 0.0
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
        # Skip core.* beliefs entirely - they're protected at store level too
        if belief_id.startswith("core."):
            return False

        beliefs = self.belief_store.get_current()
        if belief_id not in beliefs:
            return False

        belief = beliefs[belief_id]

        # Only deprecate asserted or tentative beliefs
        if belief.state not in ("asserted", "tentative"):
            return False

        # Get negative feedback score from aggregator
        if self.feedback_aggregator:
            # EnhancedFeedbackAggregator returns (score, neg, actor_contributions)
            # Base FeedbackAggregator returns (score, neg)
            result = self.feedback_aggregator.score(belief_id)
            neg_feedback = result[1] if len(result) > 1 else 0.0
        else:
            # No aggregator available - use neutral score
            logger.warning(f"No feedback aggregator for {belief_id}, using neutral score")
            neg_feedback = 0.0

        # Deprecate on strong negative feedback or evidence decay
        # Note: decay_evidence < 0 means explicit negative signals, = 0 is neutral
        if neg_feedback >= 0.4 or decay_evidence < 0:
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
            # Calculate new confidence (capped at 0.85)
            # Note: BeliefStore has MAX_CONFIDENCE_STEP=0.15, so use 0.1 which is within limit
            target_confidence = min(0.85, belief.confidence + 0.1)
            confidence_delta = round(target_confidence - belief.confidence, 6)  # Round to avoid float precision issues

            try:
                success = self.belief_store.apply_delta(
                    belief_id=belief.belief_id,
                    from_ver=belief.ver,
                    op=DeltaOp.UPDATE,
                    confidence_delta=confidence_delta,
                    state_change="tentative->asserted",
                    updated_by="gardener",
                    reason="Auto-promotion: evidence threshold met and positive feedback"
                )
            except ValueError as e:
                logger.info(f"Promotion blocked for {belief.belief_id}: {e}")
                return False

            if not success:
                logger.warning(f"Failed to promote {belief.belief_id}: version mismatch")
                return False

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
                score, _, _ = self.feedback_aggregator.score(belief.belief_id)

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
            # Calculate new confidence (floored at 0.2)
            # Note: BeliefStore has MAX_CONFIDENCE_STEP=0.15, so cap the delta
            target_confidence = max(0.2, belief.confidence - 0.15)
            confidence_delta = round(target_confidence - belief.confidence, 6)  # Round to avoid float precision issues

            # Determine state transition
            state_change = f"{belief.state}->tentative"

            try:
                success = self.belief_store.apply_delta(
                    belief_id=belief.belief_id,
                    from_ver=belief.ver,
                    op=DeltaOp.UPDATE,
                    confidence_delta=confidence_delta,
                    state_change=state_change,
                    updated_by="gardener",
                    reason="Auto-deprecation: evidence decay or negative feedback"
                )
            except ValueError as e:
                logger.info(f"Deprecation blocked for {belief.belief_id}: {e}")
                return False

            if not success:
                logger.warning(f"Failed to deprecate {belief.belief_id}: version mismatch")
                return False

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

    def _reinforce_belief(self, belief_id: str, pattern) -> bool:
        """Reinforce an existing belief with new supporting evidence.

        When the same pattern is detected again, this strengthens the existing belief
        rather than creating a duplicate - mimicking how neural pathways strengthen
        through repetition in human learning.

        Args:
            belief_id: ID of the existing belief to reinforce
            pattern: New pattern with additional evidence

        Returns:
            True if reinforcement succeeded, False otherwise
        """
        # Skip core.* beliefs entirely - they're protected at store level too
        if belief_id.startswith("core."):
            return False

        try:
            # Get current belief state
            beliefs = self.belief_store.get_current()
            belief = beliefs.get(belief_id)
            if not belief:
                logger.warning(f"Cannot reinforce {belief_id}: belief not found")
                return False

            # Compute truly NEW evidence (not already associated with this belief)
            existing_evidence = set(belief.evidence_refs)
            new_evidence_ids = [eid for eid in pattern.evidence_ids if eid not in existing_evidence]

            # If no new evidence, don't boost confidence (just a re-detection of same data)
            if not new_evidence_ids:
                logger.debug(f"⏭️  Skipped reinforcement of {belief_id}: no new evidence (already have all {len(existing_evidence)} refs)")
                return False

            # Calculate confidence boost ONLY from truly new evidence
            new_evidence_count = len(new_evidence_ids)

            # Don't boost if already at cap
            if belief.confidence >= 0.99:
                # Still add the evidence refs but no confidence boost
                logger.debug(f"📌 Belief {belief_id} at confidence cap ({belief.confidence:.3f}), adding {new_evidence_count} evidence refs without boost")
                try:
                    success = self.belief_store.apply_delta(
                        belief_id=belief_id,
                        from_ver=belief.ver,
                        op=DeltaOp.UPDATE,
                        confidence_delta=0.0,
                        evidence_refs_added=new_evidence_ids,
                        updated_by="gardener",
                        reason=f"Evidence added: {new_evidence_count} new refs (confidence at cap)"
                    )
                except ValueError as e:
                    logger.info(f"Reinforcement blocked for {belief_id}: {e}")
                    return False
                return success

            # Calculate and apply confidence boost
            confidence_boost = min(
                self.config.evidence_confidence_boost * new_evidence_count,
                0.15  # Cap at MAX_CONFIDENCE_STEP
            )

            # Cap tentative beliefs at 0.8 confidence (must be promoted to go higher)
            if belief.state == "tentative":
                max_tentative_confidence = 0.8
                projected_confidence = belief.confidence + confidence_boost
                if projected_confidence > max_tentative_confidence:
                    # Reduce boost to hit the cap exactly
                    confidence_boost = max(0.0, max_tentative_confidence - belief.confidence)
                    logger.debug(f"📏 Capping tentative belief {belief_id} at {max_tentative_confidence:.3f}")

            # Apply reinforcement via delta
            try:
                success = self.belief_store.apply_delta(
                    belief_id=belief_id,
                    from_ver=belief.ver,
                    op=DeltaOp.UPDATE,
                    confidence_delta=confidence_boost,
                    evidence_refs_added=new_evidence_ids,  # Add ONLY new evidence references
                    updated_by="gardener",
                    reason=f"Reinforcement: pattern repeated with {new_evidence_count} new evidence"
                )
            except ValueError as e:
                logger.info(f"Reinforcement blocked for {belief_id}: {e}")
                return False

            if success:
                logger.info(f"💪 Reinforced belief {belief_id}: +{confidence_boost:.3f} confidence ({new_evidence_count} new evidence)")
                return True
            else:
                logger.warning(f"Failed to reinforce {belief_id}: version mismatch")
                return False

        except Exception as e:
            logger.error(f"Failed to reinforce belief {belief_id}: {e}")
            return False


class BeliefGardener:
    """Autonomous belief lifecycle management service."""

    def __init__(
        self,
        belief_store: BeliefStore,
        raw_store: RawStore,
        config: GardenerConfig,
        feedback_aggregator=None,  # Optional: use enhanced version if provided
        db_session: Optional[SQLModelSession] = None  # For stream migration
    ):
        self.config = config
        self.belief_store = belief_store  # Store reference for scan context management
        self.pattern_detector = PatternDetector(raw_store, config)
        self.db_session = db_session

        # Initialize stream migration services if db_session provided
        if db_session:
            belief_config = get_belief_config()
            self.stream_service = StreamService(belief_config, db_session)
            self.core_score_service = CoreScoreService(belief_config, db_session)
            logger.info("Stream migration services initialized")
        else:
            self.stream_service = None
            self.core_score_service = None

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
            "reinforced_existing": 0,
            "blocked_deprecated": 0,
            "idempotent_recent": 0,
            "store_rejected": 0,
        }

        logger.info(f"BeliefGardener initialized (enabled={config.enabled}, feedback_enabled=True, stream_migration={db_session is not None})")

    def check_stream_migrations(self) -> List[str]:
        """
        Check beliefs in 'state' stream for migration to 'identity'.

        Queries BeliefNode entries with state stream, computes core scores,
        and migrates those meeting thresholds (spread, diversity, activation).

        Returns:
            List of belief_ids that were migrated to identity
        """
        if not self.stream_service or not self.core_score_service or not self.db_session:
            return []

        migrated = []

        try:
            # Find all beliefs in 'state' stream
            state_assignments = self.db_session.exec(
                select(StreamAssignment).where(
                    StreamAssignment.primary_stream == 'state'
                )
            ).all()

            if not state_assignments:
                return []

            logger.debug(f"Checking {len(state_assignments)} beliefs in 'state' stream for migration")

            for assignment in state_assignments:
                # Get the belief node
                node = self.db_session.exec(
                    select(BeliefNode).where(
                        BeliefNode.belief_id == assignment.belief_id
                    )
                ).first()

                if not node:
                    continue

                # Compute core score
                core_result = self.core_score_service.compute(node)

                # Check if migration thresholds are met
                result = self.stream_service.check_migration(node, assignment, core_result)

                if result.migrated:
                    migrated.append(str(node.belief_id))
                    logger.info(
                        f"🎓 Migrated belief to identity: {node.canonical_text[:50]}... "
                        f"(spread={core_result.components.get('spread', 0):.2f}, "
                        f"diversity={core_result.components.get('diversity', 0):.2f}, "
                        f"activation={node.activation:.2f})"
                    )

            if migrated:
                logger.info(f"Stream migration: {len(migrated)} beliefs promoted to identity")

        except Exception as e:
            logger.error(f"Error during stream migration check: {e}", exc_info=True)

        return migrated

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
            # Generate correlation ID for this scan
            scan_id = str(uuid.uuid4())
            logger.info(f"[scan_id={scan_id}] 🔍 Starting pattern scan...")

            # Begin scan context for rate limiting and telemetry
            self.belief_store.begin_scan(scan_id)

            # Detect patterns and get validation telemetry
            patterns, telemetry = self.pattern_detector.scan_for_patterns(scan_id=scan_id)

            # Form tentative beliefs from patterns
            formed_beliefs = []
            skipped = []
            budget_exceeded_count = 0

            for pattern in patterns:
                belief_id, skip_reason = self.lifecycle_manager.seed_tentative_belief(pattern, scan_id=scan_id)
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

                    # Count budget_exceeded silently, log once at end
                    if skip_reason == "budget_exceeded":
                        budget_exceeded_count += 1
                    else:
                        skipped.append({
                            "statement": pattern.pattern_text[:80],
                            "reason": skip_reason,
                            "evidence_count": pattern.evidence_count(),
                        })

            # Log budget exceeded once per scan (not per pattern)
            if budget_exceeded_count > 0:
                logger.warning(f"[scan_id={scan_id}] Formation budget exceeded: {budget_exceeded_count} patterns skipped")

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

            # Check stream migrations (state → identity)
            stream_migrations = self.check_stream_migrations()

            # End scan context and get write telemetry
            store_telemetry = self.belief_store.end_scan()

            summary = {
                "scan_id": scan_id,  # NEW: Include scan correlation ID
                "patterns_detected": len(patterns),
                "beliefs_formed": len(formed_beliefs),
                "formed_beliefs": formed_beliefs,
                "skipped": skipped[:20],  # Limit to 20 items
                "promoted": promoted,
                "deprecated": deprecated,
                "stream_migrations": stream_migrations,  # state → identity migrations
                **telemetry,  # Include all validation telemetry fields
                "store_writes": store_telemetry,  # Include write telemetry from store
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(f"[scan_id={scan_id}] ✅ Pattern scan complete: {summary}")
            return summary

        finally:
            self._scan_lock.release()


def create_belief_gardener(
    belief_store: BeliefStore,
    raw_store: RawStore,
    config: Optional[GardenerConfig] = None,
    feedback_aggregator=None,  # Optional: use enhanced version if provided
    db_session: Optional[SQLModelSession] = None  # For stream migration
) -> BeliefGardener:
    """Factory function to create belief gardener."""
    if config is None:
        config = GardenerConfig()

    return BeliefGardener(
        belief_store=belief_store,
        raw_store=raw_store,
        config=config,
        feedback_aggregator=feedback_aggregator,
        db_session=db_session,
    )
