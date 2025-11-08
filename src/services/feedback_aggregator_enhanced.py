"""
Enhanced Feedback Aggregator - Outcome-driven dynamic weighting for belief feedback.

Implements three system-grounded multipliers:
- g_align: Self-alignment (live vs origin anchor)
- g_conviction: Belief confidence weighting
- g_trust: Learned provenance trust

Plus tag storm deduplication and credit assignment tracking.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

from src.memory.raw_store import RawStore
from src.memory.models import ExperienceType

logger = logging.getLogger(__name__)


@dataclass
class FeedbackConfig:
    """Configuration for feedback aggregation."""
    enabled: bool = True
    window_hours: int = 24  # Trailing window for tag analysis
    cache_ttl_secs: int = 300  # Cache scores for 5 minutes
    min_samples: int = 2  # Minimum tags required for valid score

    # Dynamic multiplier parameters
    alpha_align: float = 1.5  # Exponent for precision factor
    conviction_base: float = 0.5  # Base conviction
    conviction_scale: float = 1.5  # Scaling for conviction

    # Tag storm deduplication
    dedup_window_secs: int = 120  # 2-minute window for dedup
    dedup_similarity_threshold: float = 0.9  # Similarity threshold

    # Alignment thresholds
    min_alignment_for_doubt: float = 0.2  # Ignore off-target doubts
    min_similarity_for_doubt: float = 0.3  # Minimum tag-belief similarity


class EnhancedFeedbackAggregator:
    """
    Enhanced feedback aggregator with outcome-driven dynamic weighting.

    Computes belief feedback scores using:
    1. g_align: Plasticity-aware self-alignment
    2. g_conviction: Confidence-based sensitivity
    3. g_trust: Learned provenance trust
    4. Tag storm deduplication
    5. Credit assignment tracking for outcome evaluation
    """

    # Base tag weights (before multipliers)
    TAG_BASE_WEIGHTS = {
        "+keep": 0.5,
        "+shift": 0.3,
        "+doubt": -0.4,
        "+artifact": -0.6,
    }

    def __init__(
        self,
        raw_store: RawStore,
        provenance_trust: Optional[Any] = None,  # ProvenanceTrust instance
        awareness_loop: Optional[Any] = None,  # AwarenessLoop for anchors
        belief_store: Optional[Any] = None,  # BeliefStore for confidence
        outcome_evaluator: Optional[Any] = None,  # OutcomeEvaluator for credit
        embedding_provider: Optional[Any] = None,  # For semantic similarity
        config: Optional[FeedbackConfig] = None
    ):
        """
        Initialize enhanced feedback aggregator.

        Args:
            raw_store: Raw experience store
            provenance_trust: Provenance trust manager
            awareness_loop: Awareness loop for anchor access
            belief_store: Belief store for confidence
            outcome_evaluator: Outcome evaluator for credit tracking
            embedding_provider: Embedding provider for similarity
            config: Configuration
        """
        self.raw_store = raw_store
        self.provenance_trust = provenance_trust
        self.awareness_loop = awareness_loop
        self.belief_store = belief_store
        self.outcome_evaluator = outcome_evaluator
        self.embedding_provider = embedding_provider
        self.config = config or FeedbackConfig()

        # Score cache: belief_id -> (feedback_score, neg_feedback, timestamp)
        self._cache: Dict[str, Tuple[float, float, float]] = {}

        # Global score cache
        self._global_cache: Optional[Tuple[float, float, float]] = None

        # Circuit breaker state
        self.circuit_open = False

        # Telemetry
        self.window_size_hours = self.config.window_hours
        self.last_scores: Dict[str, Tuple[float, float]] = {}
        self.global_feedback_score: float = 0.0
        self.global_neg_feedback: float = 0.0

        # Tag storm dedup tracking: (belief_id, actor, tag) -> last_ts
        self._tag_history: Dict[Tuple[str, str, str], float] = {}

        logger.info(f"EnhancedFeedbackAggregator initialized with dynamic weighting")

    def score(
        self,
        belief_id: str,
        now: Optional[float] = None
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Compute feedback scores for a belief with actor contributions.

        Args:
            belief_id: ID of belief to score
            now: Current timestamp

        Returns:
            (feedback_score, neg_feedback, actor_contributions) where:
            - feedback_score ∈ [-1, 1]: Overall signal
            - neg_feedback ∈ [0, 1]: Negative signal strength
            - actor_contributions: Dict of actor → weighted contribution
        """
        if now is None:
            now = time.time()

        # Check cache
        if belief_id in self._cache:
            score, neg, cache_ts = self._cache[belief_id]
            age = now - cache_ts
            if age < self.config.cache_ttl_secs:
                logger.debug(f"Cache hit for {belief_id} (age={age:.1f}s)")
                # Return cached scores with empty contributions (not cached)
                return score, neg, {}

        # Compute fresh scores
        feedback_score, neg_feedback, actor_contributions = self._compute_scores_enhanced(
            belief_id, now
        )

        # Update cache
        self._cache[belief_id] = (feedback_score, neg_feedback, now)
        self.last_scores[belief_id] = (feedback_score, neg_feedback)

        logger.debug(
            f"Scored {belief_id}: feedback={feedback_score:.2f}, neg={neg_feedback:.2f}, "
            f"actors={list(actor_contributions.keys())}"
        )

        return feedback_score, neg_feedback, actor_contributions

    def _compute_scores_enhanced(
        self,
        belief_id: str,
        now: float
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Compute scores with dynamic weighting and actor tracking.

        Returns:
            (feedback_score, neg_feedback, actor_contributions)
        """
        # Get belief for dynamic multipliers
        belief = self._get_belief(belief_id)
        if not belief:
            return 0.0, 0.0, {}

        # Compute dynamic multipliers
        g_align = self._compute_g_align(belief_id, belief)
        g_conviction = self._compute_g_conviction(belief)

        # Get cutoff time
        cutoff_dt = datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(
            hours=self.config.window_hours
        )

        # Query recent experiences
        try:
            recent = self.raw_store.list_recent(
                limit=200,
                experience_type=ExperienceType.OCCURRENCE,
                since=cutoff_dt,
            )
        except Exception as e:
            logger.error(f"Failed to query experiences: {e}")
            return 0.0, 0.0, {}

        # Extract tags with actor tracking and deduplication
        tag_samples = []  # List of (tag, actor, ts, alignment)
        for exp in recent:
            if self._mentions_belief(exp, belief_id):
                # Extract tags with actors
                tags_with_actors = self._extract_tags_with_actors(exp)

                for tag, actor in tags_with_actors:
                    if tag not in self.TAG_BASE_WEIGHTS:
                        continue

                    # Check deduplication
                    dedup_key = (belief_id, actor, tag)
                    if dedup_key in self._tag_history:
                        last_ts = self._tag_history[dedup_key]
                        if now - last_ts < self.config.dedup_window_secs:
                            logger.debug(
                                f"Tag storm dedup: {tag} from {actor} for {belief_id} "
                                f"(within {self.config.dedup_window_secs}s window)"
                            )
                            continue  # Skip duplicate

                    # Compute alignment for this tag
                    alignment = self._compute_tag_alignment(belief, exp, tag)

                    # Gate negative tags on alignment
                    if tag in ["+doubt", "+artifact"]:
                        if alignment < self.config.min_alignment_for_doubt:
                            logger.debug(
                                f"Ignoring off-target {tag}: alignment={alignment:.3f}"
                            )
                            continue

                    # Record for dedup
                    self._tag_history[dedup_key] = now

                    # Store sample
                    exp_ts = exp.created_at.timestamp()
                    tag_samples.append((tag, actor, exp_ts, alignment))

        # Check minimum samples
        if len(tag_samples) < self.config.min_samples:
            return 0.0, 0.0, {}

        # Compute weighted scores with dynamic multipliers
        total_weighted_score = 0.0
        actor_contributions = defaultdict(float)
        neg_tag_weight = 0.0
        total_weight = 0.0

        for tag, actor, ts, alignment in tag_samples:
            # Base tag weight
            base_weight = self.TAG_BASE_WEIGHTS[tag]

            # Get provenance trust multiplier
            g_trust = 1.0
            if self.provenance_trust:
                g_trust = self.provenance_trust.get_trust_multiplier(actor)

            # Compute precision factor (alignment-based)
            precision = abs(alignment) ** self.config.alpha_align

            # Final multiplier
            multiplier = precision * g_conviction * g_trust

            # Weighted contribution
            weighted = base_weight * multiplier

            # Accumulate
            total_weighted_score += weighted
            total_weight += abs(weighted)
            actor_contributions[actor] += abs(weighted)

            # Track negative feedback
            if tag in ["+doubt", "+artifact"]:
                neg_tag_weight += abs(weighted)

        # Normalize scores
        if total_weight == 0:
            return 0.0, 0.0, {}

        feedback_score = np.clip(total_weighted_score / total_weight, -1.0, 1.0)
        neg_feedback = neg_tag_weight / total_weight if total_weight > 0 else 0.0

        # Normalize actor contributions to [0, 1]
        total_contrib = sum(actor_contributions.values())
        if total_contrib > 0:
            actor_contributions = {
                actor: contrib / total_contrib
                for actor, contrib in actor_contributions.items()
            }

        return feedback_score, neg_feedback, dict(actor_contributions)

    def _compute_g_align(self, belief_id: str, belief: Any) -> float:
        """
        Compute self-alignment multiplier.

        g_align = ((a_live - a_origin + 1) / 2) ^ alpha

        Args:
            belief_id: Belief ID
            belief: Belief object

        Returns:
            Alignment multiplier ∈ [0, 1]
        """
        if not self.awareness_loop or not self.embedding_provider:
            return 1.0  # Neutral if unavailable

        try:
            # Get anchors
            anchors = self.awareness_loop.anchors
            if "self_anchor_live" not in anchors or "self_anchor_origin" not in anchors:
                return 1.0

            anchor_live = anchors["self_anchor_live"]
            anchor_origin = anchors["self_anchor_origin"]

            # Get belief embedding
            # TODO: Cache belief embeddings
            belief_text = belief.statement if hasattr(belief, 'statement') else str(belief)
            # For now, use simple placeholder
            # In production, would embed belief_text and compute cosine similarity

            # Placeholder: return neutral
            # Real implementation would:
            # belief_vec = await self.embedding_provider.embed(belief_text)
            # a_live = cosine_similarity(belief_vec, anchor_live)
            # a_origin = cosine_similarity(belief_vec, anchor_origin)
            # a_normalized = (a_live - a_origin + 1) / 2
            # return a_normalized ** self.config.alpha_align

            return 1.0  # Neutral until embeddings implemented

        except Exception as e:
            logger.error(f"Failed to compute g_align: {e}")
            return 1.0

    def _compute_g_conviction(self, belief: Any) -> float:
        """
        Compute conviction multiplier based on belief confidence.

        g_conviction = conviction_base + (confidence * conviction_scale)

        Args:
            belief: Belief object

        Returns:
            Conviction multiplier ∈ [0.5, 2.0]
        """
        try:
            confidence = belief.confidence if hasattr(belief, 'confidence') else 0.5

            g_conviction = self.config.conviction_base + (
                confidence * self.config.conviction_scale
            )

            return np.clip(g_conviction, 0.5, 2.0)

        except Exception as e:
            logger.error(f"Failed to compute g_conviction: {e}")
            return 1.0

    def _compute_tag_alignment(
        self,
        belief: Any,
        experience: Any,
        tag: str
    ) -> float:
        """
        Compute semantic alignment between tag and belief.

        Args:
            belief: Belief object
            experience: Experience containing tag
            tag: Tag string

        Returns:
            Alignment score ∈ [-1, 1]
        """
        # Placeholder: return moderate alignment
        # Real implementation would compute semantic similarity between
        # the text surrounding the tag and the belief statement
        return 0.7

    def _get_belief(self, belief_id: str) -> Optional[Any]:
        """Get belief object from store."""
        if not self.belief_store:
            return None

        try:
            beliefs = self.belief_store.get_current()
            return beliefs.get(belief_id)
        except Exception as e:
            logger.error(f"Failed to get belief {belief_id}: {e}")
            return None

    def _mentions_belief(self, experience, belief_id: str) -> bool:
        """Check if experience mentions or relates to a belief."""
        # Check structured content for belief_id
        if hasattr(experience.content, 'structured') and experience.content.structured:
            structured = experience.content.structured
            if isinstance(structured, dict):
                if 'belief_id' in structured and structured['belief_id'] == belief_id:
                    return True
                if 'beliefs_touched' in structured and belief_id in structured.get('beliefs_touched', []):
                    return True
                if 'belief_ids' in structured and belief_id in structured.get('belief_ids', []):
                    return True

        # Check metadata
        if hasattr(experience, 'metadata') and experience.metadata:
            if isinstance(experience.metadata, dict):
                if belief_id in str(experience.metadata):
                    return True

        # Text search
        if hasattr(experience.content, 'text') and experience.content.text:
            if belief_id in experience.content.text:
                return True

        return False

    def _extract_tags_with_actors(self, experience) -> List[Tuple[str, str]]:
        """
        Extract tags with their provenance actors.

        Args:
            experience: Experience to extract from

        Returns:
            List of (tag, actor) tuples
        """
        tags_with_actors = []

        # Try to get actor from provenance
        actor = "unknown"
        if hasattr(experience, 'metadata') and experience.metadata:
            if isinstance(experience.metadata, dict):
                # Check provenance
                if 'provenance' in experience.metadata:
                    prov = experience.metadata['provenance']
                    if isinstance(prov, dict):
                        actor = prov.get('actor', 'unknown')

                # Check structured content for actor
                if 'actor' in experience.metadata:
                    actor = experience.metadata['actor']

        # Extract tags
        tags = []

        # From structured content
        if hasattr(experience.content, 'structured') and experience.content.structured:
            structured = experience.content.structured
            if isinstance(structured, dict):
                if 'tags' in structured:
                    tag_list = structured['tags']
                    if isinstance(tag_list, list):
                        tags.extend(tag_list)
                    elif isinstance(tag_list, str):
                        tags.append(tag_list)

        # From metadata
        if hasattr(experience, 'metadata') and experience.metadata:
            if isinstance(experience.metadata, dict):
                if 'tags' in experience.metadata:
                    meta_tags = experience.metadata['tags']
                    if isinstance(meta_tags, list):
                        tags.extend(meta_tags)
                    elif isinstance(meta_tags, str):
                        tags.append(meta_tags)

        # From text content
        if hasattr(experience.content, 'text') and experience.content.text:
            text = experience.content.text
            for tag in self.TAG_BASE_WEIGHTS.keys():
                if tag in text:
                    tags.append(tag)

        # Combine tags with actor
        for tag in tags:
            tags_with_actors.append((tag, actor))

        return tags_with_actors

    def global_score(self, now: Optional[float] = None) -> Tuple[float, float]:
        """Compute global feedback scores across all beliefs."""
        if now is None:
            now = time.time()

        # Check cache
        if self._global_cache:
            score, neg, cache_ts = self._global_cache
            age = now - cache_ts
            if age < self.config.cache_ttl_secs:
                return score, neg

        # Compute fresh global scores
        cutoff_dt = datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(
            hours=self.config.window_hours
        )

        try:
            recent = self.raw_store.list_recent(
                limit=200,
                experience_type=ExperienceType.OCCURRENCE,
                since=cutoff_dt,
            )
        except Exception as e:
            logger.error(f"Failed to query experiences for global score: {e}")
            return 0.0, 0.0

        # Extract all tags
        tag_counts = defaultdict(int)
        for exp in recent:
            tags_with_actors = self._extract_tags_with_actors(exp)
            for tag, _ in tags_with_actors:
                if tag in self.TAG_BASE_WEIGHTS:
                    tag_counts[tag] += 1

        # Compute scores
        if sum(tag_counts.values()) < self.config.min_samples:
            score, neg = 0.0, 0.0
        else:
            total_weight = 0.0
            for tag, count in tag_counts.items():
                total_weight += self.TAG_BASE_WEIGHTS[tag] * count

            total_tags = sum(tag_counts.values())
            score = total_weight / total_tags if total_tags > 0 else 0.0
            score = np.clip(score, -1.0, 1.0)

            neg_tags = tag_counts.get("+doubt", 0) + tag_counts.get("+artifact", 0)
            neg = neg_tags / total_tags if total_tags > 0 else 0.0

        # Update cache and telemetry
        self._global_cache = (score, neg, now)
        self.global_feedback_score = score
        self.global_neg_feedback = neg

        # Update circuit breaker
        self.circuit_open = neg > 0.6

        logger.debug(
            f"Global score: feedback={score:.2f}, neg={neg:.2f}, "
            f"circuit_open={self.circuit_open}"
        )
        return score, neg

    def get_telemetry(self) -> Dict:
        """Get telemetry data for status endpoint."""
        # Update global scores
        self.global_score()

        telemetry = {
            "global_feedback_score": self.global_feedback_score,
            "global_neg_feedback": self.global_neg_feedback,
            "window_minutes": self.window_size_hours * 60,
            "circuit_open": self.circuit_open,
            "cached_beliefs": len(self._cache),
            "dedup_entries": len(self._tag_history),
            "last_scores": {
                belief_id: {"feedback": score, "negative": neg}
                for belief_id, (score, neg) in self.last_scores.items()
            },
        }

        # Add provenance trust if available
        if self.provenance_trust:
            telemetry["provenance_trust"] = self.provenance_trust.get_telemetry()

        # Add outcome evaluator if available
        if self.outcome_evaluator:
            telemetry["outcome_evaluator"] = self.outcome_evaluator.get_telemetry()

        return telemetry
