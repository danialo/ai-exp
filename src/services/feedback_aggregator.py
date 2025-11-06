"""
Feedback Aggregator - Tag-based feedback scoring for belief lifecycle decisions.

Analyzes conversation tags (+keep, +shift, +doubt, +artifact) over a trailing
window to produce feedback scores that drive promotion and deprecation decisions.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

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


class FeedbackAggregator:
    """
    Aggregates conversation tags to produce belief feedback scores.

    Scores:
    - feedback_score ∈ [-1, 1]: Overall signal
      - Positive: +keep, +shift (belief is useful/relevant)
      - Negative: +doubt, +artifact (belief is questionable/noise)
    - neg_feedback ∈ [0, 1]: Strength of negative signal specifically
    """

    # Tag weights for scoring
    TAG_WEIGHTS = {
        "+keep": 0.5,      # Strong positive: belief should persist
        "+shift": 0.3,     # Moderate positive: belief is evolving/useful
        "+doubt": -0.4,    # Moderate negative: belief is questionable
        "+artifact": -0.6, # Strong negative: belief is noise/template
    }

    def __init__(self, raw_store: RawStore, config: Optional[FeedbackConfig] = None):
        self.raw_store = raw_store
        self.config = config or FeedbackConfig()

        # Score cache: belief_id -> (feedback_score, neg_feedback, timestamp)
        self._cache: Dict[str, Tuple[float, float, float]] = {}

        # Global score cache (for all beliefs)
        self._global_cache: Optional[Tuple[float, float, float]] = None  # (score, neg, timestamp)

        # Circuit breaker state
        self.circuit_open = False

        # Telemetry
        self.window_size_hours = self.config.window_hours
        self.last_scores: Dict[str, Tuple[float, float]] = {}  # For status endpoint
        self.global_feedback_score: float = 0.0
        self.global_neg_feedback: float = 0.0

        logger.info(f"FeedbackAggregator initialized (window={self.config.window_hours}h)")

    def score(self, belief_id: str, now: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute feedback scores for a belief based on recent conversation tags.

        Args:
            belief_id: ID of belief to score
            now: Current timestamp (default: time.time())

        Returns:
            (feedback_score, neg_feedback) where:
            - feedback_score ∈ [-1, 1]: Overall signal (positive=good, negative=bad)
            - neg_feedback ∈ [0, 1]: Strength of negative signal only
        """
        if now is None:
            now = time.time()

        # Check cache
        if belief_id in self._cache:
            score, neg, cache_ts = self._cache[belief_id]
            age = now - cache_ts
            if age < self.config.cache_ttl_secs:
                logger.debug(f"Cache hit for {belief_id} (age={age:.1f}s)")
                return score, neg

        # Compute fresh scores
        feedback_score, neg_feedback = self._compute_scores(belief_id, now)

        # Update cache
        self._cache[belief_id] = (feedback_score, neg_feedback, now)
        self.last_scores[belief_id] = (feedback_score, neg_feedback)

        logger.debug(f"Scored {belief_id}: feedback={feedback_score:.2f}, neg={neg_feedback:.2f}")
        return feedback_score, neg_feedback

    def _compute_scores(self, belief_id: str, now: float) -> Tuple[float, float]:
        """
        Compute scores by analyzing tags in recent experiences.

        Strategy:
        1. Query recent OCCURRENCE experiences (conversations)
        2. Look for belief_id mentions in structured content or metadata
        3. Extract tags from those experiences
        4. Compute weighted scores based on tag frequency/recency
        """
        # Get cutoff time for trailing window
        cutoff_dt = datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(hours=self.config.window_hours)

        # Query recent conversation experiences within the analysis window
        try:
            recent = self.raw_store.list_recent(
                limit=200,
                experience_type=ExperienceType.OCCURRENCE,
                since=cutoff_dt,
            )
        except Exception as e:
            logger.error(f"Failed to query experiences: {e}")
            return 0.0, 0.0

        # Extract tags related to this belief
        tag_counts = defaultdict(int)
        for exp in recent:
            # Check if experience mentions this belief
            # Look in structured content, metadata, or text
            if self._mentions_belief(exp, belief_id):
                # Extract tags from experience metadata or structured content
                tags = self._extract_tags(exp)
                for tag in tags:
                    if tag in self.TAG_WEIGHTS:
                        tag_counts[tag] += 1

        # Compute scores from tag counts
        if sum(tag_counts.values()) < self.config.min_samples:
            # Not enough data
            return 0.0, 0.0

        # Weighted sum for overall feedback score
        total_weight = 0.0
        for tag, count in tag_counts.items():
            total_weight += self.TAG_WEIGHTS[tag] * count

        # Normalize to [-1, 1]
        total_tags = sum(tag_counts.values())
        feedback_score = total_weight / total_tags if total_tags > 0 else 0.0
        feedback_score = max(-1.0, min(1.0, feedback_score))

        # Compute negative feedback strength [0, 1]
        neg_tags = tag_counts.get("+doubt", 0) + tag_counts.get("+artifact", 0)
        neg_feedback = neg_tags / total_tags if total_tags > 0 else 0.0

        return feedback_score, neg_feedback

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
                # Check for belief_ids from tag injector
                if 'belief_ids' in structured and belief_id in structured.get('belief_ids', []):
                    return True

        # Check metadata
        if hasattr(experience, 'metadata') and experience.metadata:
            if isinstance(experience.metadata, dict):
                if belief_id in str(experience.metadata):
                    return True

        # For now, simple text search (TODO: improve with semantic search)
        if hasattr(experience.content, 'text') and experience.content.text:
            if belief_id in experience.content.text:
                return True

        return False

    def _extract_tags(self, experience) -> List[str]:
        """Extract feedback tags from experience."""
        tags = []

        # Look in structured content
        if hasattr(experience.content, 'structured') and experience.content.structured:
            structured = experience.content.structured
            if isinstance(structured, dict):
                # Check for tags field
                if 'tags' in structured:
                    tag_list = structured['tags']
                    if isinstance(tag_list, list):
                        tags.extend(tag_list)
                    elif isinstance(tag_list, str):
                        tags.append(tag_list)

        # Look in metadata
        if hasattr(experience, 'metadata') and experience.metadata:
            if isinstance(experience.metadata, dict):
                if 'tags' in experience.metadata:
                    meta_tags = experience.metadata['tags']
                    if isinstance(meta_tags, list):
                        tags.extend(meta_tags)
                    elif isinstance(meta_tags, str):
                        tags.append(meta_tags)

        # Look in text content for inline tags
        if hasattr(experience.content, 'text') and experience.content.text:
            text = experience.content.text
            for tag in self.TAG_WEIGHTS.keys():
                if tag in text:
                    tags.append(tag)

        return tags

    def global_score(self, now: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute global feedback scores across all beliefs.

        Returns:
            (global_feedback_score, global_neg_feedback)
        """
        if now is None:
            now = time.time()

        # Check cache
        if self._global_cache:
            score, neg, cache_ts = self._global_cache
            age = now - cache_ts
            if age < self.config.cache_ttl_secs:
                return score, neg

        # Compute fresh global scores
        cutoff_dt = datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(hours=self.config.window_hours)

        try:
            recent = self.raw_store.list_recent(
                limit=200,
                experience_type=ExperienceType.OCCURRENCE,
                since=cutoff_dt,
            )
        except Exception as e:
            logger.error(f"Failed to query experiences for global score: {e}")
            return 0.0, 0.0

        # Extract all tags (global, not belief-specific)
        tag_counts = defaultdict(int)
        for exp in recent:
            tags = self._extract_tags(exp)
            for tag in tags:
                if tag in self.TAG_WEIGHTS:
                    tag_counts[tag] += 1

        # Compute scores
        if sum(tag_counts.values()) < self.config.min_samples:
            score, neg = 0.0, 0.0
        else:
            total_weight = 0.0
            for tag, count in tag_counts.items():
                total_weight += self.TAG_WEIGHTS[tag] * count

            total_tags = sum(tag_counts.values())
            score = total_weight / total_tags if total_tags > 0 else 0.0
            score = max(-1.0, min(1.0, score))

            neg_tags = tag_counts.get("+doubt", 0) + tag_counts.get("+artifact", 0)
            neg = neg_tags / total_tags if total_tags > 0 else 0.0

        # Update cache and telemetry
        self._global_cache = (score, neg, now)
        self.global_feedback_score = score
        self.global_neg_feedback = neg

        # Update circuit breaker
        self.circuit_open = neg > 0.6

        logger.debug(f"Global score: feedback={score:.2f}, neg={neg:.2f}, circuit_open={self.circuit_open}")
        return score, neg

    def get_telemetry(self) -> Dict:
        """Get telemetry data for status endpoint."""
        # Update global scores
        self.global_score()

        return {
            "global_feedback_score": self.global_feedback_score,
            "global_neg_feedback": self.global_neg_feedback,
            "window_minutes": self.window_size_hours * 60,
            "circuit_open": self.circuit_open,
            "cached_beliefs": len(self._cache),
            "last_scores": {
                belief_id: {"feedback": score, "negative": neg}
                for belief_id, (score, neg) in self.last_scores.items()
            },
        }
