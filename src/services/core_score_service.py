"""
Core Score Service for HTN Self-Belief Decomposer.

Computes belief centrality score based on:
- Support: How many occurrences (weighted)
- Spread: How many distinct days
- Diversity: How many distinct contexts
- Conflict penalty: Recent conflicts reduce score

Uses bounded components (sigmoid functions) to prevent single-factor dominance.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid as uuid_module

from sqlmodel import Session, select

from src.utils.belief_config import BeliefSystemConfig, get_belief_config
from src.memory.models.belief_node import BeliefNode
from src.memory.models.belief_occurrence import BeliefOccurrence
from src.services.conflict_engine import ConflictEngine

logger = logging.getLogger(__name__)


@dataclass
class CoreScoreResult:
    """
    Result of core score computation.

    Attributes:
        core_score: The final score
        status: surface, developing, or core
        components: Breakdown of score components
    """
    core_score: float
    status: str
    components: Dict[str, float]


class CoreScoreService:
    """
    Compute belief centrality score with bounded components.

    Components:
    - support = 1 - exp(-n_weighted / k_n)
    - spread = sigmoid((distinct_days - midpoint_days) / temperature_days)
    - diversity = sigmoid((distinct_contexts - midpoint_contexts) / temperature_contexts)

    base = support * spread * diversity

    conflict_penalty = weight * (recent_conflicts / n_weighted)

    core_score = max(0, base - conflict_penalty)

    Status thresholds:
    - core_score >= core_threshold -> "core"
    - core_score >= developing_threshold -> "developing"
    - else -> "surface"
    """

    def __init__(
        self,
        config: Optional[BeliefSystemConfig] = None,
        db_session: Optional[Session] = None
    ):
        """
        Initialize the core score service.

        Args:
            config: Configuration object
            db_session: Database session
        """
        if config is None:
            config = get_belief_config()

        self.config = config.scoring
        self.db = db_session

        # Support parameters
        self.k_n = config.scoring.support['k_n']

        # Spread parameters
        self.spread_midpoint = config.scoring.spread['midpoint_days']
        self.spread_temp = config.scoring.spread['temperature']

        # Diversity parameters
        self.diversity_midpoint = config.scoring.diversity['midpoint_contexts']
        self.diversity_temp = config.scoring.diversity['temperature']

        # Conflict penalty
        self.conflict_enabled = config.scoring.conflict_penalty.enabled
        self.conflict_window = config.scoring.conflict_penalty.recent_window_days
        self.conflict_weight = config.scoring.conflict_penalty.weight

        # Status thresholds
        self.developing_threshold = config.scoring.status_thresholds.developing
        self.core_threshold = config.scoring.status_thresholds.core

    def sigmoid(self, x: float) -> float:
        """Standard logistic sigmoid: 1 / (1 + exp(-x))"""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def compute_core_score(
        self,
        node: BeliefNode,
        occurrences: Optional[List[BeliefOccurrence]] = None,
        conflict_count_recent: Optional[int] = None
    ) -> CoreScoreResult:
        """
        Compute core score for a belief node.

        Args:
            node: The belief node
            occurrences: Optional list of occurrences (will query if not provided)
            conflict_count_recent: Optional recent conflict count

        Returns:
            CoreScoreResult with score, status, and component breakdown
        """
        # Get occurrences if not provided
        if occurrences is None:
            if self.db:
                occurrences = list(self.db.exec(
                    select(BeliefOccurrence).where(
                        BeliefOccurrence.belief_id == node.belief_id,
                        BeliefOccurrence.deleted_at.is_(None)
                    )
                ).all())
            else:
                occurrences = []

        if not occurrences:
            return CoreScoreResult(
                core_score=0.0,
                status='surface',
                components={
                    'support': 0.0,
                    'spread': 0.0,
                    'diversity': 0.0,
                    'base': 0.0,
                    'conflict_penalty': 0.0,
                    'n_weighted': 0.0,
                    'distinct_days': 0,
                    'distinct_contexts': 0,
                }
            )

        # Compute n_weighted
        n_weighted = sum(occ.source_weight for occ in occurrences)

        # Compute distinct days
        dates = set()
        for occ in occurrences:
            dates.add(occ.created_at.date())
        distinct_days = len(dates)

        # Compute distinct contexts
        contexts = set()
        for occ in occurrences:
            contexts.add(occ.context_id)
        distinct_contexts = len(contexts)

        # Compute components
        # Support: saturating function
        support = 1.0 - math.exp(-n_weighted / self.k_n)

        # Spread: sigmoid centered at midpoint
        spread_input = (distinct_days - self.spread_midpoint) / self.spread_temp
        spread = self.sigmoid(spread_input)

        # Diversity: sigmoid centered at midpoint
        diversity_input = (distinct_contexts - self.diversity_midpoint) / self.diversity_temp
        diversity = self.sigmoid(diversity_input)

        # Base score
        base = support * spread * diversity

        # Conflict penalty
        conflict_penalty = 0.0
        if self.conflict_enabled and conflict_count_recent is not None and n_weighted > 0:
            conflict_penalty = self.conflict_weight * (conflict_count_recent / n_weighted)

        # Final score
        core_score = max(0.0, base - conflict_penalty)

        # Determine status
        if core_score >= self.core_threshold:
            status = 'core'
        elif core_score >= self.developing_threshold:
            status = 'developing'
        else:
            status = 'surface'

        return CoreScoreResult(
            core_score=core_score,
            status=status,
            components={
                'support': support,
                'spread': spread,
                'diversity': diversity,
                'base': base,
                'conflict_penalty': conflict_penalty,
                'n_weighted': n_weighted,
                'distinct_days': distinct_days,
                'distinct_contexts': distinct_contexts,
            }
        )

    def update_core_score(
        self,
        node: BeliefNode,
        conflict_engine: Optional[ConflictEngine] = None
    ) -> CoreScoreResult:
        """
        Compute and save core score to the node.

        Args:
            node: The belief node
            conflict_engine: Optional conflict engine for penalty calculation

        Returns:
            CoreScoreResult
        """
        # Get conflict count if engine provided
        conflict_count = None
        if conflict_engine and self.conflict_enabled:
            conflict_count = conflict_engine.count_recent_conflicts(
                node.belief_id,
                self.conflict_window
            )

        result = self.compute_core_score(node, conflict_count_recent=conflict_count)

        node.core_score = result.core_score
        node.status = result.status

        if self.db:
            self.db.add(node)
            self.db.commit()

        return result

    def recompute_all(
        self,
        conflict_engine: Optional[ConflictEngine] = None
    ) -> int:
        """
        Recompute core scores for all nodes.

        Returns:
            Count of nodes updated
        """
        if not self.db:
            return 0

        nodes = self.db.exec(select(BeliefNode)).all()
        count = 0

        for node in nodes:
            self.update_core_score(node, conflict_engine)
            count += 1

        return count
