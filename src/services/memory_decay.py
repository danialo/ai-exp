"""Memory decay calculation service.

Implements multi-factor decay scoring based on:
- Access frequency: More accessed memories decay slower
- Emotional salience: High-valence memories resist decay
- Information entropy: Unique information decays slower
- Time-based decay: Exponential decay baseline
"""

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List

from sqlalchemy import create_engine
from sqlmodel import Session as DBSession, select
import numpy as np

from src.memory.models import MemoryDecayMetrics, ExperienceModel
from src.memory.embedding import EmbeddingProvider


class MemoryDecayCalculator:
    """Service for calculating and updating memory decay factors."""

    def __init__(
        self,
        db_path: str | Path,
        embedding_provider: Optional[EmbeddingProvider] = None,
        time_decay_halflife_days: float = 30.0,
        access_weight: float = 0.3,
        salience_weight: float = 0.3,
        entropy_weight: float = 0.2,
        time_weight: float = 0.2,
    ):
        """Initialize memory decay calculator.

        Args:
            db_path: Path to SQLite database
            embedding_provider: Optional embedding provider for entropy calculation
            time_decay_halflife_days: Half-life for time-based decay in days
            access_weight: Weight for access frequency factor (0-1)
            salience_weight: Weight for emotional salience factor (0-1)
            entropy_weight: Weight for information entropy factor (0-1)
            time_weight: Weight for time-based decay factor (0-1)
        """
        self.db_path = Path(db_path)
        self.embedding_provider = embedding_provider
        self.time_decay_halflife_days = time_decay_halflife_days
        self.access_weight = access_weight
        self.salience_weight = salience_weight
        self.entropy_weight = entropy_weight
        self.time_weight = time_weight

        # Validate weights sum to 1.0
        total_weight = access_weight + salience_weight + entropy_weight + time_weight
        if not math.isclose(total_weight, 1.0, abs_tol=0.01):
            raise ValueError(f"Decay weights must sum to 1.0, got {total_weight}")

        # Create engine
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})

    def initialize_metrics(self, experience: ExperienceModel) -> MemoryDecayMetrics:
        """Initialize decay metrics for a new experience.

        Args:
            experience: Experience to initialize metrics for

        Returns:
            Created MemoryDecayMetrics object
        """
        # Calculate initial emotional salience
        emotional_salience = abs(experience.affect.vad.v)

        metrics = MemoryDecayMetrics(
            experience_id=experience.id,
            access_count=0,
            last_accessed=None,
            entropy_score=0.5,  # Default medium entropy, will be calculated later
            emotional_salience=emotional_salience,
            decay_factor=1.0,  # No decay initially
            last_calculated=datetime.now(timezone.utc),
        )

        with DBSession(self.engine) as db:
            db.add(metrics)
            db.commit()
            db.refresh(metrics)

        return metrics

    def record_access(self, experience_id: str) -> bool:
        """Record that an experience was accessed (retrieved).

        Args:
            experience_id: Experience ID

        Returns:
            True if recorded, False if metrics not found
        """
        with DBSession(self.engine) as db:
            metrics = db.get(MemoryDecayMetrics, experience_id)
            if not metrics:
                return False

            metrics.access_count += 1
            metrics.last_accessed = datetime.now(timezone.utc)
            db.add(metrics)
            db.commit()
            return True

    def calculate_decay_factor(
        self,
        experience: ExperienceModel,
        metrics: MemoryDecayMetrics,
        all_embeddings: Optional[List[np.ndarray]] = None,
    ) -> float:
        """Calculate decay factor for an experience.

        Args:
            experience: Experience object
            metrics: Current decay metrics
            all_embeddings: Optional list of all embeddings for entropy calculation

        Returns:
            Decay factor (0.0 = fully decayed, 1.0 = no decay)
        """
        # 1. Access frequency factor (more accesses = slower decay)
        # Use logarithmic scaling: log(access_count + 1) / log(10)
        access_factor = min(1.0, math.log(metrics.access_count + 1) / math.log(10))

        # 2. Emotional salience factor (high |valence| = slower decay)
        salience_factor = metrics.emotional_salience

        # 3. Information entropy factor (unique = slower decay)
        entropy_factor = metrics.entropy_score

        # 4. Time-based decay factor (exponential decay)
        created_at = experience.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        age_days = (datetime.now(timezone.utc) - created_at).total_seconds() / 86400.0
        time_factor = 0.5 ** (age_days / self.time_decay_halflife_days)

        # Weighted combination
        decay_factor = (
            self.access_weight * access_factor +
            self.salience_weight * salience_factor +
            self.entropy_weight * entropy_factor +
            self.time_weight * time_factor
        )

        return max(0.0, min(1.0, decay_factor))

    def update_entropy_score(
        self,
        experience_id: str,
        embedding: np.ndarray,
        all_embeddings: List[np.ndarray],
    ) -> bool:
        """Update entropy score for an experience based on embedding uniqueness.

        Args:
            experience_id: Experience ID
            embedding: Embedding vector for this experience
            all_embeddings: List of all other embeddings for comparison

        Returns:
            True if updated, False if metrics not found
        """
        if not all_embeddings:
            return False

        # Calculate average cosine similarity to all other embeddings
        similarities = []
        for other_emb in all_embeddings:
            # Cosine similarity
            sim = np.dot(embedding, other_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(other_emb) + 1e-10
            )
            similarities.append(sim)

        # Entropy score: 1 - avg_similarity
        # (high similarity = low uniqueness = low entropy)
        avg_similarity = np.mean(similarities)
        entropy_score = max(0.0, min(1.0, 1.0 - avg_similarity))

        with DBSession(self.engine) as db:
            metrics = db.get(MemoryDecayMetrics, experience_id)
            if not metrics:
                return False

            metrics.entropy_score = entropy_score
            db.add(metrics)
            db.commit()
            return True

    def recalculate_all_decay(
        self,
        experiences: List[ExperienceModel],
    ) -> int:
        """Recalculate decay factors for all experiences.

        Args:
            experiences: List of all experiences to recalculate

        Returns:
            Number of experiences updated
        """
        updated_count = 0

        with DBSession(self.engine) as db:
            for exp in experiences:
                metrics = db.get(MemoryDecayMetrics, exp.id)
                if not metrics:
                    # Initialize if missing
                    metrics = self.initialize_metrics(exp)

                # Calculate new decay factor
                new_decay = self.calculate_decay_factor(exp, metrics)

                # Update metrics
                metrics.decay_factor = new_decay
                metrics.last_calculated = datetime.now(timezone.utc)
                db.add(metrics)
                updated_count += 1

            db.commit()

        return updated_count

    def get_metrics(self, experience_id: str) -> Optional[MemoryDecayMetrics]:
        """Get decay metrics for an experience.

        Args:
            experience_id: Experience ID

        Returns:
            MemoryDecayMetrics or None if not found
        """
        with DBSession(self.engine) as db:
            return db.get(MemoryDecayMetrics, experience_id)

    def close(self):
        """Close database connection."""
        self.engine.dispose()


def create_memory_decay_calculator(
    db_path: Optional[str | Path] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    time_decay_halflife_days: float = 30.0,
) -> MemoryDecayCalculator:
    """Factory function to create MemoryDecayCalculator instance.

    Args:
        db_path: Database path (defaults to data/raw_store.db)
        embedding_provider: Optional embedding provider
        time_decay_halflife_days: Half-life for time decay in days

    Returns:
        Initialized MemoryDecayCalculator instance
    """
    if db_path is None:
        db_path = Path("data/raw_store.db")
    return MemoryDecayCalculator(
        db_path,
        embedding_provider=embedding_provider,
        time_decay_halflife_days=time_decay_halflife_days,
    )
