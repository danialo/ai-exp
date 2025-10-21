"""Consolidation pipeline for transforming session experiences into long-term narratives.

Orchestrates the process of:
1. Gathering session experiences
2. Generating first-person narrative
3. Creating OBSERVATION experience
4. Indexing in long-term vector store
5. Marking originals as consolidated
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List

from src.memory.raw_store import RawStore
from src.memory.vector_store import VectorStore
from src.memory.embedding import EmbeddingProvider
from src.memory.models import (
    ExperienceModel,
    ExperienceType,
    Actor,
    CaptureMethod,
    ContentModel,
    ProvenanceModel,
)
from src.services.session_tracker import SessionTracker
from src.services.narrative_transformer import NarrativeTransformer
from src.services.memory_decay import MemoryDecayCalculator
from src.services.emotional_extractor import EmotionalExperienceExtractor

logger = logging.getLogger(__name__)


class ConsolidationResult:
    """Result from session consolidation."""

    def __init__(
        self,
        session_id: str,
        narrative_id: str,
        narrative_text: str,
        consolidated_count: int,
        success: bool,
        error: Optional[str] = None,
    ):
        """Initialize result.

        Args:
            session_id: Session ID
            narrative_id: ID of created narrative experience
            narrative_text: Generated narrative text
            consolidated_count: Number of experiences consolidated
            success: Whether consolidation succeeded
            error: Error message if failed
        """
        self.session_id = session_id
        self.narrative_id = narrative_id
        self.narrative_text = narrative_text
        self.consolidated_count = consolidated_count
        self.success = success
        self.error = error


class ConsolidationPipeline:
    """Pipeline for consolidating session experiences into long-term narratives."""

    def __init__(
        self,
        raw_store: RawStore,
        short_term_store: VectorStore,
        long_term_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        session_tracker: SessionTracker,
        narrative_transformer: NarrativeTransformer,
        decay_calculator: Optional[MemoryDecayCalculator] = None,
        emotional_extractor: Optional[EmotionalExperienceExtractor] = None,
    ):
        """Initialize consolidation pipeline.

        Args:
            raw_store: Raw experience store
            short_term_store: Short-term vector store
            long_term_store: Long-term vector store
            embedding_provider: Embedding provider
            session_tracker: Session tracker service
            narrative_transformer: Narrative transformation service
            decay_calculator: Optional decay calculator
            emotional_extractor: Optional emotional dimension extractor
        """
        self.raw_store = raw_store
        self.short_term_store = short_term_store
        self.long_term_store = long_term_store
        self.embedding_provider = embedding_provider
        self.session_tracker = session_tracker
        self.narrative_transformer = narrative_transformer
        self.decay_calculator = decay_calculator
        self.emotional_extractor = emotional_extractor

    def consolidate_session(self, session_id: str) -> ConsolidationResult:
        """Consolidate a session into a long-term narrative memory.

        Args:
            session_id: Session ID to consolidate

        Returns:
            ConsolidationResult with outcome
        """
        try:
            # 1. Get session and validate
            session = self.session_tracker.get_session(session_id)
            if not session:
                return ConsolidationResult(
                    session_id=session_id,
                    narrative_id="",
                    narrative_text="",
                    consolidated_count=0,
                    success=False,
                    error="Session not found",
                )

            if not session.experience_ids:
                return ConsolidationResult(
                    session_id=session_id,
                    narrative_id="",
                    narrative_text="",
                    consolidated_count=0,
                    success=False,
                    error="No experiences in session",
                )

            logger.info(f"Consolidating session {session_id} with {len(session.experience_ids)} experiences")

            # 2. Gather all session experiences
            experiences = []
            for exp_id in session.experience_ids:
                exp = self.raw_store.get_experience(exp_id)
                if exp:
                    experiences.append(exp)

            if not experiences:
                return ConsolidationResult(
                    session_id=session_id,
                    narrative_id="",
                    narrative_text="",
                    consolidated_count=0,
                    success=False,
                    error="Could not load experiences",
                )

            # 3. Generate first-person narrative
            narrative_text = self.narrative_transformer.transform_session(
                experiences=experiences,
                include_emotional_meta=True,
            )

            logger.info(f"Generated narrative: {narrative_text[:100]}...")

            # 3.5. Extract emotional dimensions if extractor available
            emotional_state = None
            if self.emotional_extractor:
                # Create a temporary narrative model for extraction
                temp_narrative = ExperienceModel(
                    id="temp_narrative",
                    type=ExperienceType.OBSERVATION,
                    content=ContentModel(text=narrative_text),
                    provenance=ProvenanceModel(
                        actor=Actor.AGENT,
                        method=CaptureMethod.MODEL_INFER,
                    ),
                )

                try:
                    emotional_dims = self.emotional_extractor.extract_emotional_dimensions(temp_narrative)
                    emotional_state = emotional_dims.to_dict()
                    logger.info(f"Extracted emotional state: {emotional_state.get('felt_emotions', [])}")
                except Exception as e:
                    logger.warning(f"Failed to extract emotional dimensions: {e}")

            # 4. Create OBSERVATION experience for narrative
            narrative_id = self._create_narrative_experience(
                session_id=session_id,
                narrative_text=narrative_text,
                parent_ids=[exp.id for exp in experiences],
                emotional_state=emotional_state,
            )

            # 5. Index narrative in long-term vector store
            narrative_embedding = self.embedding_provider.embed(narrative_text)
            self.long_term_store.upsert(
                id=narrative_id,
                vector=narrative_embedding,
                metadata={
                    "type": "narrative",
                    "session_id": session_id,
                    "experience_count": len(experiences),
                },
            )

            logger.info(f"Indexed narrative {narrative_id} in long-term store")

            # 6. Mark original experiences as consolidated
            self._mark_experiences_consolidated(experiences, session_id)

            # 7. Initialize decay metrics for narrative
            if self.decay_calculator:
                narrative_exp = self.raw_store.get_experience(narrative_id)
                if narrative_exp:
                    self.decay_calculator.initialize_metrics(narrative_exp)

            # 8. Mark session as consolidated
            self.session_tracker.mark_consolidated(session_id, narrative_id)

            logger.info(f"Successfully consolidated session {session_id}")

            return ConsolidationResult(
                session_id=session_id,
                narrative_id=narrative_id,
                narrative_text=narrative_text,
                consolidated_count=len(experiences),
                success=True,
            )

        except Exception as e:
            logger.error(f"Consolidation failed for session {session_id}: {e}")
            return ConsolidationResult(
                session_id=session_id,
                narrative_id="",
                narrative_text="",
                consolidated_count=0,
                success=False,
                error=str(e),
            )

    def _create_narrative_experience(
        self,
        session_id: str,
        narrative_text: str,
        parent_ids: List[str],
        emotional_state: Optional[dict] = None,
    ) -> str:
        """Create an OBSERVATION experience for the narrative.

        Args:
            session_id: Session ID
            narrative_text: Generated narrative text
            parent_ids: IDs of parent experiences
            emotional_state: Optional emotional dimensions extracted from narrative

        Returns:
            Created experience ID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        narrative_id = f"narrative_{session_id}_{timestamp}"

        # Build structured content with emotional state if available
        structured_content = {
            "session_id": session_id,
            "type": "consolidated_narrative",
        }
        if emotional_state:
            structured_content["emotional_state"] = emotional_state

        narrative_exp = ExperienceModel(
            id=narrative_id,
            type=ExperienceType.OBSERVATION,
            content=ContentModel(
                text=narrative_text,
                structured=structured_content,
            ),
            provenance=ProvenanceModel(
                actor=Actor.AGENT,
                method=CaptureMethod.MODEL_INFER,
            ),
            parents=parent_ids,
            session_id=session_id,
            consolidated=False,  # The narrative itself is not consolidated
        )

        self.raw_store.append_experience(narrative_exp)
        return narrative_id

    def _mark_experiences_consolidated(
        self,
        experiences: List[ExperienceModel],
        session_id: str,
    ) -> None:
        """Mark experiences as consolidated in database.

        Args:
            experiences: List of experiences
            session_id: Session ID
        """
        # Import here to avoid circular dependency
        from sqlmodel import Session as DBSession, select
        from src.memory.models import Experience

        with DBSession(self.raw_store.engine) as db:
            for exp in experiences:
                db_exp = db.get(Experience, exp.id)
                if db_exp:
                    db_exp.consolidated = True
                    db.add(db_exp)

            db.commit()

        logger.info(f"Marked {len(experiences)} experiences as consolidated")

    def consolidate_all_ended_sessions(self, limit: int = 10) -> List[ConsolidationResult]:
        """Consolidate all ended sessions.

        Args:
            limit: Maximum number of sessions to consolidate

        Returns:
            List of ConsolidationResults
        """
        ended_sessions = self.session_tracker.list_ended_sessions(limit=limit)
        results = []

        for session in ended_sessions:
            result = self.consolidate_session(session.id)
            results.append(result)

        return results


def create_consolidation_pipeline(
    raw_store: RawStore,
    short_term_store: VectorStore,
    long_term_store: VectorStore,
    embedding_provider: EmbeddingProvider,
    session_tracker: SessionTracker,
    narrative_transformer: NarrativeTransformer,
    decay_calculator: Optional[MemoryDecayCalculator] = None,
    emotional_extractor: Optional[EmotionalExperienceExtractor] = None,
) -> ConsolidationPipeline:
    """Factory function to create ConsolidationPipeline.

    Args:
        raw_store: Raw experience store
        short_term_store: Short-term vector store
        long_term_store: Long-term vector store
        embedding_provider: Embedding provider
        session_tracker: Session tracker
        narrative_transformer: Narrative transformer
        decay_calculator: Optional decay calculator
        emotional_extractor: Optional emotional dimension extractor

    Returns:
        Initialized ConsolidationPipeline instance
    """
    return ConsolidationPipeline(
        raw_store=raw_store,
        short_term_store=short_term_store,
        long_term_store=long_term_store,
        embedding_provider=embedding_provider,
        session_tracker=session_tracker,
        narrative_transformer=narrative_transformer,
        decay_calculator=decay_calculator,
        emotional_extractor=emotional_extractor,
    )
