"""Self-consolidation pipeline for extracting and managing emergent self-concept.

Periodically analyzes narratives to extract self-definitions, manages their lifecycle,
and handles reinforcement/contradiction of existing traits.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from src.memory.raw_store import RawStore
from src.memory.vector_store import VectorStore
from src.memory.embedding import EmbeddingProvider
from src.memory.models import (
    ExperienceModel,
    ExperienceType,
    TraitType,
    TraitStability,
    ContentModel,
    ProvenanceModel,
    Actor,
    CaptureMethod,
)
from src.services.self_extractor import SelfConceptExtractor, SelfDefinition

logger = logging.getLogger(__name__)


class SelfConsolidationResult:
    """Result from self-consolidation process."""

    def __init__(
        self,
        new_definitions: List[str],
        updated_definitions: List[str],
        decayed_definitions: List[str],
        narratives_analyzed: int,
        success: bool,
        error: Optional[str] = None,
    ):
        """Initialize result.

        Args:
            new_definitions: IDs of newly created self-definitions
            updated_definitions: IDs of updated self-definitions
            decayed_definitions: IDs of decayed self-definitions
            narratives_analyzed: Number of narratives analyzed
            success: Whether consolidation succeeded
            error: Error message if failed
        """
        self.new_definitions = new_definitions
        self.updated_definitions = updated_definitions
        self.decayed_definitions = decayed_definitions
        self.narratives_analyzed = narratives_analyzed
        self.success = success
        self.error = error


class SelfConsolidationPipeline:
    """Pipeline for extracting and managing emergent self-concept from narratives."""

    def __init__(
        self,
        raw_store: RawStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        self_extractor: SelfConceptExtractor,
        lookback_days: int = 30,
        surface_decay_days: int = 7,
    ):
        """Initialize self-consolidation pipeline.

        Args:
            raw_store: Raw experience store
            vector_store: Vector store for self-definitions
            embedding_provider: Embedding provider
            self_extractor: Self-concept extractor
            lookback_days: Days of narratives to analyze
            surface_decay_days: Days before surface traits decay
        """
        self.raw_store = raw_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.self_extractor = self_extractor
        self.lookback_days = lookback_days
        self.surface_decay_days = surface_decay_days

    def consolidate_self_concept(
        self,
        force_full_analysis: bool = False,
    ) -> SelfConsolidationResult:
        """Run self-consolidation to extract/update self-definitions.

        Args:
            force_full_analysis: Force analysis of all narratives

        Returns:
            SelfConsolidationResult with outcome
        """
        try:
            logger.info("Starting self-concept consolidation")

            # 1. Gather recent narratives
            narratives = self._gather_narratives(force_full_analysis)
            if not narratives:
                return SelfConsolidationResult(
                    new_definitions=[],
                    updated_definitions=[],
                    decayed_definitions=[],
                    narratives_analyzed=0,
                    success=True,
                )

            logger.info(f"Analyzing {len(narratives)} narratives for self-concept patterns")

            # 2. Get existing self-definitions
            existing_definitions = self._get_existing_definitions()

            # 3. Extract new patterns
            extracted_definitions = self.self_extractor.extract_from_narratives(
                narratives=narratives,
                existing_definitions=existing_definitions,
            )

            # 4. Process extracted definitions
            new_ids = []
            updated_ids = []

            for definition in extracted_definitions:
                # Check if similar definition exists
                existing = self._find_similar_definition(definition, existing_definitions)

                if existing:
                    # Update existing definition
                    updated_id = self._update_definition(existing, definition)
                    if updated_id:
                        updated_ids.append(updated_id)
                else:
                    # Create new definition
                    new_id = self._create_definition(definition)
                    if new_id:
                        new_ids.append(new_id)

            # 5. Handle decay of surface traits
            decayed_ids = self._decay_surface_traits()

            logger.info(f"Self-consolidation complete: {len(new_ids)} new, {len(updated_ids)} updated, {len(decayed_ids)} decayed")

            return SelfConsolidationResult(
                new_definitions=new_ids,
                updated_definitions=updated_ids,
                decayed_definitions=decayed_ids,
                narratives_analyzed=len(narratives),
                success=True,
            )

        except Exception as e:
            logger.error(f"Self-consolidation failed: {e}")
            return SelfConsolidationResult(
                new_definitions=[],
                updated_definitions=[],
                decayed_definitions=[],
                narratives_analyzed=0,
                success=False,
                error=str(e),
            )

    def _gather_narratives(self, force_full: bool) -> List[ExperienceModel]:
        """Gather narratives for analysis.

        Args:
            force_full: Force gathering all narratives

        Returns:
            List of narrative experiences
        """
        from sqlmodel import Session as DBSession, select
        from src.memory.models import Experience

        narratives = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)

        with DBSession(self.raw_store.engine) as session:
            statement = (
                select(Experience)
                .where(Experience.type == ExperienceType.OBSERVATION.value)
            )

            if not force_full:
                statement = statement.where(Experience.created_at >= cutoff_date)

            statement = statement.order_by(Experience.created_at.desc())

            for exp in session.exec(statement).all():
                # Check if it's a narrative (not a reflection)
                if exp.content.get("structured", {}).get("type") == "consolidated_narrative":
                    exp_model = self._db_to_model(exp)
                    narratives.append(exp_model)

        return narratives

    def _get_existing_definitions(self) -> List[ExperienceModel]:
        """Get existing self-definition experiences.

        Returns:
            List of self-definition experiences
        """
        from sqlmodel import Session as DBSession, select
        from src.memory.models import Experience

        definitions = []

        with DBSession(self.raw_store.engine) as session:
            statement = (
                select(Experience)
                .where(Experience.type == ExperienceType.SELF_DEFINITION.value)
                .order_by(Experience.created_at.desc())
            )

            for exp in session.exec(statement).all():
                exp_model = self._db_to_model(exp)
                definitions.append(exp_model)

        return definitions

    def _find_similar_definition(
        self,
        new_definition: SelfDefinition,
        existing: List[ExperienceModel],
    ) -> Optional[ExperienceModel]:
        """Find if a similar self-definition already exists.

        Args:
            new_definition: Newly extracted definition
            existing: Existing self-definition experiences

        Returns:
            Existing experience if similar, None otherwise
        """
        for exp in existing:
            # Check if same trait type
            structured = exp.content.structured
            if structured.get("trait_type") != new_definition.trait_type.value:
                continue

            # Check if descriptors are similar (simple string matching for now)
            existing_desc = structured.get("descriptor", "").lower()
            new_desc = new_definition.descriptor.lower()

            # Simple similarity: check if key words overlap
            existing_words = set(existing_desc.split())
            new_words = set(new_desc.split())
            overlap = len(existing_words & new_words) / max(len(existing_words), len(new_words), 1)

            if overlap > 0.6:  # 60% word overlap threshold
                return exp

        return None

    def _create_definition(self, definition: SelfDefinition) -> str:
        """Create a new self-definition experience.

        Args:
            definition: Self-definition to create

        Returns:
            Created experience ID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        definition_id = f"self_{definition.trait_type.value}_{timestamp}"

        experience = ExperienceModel(
            id=definition_id,
            type=ExperienceType.SELF_DEFINITION,
            content=ContentModel(
                text=definition.descriptor,
                structured={
                    "trait_type": definition.trait_type.value,
                    "descriptor": definition.descriptor,
                    "confidence": definition.confidence,
                    "stability": definition.stability.value,
                    "first_observed": definition.first_observed.isoformat(),
                    "last_reinforced": definition.last_reinforced.isoformat(),
                    "counter_evidence_count": definition.counter_evidence_count,
                },
            ),
            provenance=ProvenanceModel(
                actor=Actor.AGENT,
                method=CaptureMethod.MODEL_INFER,
            ),
            parents=definition.evidence_ids,
        )

        self.raw_store.append_experience(experience)

        # Embed and index for retrieval
        embedding = self.embedding_provider.embed(definition.descriptor)
        self.vector_store.upsert(
            id=definition_id,
            vector=embedding,
            metadata={
                "type": "self_definition",
                "trait_type": definition.trait_type.value,
                "stability": definition.stability.value,
                "confidence": definition.confidence,
            },
        )

        logger.info(f"Created self-definition: {definition.descriptor}")
        return definition_id

    def _update_definition(
        self,
        existing: ExperienceModel,
        new_definition: SelfDefinition,
    ) -> str:
        """Update an existing self-definition (by creating new version).

        Args:
            existing: Existing self-definition experience
            new_definition: New definition with updated evidence

        Returns:
            Updated experience ID
        """
        # Since experiences are immutable, create a new version
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        definition_id = f"self_{new_definition.trait_type.value}_{timestamp}_v2"

        # Merge evidence
        existing_evidence = existing.parents
        merged_evidence = list(set(existing_evidence + new_definition.evidence_ids))

        # Update confidence (weighted average)
        existing_conf = existing.content.structured.get("confidence", 0.5)
        new_conf = (existing_conf * 0.7 + new_definition.confidence * 0.3)  # Bias toward existing

        # Increment counter_evidence if contradictions found
        counter_count = existing.content.structured.get("counter_evidence_count", 0)
        counter_count += new_definition.counter_evidence_count

        experience = ExperienceModel(
            id=definition_id,
            type=ExperienceType.SELF_DEFINITION,
            content=ContentModel(
                text=new_definition.descriptor,
                structured={
                    "trait_type": new_definition.trait_type.value,
                    "descriptor": new_definition.descriptor,
                    "confidence": new_conf,
                    "stability": new_definition.stability.value,
                    "first_observed": existing.content.structured.get("first_observed"),
                    "last_reinforced": datetime.now(timezone.utc).isoformat(),
                    "counter_evidence_count": counter_count,
                },
            ),
            provenance=ProvenanceModel(
                actor=Actor.AGENT,
                method=CaptureMethod.MODEL_INFER,
            ),
            parents=merged_evidence + [existing.id],  # Link to previous version
        )

        self.raw_store.append_experience(experience)

        # Update vector index
        embedding = self.embedding_provider.embed(new_definition.descriptor)
        self.vector_store.upsert(
            id=definition_id,
            vector=embedding,
            metadata={
                "type": "self_definition",
                "trait_type": new_definition.trait_type.value,
                "stability": new_definition.stability.value,
                "confidence": new_conf,
            },
        )

        logger.info(f"Updated self-definition: {new_definition.descriptor}")
        return definition_id

    def _decay_surface_traits(self) -> List[str]:
        """Decay surface traits that haven't been reinforced recently.

        Returns:
            List of decayed definition IDs
        """
        existing_definitions = self._get_existing_definitions()
        decayed_ids = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.surface_decay_days)

        for exp in existing_definitions:
            structured = exp.content.structured

            # Only decay surface traits
            if structured.get("stability") != TraitStability.SURFACE.value:
                continue

            # Check if last reinforced is older than cutoff
            last_reinforced_str = structured.get("last_reinforced")
            if last_reinforced_str:
                last_reinforced = datetime.fromisoformat(last_reinforced_str)
                if last_reinforced.tzinfo is None:
                    last_reinforced = last_reinforced.replace(tzinfo=timezone.utc)

                if last_reinforced < cutoff_date:
                    # Reduce confidence (create updated version with lower confidence)
                    # For simplicity, just log for now
                    logger.info(f"Surface trait decaying: {structured.get('descriptor')}")
                    decayed_ids.append(exp.id)

        return decayed_ids

    def _db_to_model(self, exp) -> ExperienceModel:
        """Convert DB experience to ExperienceModel.

        Args:
            exp: Database experience object

        Returns:
            ExperienceModel
        """
        from src.memory.models import experience_to_model
        return experience_to_model(exp)


def create_self_consolidation_pipeline(
    raw_store: RawStore,
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
    self_extractor: SelfConceptExtractor,
    lookback_days: int = 30,
    surface_decay_days: int = 7,
) -> SelfConsolidationPipeline:
    """Factory function to create SelfConsolidationPipeline.

    Args:
        raw_store: Raw experience store
        vector_store: Vector store for self-definitions
        embedding_provider: Embedding provider
        self_extractor: Self-concept extractor
        lookback_days: Days of narratives to analyze
        surface_decay_days: Days before surface traits decay

    Returns:
        Initialized SelfConsolidationPipeline instance
    """
    return SelfConsolidationPipeline(
        raw_store=raw_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        self_extractor=self_extractor,
        lookback_days=lookback_days,
        surface_decay_days=surface_decay_days,
    )
