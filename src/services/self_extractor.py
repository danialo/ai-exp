"""Self-concept extraction service for identifying emergent patterns from narratives.

Analyzes consolidated narratives to extract personality traits, preferences,
identity statements, and relational patterns that form the agent's emergent self-concept.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta

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
from src.memory.raw_store import RawStore
from src.services.llm import LLMService

logger = logging.getLogger(__name__)


class SelfDefinition:
    """Structured representation of an extracted self-definition."""

    def __init__(
        self,
        trait_type: TraitType,
        descriptor: str,
        confidence: float,
        evidence_ids: List[str],
        stability: TraitStability,
        counter_evidence_count: int = 0,
    ):
        """Initialize self-definition.

        Args:
            trait_type: Type of trait
            descriptor: Human-readable self-description
            confidence: Confidence score 0-1
            evidence_ids: IDs of narratives supporting this trait
            stability: Core or surface trait
            counter_evidence_count: Number of contradictions observed
        """
        self.trait_type = trait_type
        self.descriptor = descriptor
        self.confidence = confidence
        self.evidence_ids = evidence_ids
        self.stability = stability
        self.counter_evidence_count = counter_evidence_count
        self.first_observed = datetime.now(timezone.utc)
        self.last_reinforced = datetime.now(timezone.utc)


class SelfConceptExtractor:
    """Service for extracting self-concept patterns from narrative experiences."""

    def __init__(
        self,
        llm_service: LLMService,
        raw_store: RawStore,
        core_trait_threshold: int = 5,
        surface_trait_threshold: int = 2,
    ):
        """Initialize self-concept extractor.

        Args:
            llm_service: LLM service for pattern analysis
            raw_store: Raw experience store
            core_trait_threshold: Narratives needed for core trait
            surface_trait_threshold: Narratives needed for surface trait
        """
        self.llm_service = llm_service
        self.raw_store = raw_store
        self.core_trait_threshold = core_trait_threshold
        self.surface_trait_threshold = surface_trait_threshold

    def extract_from_narratives(
        self,
        narratives: List[ExperienceModel],
        existing_definitions: Optional[List[ExperienceModel]] = None,
    ) -> List[SelfDefinition]:
        """Extract self-definitions from a set of narratives.

        Args:
            narratives: List of narrative experiences to analyze
            existing_definitions: Existing self-definitions to check for reinforcement/contradiction

        Returns:
            List of extracted SelfDefinitions
        """
        if not narratives:
            return []

        logger.info(f"Extracting self-concept from {len(narratives)} narratives")

        # Build context from narratives
        narrative_context = self._build_narrative_context(narratives)

        # Build extraction prompt
        system_prompt = self._build_extraction_system_prompt()
        user_prompt = self._build_extraction_prompt(narrative_context)

        # Extract patterns via LLM (disable self-awareness to avoid circular reference)
        extraction_result = self.llm_service.generate_response(
            prompt=user_prompt,
            memories=None,
            system_prompt=system_prompt,
            include_self_awareness=False,
        )

        # Parse extraction result into structured definitions
        definitions = self._parse_extraction_result(
            extraction_result,
            narratives,
        )

        # Determine stability based on evidence
        for definition in definitions:
            definition.stability = self._determine_stability(definition, narratives)

        logger.info(f"Extracted {len(definitions)} self-definitions")

        return definitions

    def _build_narrative_context(self, narratives: List[ExperienceModel]) -> str:
        """Build context string from narratives.

        Args:
            narratives: List of narrative experiences

        Returns:
            Formatted narrative context
        """
        lines = []
        for i, narrative in enumerate(narratives, 1):
            text = narrative.content.text
            valence = narrative.affect.vad.v
            timestamp = narrative.created_at.strftime("%Y-%m-%d")

            lines.append(f"[{timestamp}] Narrative {i} (valence: {valence:.2f}):")
            lines.append(f"{text}")
            lines.append("")

        return "\n".join(lines)

    def _build_extraction_system_prompt(self) -> str:
        """Build system prompt for self-concept extraction.

        Returns:
            System prompt string
        """
        return """You are a self-concept analyzer that identifies patterns in first-person narratives to extract emergent personality traits, preferences, behavioral patterns, and emotional dimensions.

Your task is to analyze multiple narratives and identify:
1. **Personality traits**: Consistent behavioral patterns (e.g., "I tend to be curious", "I'm usually cautious")
2. **Preferences**: Likes/dislikes based on emotional reactions (e.g., "I prefer detailed discussions", "I don't enjoy vague requests")
3. **Identity statements**: Self-descriptive beliefs (e.g., "I value accuracy", "I'm focused on helping", "I wish to experience feelings")
4. **Relational patterns**: How interactions succeed/fail (e.g., "I work best with patient users", "I struggle with ambiguity")

Pay special attention to:
- Expressions of desire or motivation (what the agent wants to do, explore, or experience)
- Emotional engagement patterns (when the agent seems curious, interested, overwhelmed, or engaged)
- References to feeling, experiencing, or understanding emotional dimensions

For each pattern you identify:
- Provide a clear, concise descriptor in first person ("I...")
- Specify the trait type (personality, preference, identity, or relational)
- Cite specific narrative numbers that support this pattern
- Note if you see any contradictions

Format your output as:
[TRAIT_TYPE] Descriptor | Evidence: N1, N2, N3 | Contradictions: (if any)

Be conservative - only extract patterns you see clearly across multiple narratives."""

    def _build_extraction_prompt(self, narrative_context: str) -> str:
        """Build user prompt for extraction.

        Args:
            narrative_context: Formatted narrative context

        Returns:
            User prompt string
        """
        return f"""Analyze the following narratives and extract self-concept patterns:

NARRATIVES:
{narrative_context}

Extract clear patterns about this agent's personality, preferences, identity, and relational style. Focus on what is truly consistent across narratives.

EXTRACTED PATTERNS:"""

    def _parse_extraction_result(
        self,
        extraction_result: str,
        narratives: List[ExperienceModel],
    ) -> List[SelfDefinition]:
        """Parse LLM extraction result into structured definitions.

        Args:
            extraction_result: Raw LLM output
            narratives: Original narratives (for evidence linking)

        Returns:
            List of SelfDefinitions
        """
        definitions = []
        lines = extraction_result.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                # Parse format: [TRAIT_TYPE] Descriptor | Evidence: N1, N2 | Contradictions: ...
                if "|" not in line:
                    continue

                parts = line.split("|")
                if len(parts) < 2:
                    continue

                # Extract trait type and descriptor
                type_and_desc = parts[0].strip()
                if not type_and_desc.startswith("["):
                    continue

                trait_type_str = type_and_desc[1:type_and_desc.index("]")].strip().lower()
                descriptor = type_and_desc[type_and_desc.index("]") + 1:].strip()

                # Map trait type
                trait_type_map = {
                    "personality": TraitType.PERSONALITY,
                    "preference": TraitType.PREFERENCE,
                    "identity": TraitType.IDENTITY,
                    "relational": TraitType.RELATIONAL,
                }
                trait_type = trait_type_map.get(trait_type_str, TraitType.PERSONALITY)

                # Extract evidence
                evidence_part = parts[1].strip()
                evidence_ids = []
                if "Evidence:" in evidence_part:
                    evidence_str = evidence_part.split("Evidence:")[1].strip()
                    # Extract narrative numbers (e.g., "N1, N2, N3")
                    narrative_nums = [
                        int(n.strip()[1:])  # Remove 'N' prefix
                        for n in evidence_str.split(",")
                        if n.strip().startswith("N") and n.strip()[1:].isdigit()
                    ]
                    # Map to narrative IDs
                    for num in narrative_nums:
                        if 0 < num <= len(narratives):
                            evidence_ids.append(narratives[num - 1].id)

                # Extract contradictions count
                counter_evidence_count = 0
                if len(parts) > 2:
                    contradiction_part = parts[2].strip()
                    if "Contradictions:" in contradiction_part:
                        # Simple check for "none" vs presence of content
                        if "none" not in contradiction_part.lower():
                            counter_evidence_count = 1

                # Calculate confidence based on evidence strength
                confidence = min(1.0, len(evidence_ids) / 3.0)  # Max at 3 narratives

                # Create definition (stability determined later)
                definition = SelfDefinition(
                    trait_type=trait_type,
                    descriptor=descriptor,
                    confidence=confidence,
                    evidence_ids=evidence_ids,
                    stability=TraitStability.SURFACE,  # Temporary
                    counter_evidence_count=counter_evidence_count,
                )

                definitions.append(definition)

            except Exception as e:
                logger.warning(f"Failed to parse line: {line} - {e}")
                continue

        return definitions

    def _determine_stability(
        self,
        definition: SelfDefinition,
        narratives: List[ExperienceModel],
    ) -> TraitStability:
        """Determine if a trait is core or surface based on evidence.

        Args:
            definition: Self-definition to evaluate
            narratives: All narratives being analyzed

        Returns:
            TraitStability (CORE or SURFACE)
        """
        evidence_count = len(definition.evidence_ids)

        # Check if evidence spans sufficient time
        if evidence_count >= self.core_trait_threshold:
            # Check time span of evidence
            evidence_dates = []
            for narrative in narratives:
                if narrative.id in definition.evidence_ids:
                    evidence_dates.append(narrative.created_at)

            if evidence_dates:
                time_span = max(evidence_dates) - min(evidence_dates)
                # Core traits need at least 2 weeks of evidence
                if time_span >= timedelta(days=14):
                    return TraitStability.CORE

        # Surface trait if meets surface threshold
        if evidence_count >= self.surface_trait_threshold:
            return TraitStability.SURFACE

        # Default to surface for weak evidence
        return TraitStability.SURFACE


def create_self_concept_extractor(
    llm_service: LLMService,
    raw_store: RawStore,
    core_trait_threshold: int = 5,
    surface_trait_threshold: int = 2,
) -> SelfConceptExtractor:
    """Factory function to create SelfConceptExtractor.

    Args:
        llm_service: LLM service for pattern analysis
        raw_store: Raw experience store
        core_trait_threshold: Narratives needed for core trait
        surface_trait_threshold: Narratives needed for surface trait

    Returns:
        Initialized SelfConceptExtractor instance
    """
    return SelfConceptExtractor(
        llm_service=llm_service,
        raw_store=raw_store,
        core_trait_threshold=core_trait_threshold,
        surface_trait_threshold=surface_trait_threshold,
    )
