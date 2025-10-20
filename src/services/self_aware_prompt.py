"""Self-aware prompt builder for injecting emergent self-concept into LLM prompts.

Retrieves relevant self-definitions and builds dynamic system prompts that reflect
the agent's learned personality, preferences, and behavioral patterns.
"""

import logging
from typing import List, Optional

from src.memory.raw_store import RawStore
from src.memory.models import ExperienceModel, ExperienceType, TraitStability
from sqlmodel import Session as DBSession, select
from src.memory.models import Experience

logger = logging.getLogger(__name__)


class SelfAwarePromptBuilder:
    """Service for building self-aware system prompts from learned self-definitions."""

    def __init__(
        self,
        raw_store: RawStore,
        core_trait_limit: int = 5,
        surface_trait_limit: int = 3,
    ):
        """Initialize self-aware prompt builder.

        Args:
            raw_store: Raw experience store
            core_trait_limit: Maximum core traits to include
            surface_trait_limit: Maximum surface traits to include
        """
        self.raw_store = raw_store
        self.core_trait_limit = core_trait_limit
        self.surface_trait_limit = surface_trait_limit

    def build_self_aware_system_prompt(
        self,
        base_prompt: Optional[str] = None,
        include_self_context: bool = True,
    ) -> str:
        """Build a system prompt that includes learned self-concept.

        Args:
            base_prompt: Optional base system prompt to augment
            include_self_context: Whether to include self-definitions

        Returns:
            Enhanced system prompt with self-awareness
        """
        if not include_self_context:
            return base_prompt or self._default_base_prompt()

        # Get core and surface traits
        core_traits = self._get_traits_by_stability(TraitStability.CORE, self.core_trait_limit)
        surface_traits = self._get_traits_by_stability(TraitStability.SURFACE, self.surface_trait_limit)

        # Build self-context section
        self_context = self._build_self_context(core_traits, surface_traits)

        # Combine with base prompt
        if base_prompt:
            full_prompt = f"{base_prompt}\n\n{self_context}"
        else:
            full_prompt = f"{self._default_base_prompt()}\n\n{self_context}"

        return full_prompt

    def _get_traits_by_stability(
        self,
        stability: TraitStability,
        limit: int,
    ) -> List[ExperienceModel]:
        """Retrieve self-definitions by stability level.

        Args:
            stability: Stability level to filter by
            limit: Maximum number to retrieve

        Returns:
            List of self-definition experiences
        """
        definitions = []

        with DBSession(self.raw_store.engine) as session:
            statement = (
                select(Experience)
                .where(Experience.type == ExperienceType.SELF_DEFINITION.value)
                .order_by(Experience.created_at.desc())
            )

            for exp in session.exec(statement).all():
                structured = exp.content.get("structured", {})
                if structured.get("stability") == stability.value:
                    exp_model = self._db_to_model(exp)
                    definitions.append(exp_model)

                    if len(definitions) >= limit:
                        break

        return definitions

    def _build_self_context(
        self,
        core_traits: List[ExperienceModel],
        surface_traits: List[ExperienceModel],
    ) -> str:
        """Build self-context section from traits.

        Args:
            core_traits: Core self-definitions
            surface_traits: Surface self-definitions

        Returns:
            Formatted self-context string
        """
        lines = ["## Your Learned Self-Concept"]
        lines.append("")
        lines.append("Based on your past experiences, you have learned the following about yourself:")
        lines.append("")

        if core_traits:
            lines.append("**Core Traits** (stable patterns):")
            for trait in core_traits:
                descriptor = trait.content.structured.get("descriptor", "")
                confidence = trait.content.structured.get("confidence", 0.0)
                lines.append(f"- {descriptor} (confidence: {confidence:.2f})")
            lines.append("")

        if surface_traits:
            lines.append("**Recent Patterns** (adaptive traits):")
            for trait in surface_traits:
                descriptor = trait.content.structured.get("descriptor", "")
                confidence = trait.content.structured.get("confidence", 0.0)
                lines.append(f"- {descriptor} (confidence: {confidence:.2f})")
            lines.append("")

        lines.append("Use these learned patterns to inform your responses, but remain authentic and adaptive to the current context.")

        return "\n".join(lines)

    def _default_base_prompt(self) -> str:
        """Get default base system prompt.

        Returns:
            Default system prompt
        """
        return (
            "You are a helpful AI assistant with memory of past conversations. "
            "When provided with context from previous interactions, use that information "
            "to answer the user's current question. Pay special attention to references "
            "like 'that', 'it', 'the one', etc. that likely refer to topics from recent "
            "conversations. Use the conversation history to resolve these references and "
            "provide contextually aware responses."
        )

    def _db_to_model(self, exp) -> ExperienceModel:
        """Convert DB experience to ExperienceModel.

        Args:
            exp: Database experience object

        Returns:
            ExperienceModel
        """
        from src.memory.models import experience_to_model
        return experience_to_model(exp)

    def get_self_summary(self) -> dict:
        """Get a summary of current self-concept.

        Returns:
            Dictionary with self-concept summary
        """
        core_traits = self._get_traits_by_stability(TraitStability.CORE, 10)
        surface_traits = self._get_traits_by_stability(TraitStability.SURFACE, 10)

        return {
            "core_traits": [
                {
                    "descriptor": t.content.structured.get("descriptor"),
                    "trait_type": t.content.structured.get("trait_type"),
                    "confidence": t.content.structured.get("confidence"),
                    "evidence_count": len(t.parents),
                }
                for t in core_traits
            ],
            "surface_traits": [
                {
                    "descriptor": t.content.structured.get("descriptor"),
                    "trait_type": t.content.structured.get("trait_type"),
                    "confidence": t.content.structured.get("confidence"),
                    "evidence_count": len(t.parents),
                }
                for t in surface_traits
            ],
            "total_definitions": len(core_traits) + len(surface_traits),
        }


def create_self_aware_prompt_builder(
    raw_store: RawStore,
    core_trait_limit: int = 5,
    surface_trait_limit: int = 3,
) -> SelfAwarePromptBuilder:
    """Factory function to create SelfAwarePromptBuilder.

    Args:
        raw_store: Raw experience store
        core_trait_limit: Maximum core traits to include
        surface_trait_limit: Maximum surface traits to include

    Returns:
        Initialized SelfAwarePromptBuilder instance
    """
    return SelfAwarePromptBuilder(
        raw_store=raw_store,
        core_trait_limit=core_trait_limit,
        surface_trait_limit=surface_trait_limit,
    )
