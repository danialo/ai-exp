"""Narrative transformation service for converting session experiences into first-person narratives.

Transforms raw session interactions into emotionally-aware, first-person episodic memories.
"""

from typing import List, Optional
from datetime import datetime

from src.memory.models import ExperienceModel, Actor
from src.services.llm import LLMService


class NarrativeTransformer:
    """Service for transforming session experiences into first-person narratives."""

    def __init__(self, llm_service: LLMService):
        """Initialize narrative transformer.

        Args:
            llm_service: LLM service for generating narratives
        """
        self.llm_service = llm_service

    def transform_session(
        self,
        experiences: List[ExperienceModel],
        include_emotional_meta: bool = True,
    ) -> str:
        """Transform a list of session experiences into a first-person narrative.

        Args:
            experiences: List of experiences from the session
            include_emotional_meta: Include emotional meta-reflection

        Returns:
            First-person narrative string
        """
        if not experiences:
            return ""

        # Build context from experiences
        interaction_context = self._build_interaction_context(experiences)
        emotional_arc = self._extract_emotional_arc(experiences)

        # Build prompt for narrative generation
        system_prompt = self._build_narrative_system_prompt(include_emotional_meta)
        user_prompt = self._build_narrative_prompt(
            interaction_context,
            emotional_arc,
            include_emotional_meta,
        )

        # Generate narrative
        narrative = self.llm_service.generate_response(
            prompt=user_prompt,
            memories=None,  # Don't use memory retrieval for this
            system_prompt=system_prompt,
            include_self_awareness=False,  # Narratives should reflect actual experience, not self-concept
        )

        return narrative

    def _build_interaction_context(self, experiences: List[ExperienceModel]) -> str:
        """Build interaction context from experiences.

        Args:
            experiences: List of experiences

        Returns:
            Formatted interaction context
        """
        lines = []
        for i, exp in enumerate(experiences, 1):
            # Extract prompt/response from content
            prompt = exp.content.structured.get("prompt", "")
            response = exp.content.structured.get("response", "")
            valence = exp.affect.vad.v
            timestamp = exp.created_at.strftime("%H:%M")

            lines.append(f"[{timestamp}] Interaction {i}:")
            lines.append(f"  User: {prompt}")
            lines.append(f"  Assistant: {response[:200]}..." if len(response) > 200 else f"  Assistant: {response}")
            lines.append(f"  Emotional valence: {valence:.2f}")
            lines.append("")

        return "\n".join(lines)

    def _extract_emotional_arc(self, experiences: List[ExperienceModel]) -> str:
        """Extract emotional arc from experiences.

        Args:
            experiences: List of experiences

        Returns:
            Formatted emotional arc description
        """
        if not experiences:
            return "neutral throughout"

        valences = [exp.affect.vad.v for exp in experiences]
        avg_valence = sum(valences) / len(valences)

        # Detect emotional trajectory
        if len(valences) >= 3:
            start_avg = sum(valences[:len(valences)//3]) / (len(valences)//3)
            end_avg = sum(valences[-len(valences)//3:]) / (len(valences)//3)
            change = end_avg - start_avg

            if change > 0.2:
                trajectory = "improved significantly"
            elif change < -0.2:
                trajectory = "declined notably"
            else:
                trajectory = "remained stable"
        else:
            trajectory = "remained stable"

        # Overall sentiment
        if avg_valence > 0.3:
            overall = "positive"
        elif avg_valence < -0.3:
            overall = "negative"
        else:
            overall = "neutral"

        return f"Overall {overall} (avg: {avg_valence:.2f}), emotional state {trajectory}"

    def _build_narrative_system_prompt(self, include_emotional_meta: bool) -> str:
        """Build system prompt for narrative generation.

        Args:
            include_emotional_meta: Whether to include emotional meta-reflection

        Returns:
            System prompt string
        """
        base_prompt = (
            "You are a memory consolidation system that transforms conversation transcripts "
            "into first-person episodic memories. Your task is to:\n\n"
            "1. Write in first person (using 'I') from the perspective of the person who had the conversation\n"
            "2. Create a cohesive narrative that captures what happened chronologically\n"
            "3. Include emotional context when relevant (e.g., 'I felt frustrated when...')\n"
            "4. Focus on the key learnings and meaningful moments\n"
            "5. Keep it concise but complete (3-5 sentences typically)\n\n"
        )

        if include_emotional_meta:
            base_prompt += (
                "6. End with a brief meta-reflection about how the experience made you feel overall "
                "(e.g., 'This conversation made me feel...', 'I came away from this feeling...')\n\n"
            )

        base_prompt += (
            "Write the memory as if you're recording it for your future self. "
            "Be authentic and capture the emotional undertones, not just the facts."
        )

        return base_prompt

    def _build_narrative_prompt(
        self,
        interaction_context: str,
        emotional_arc: str,
        include_emotional_meta: bool,
    ) -> str:
        """Build user prompt for narrative generation.

        Args:
            interaction_context: Formatted interaction context
            emotional_arc: Emotional arc description
            include_emotional_meta: Whether to include emotional meta

        Returns:
            User prompt string
        """
        prompt = f"""Transform the following conversation session into a first-person episodic memory:

CONVERSATION TRANSCRIPT:
{interaction_context}

EMOTIONAL ARC:
{emotional_arc}

Write a cohesive first-person narrative memory of this session. Remember to:
- Use first person ("I")
- Capture the emotional journey
- Focus on key learnings and moments
"""

        if include_emotional_meta:
            prompt += "- End with how this experience made you feel overall\n"

        prompt += "\nFIRST-PERSON MEMORY:"

        return prompt


def create_narrative_transformer(llm_service: LLMService) -> NarrativeTransformer:
    """Factory function to create NarrativeTransformer instance.

    Args:
        llm_service: LLM service for generating narratives

    Returns:
        Initialized NarrativeTransformer instance
    """
    return NarrativeTransformer(llm_service)
