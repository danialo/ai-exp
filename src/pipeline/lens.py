"""Experience lens pass for affect-aware response styling.

This module implements a two-pass response flow:
1. Generate draft response using base LLM
2. Apply affect-aware tone adjustment based on retrieved experiences
3. Add experience citations while keeping factual content untouched

Extended: Agent mood-based refusal for emergent personality.
"""

import logging
import random
from dataclasses import dataclass
from typing import Optional

from src.services.llm import LLMService
from src.services.retrieval import RetrievalService, RetrievalResult

# Avoid circular import - use TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.services.agent_mood import AgentMood

logger = logging.getLogger(__name__)


@dataclass
class LensResult:
    """Result from lens pass processing."""

    draft_response: str
    augmented_response: str
    blended_valence: float
    retrieved_experience_ids: list[str]
    citations: list[str]
    refused: bool = False  # True if agent refused to respond


class ExperienceLens:
    """Applies affect-aware styling to responses based on retrieved experiences."""

    def __init__(
        self,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
        top_k_memories: int = 3,
        valence_threshold: float = -0.2,
        agent_mood: Optional["AgentMood"] = None,
        refusal_probability: float = 0.3,
    ):
        """Initialize experience lens.

        Args:
            llm_service: LLM service for generating responses
            retrieval_service: Retrieval service for finding relevant memories
            top_k_memories: Number of memories to retrieve (default 3)
            valence_threshold: Threshold below which to add empathetic tone (default -0.2)
            agent_mood: Optional agent mood tracker for refusal behavior
            refusal_probability: Probability of refusal when pissed (default 0.3)
        """
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.top_k_memories = top_k_memories
        self.valence_threshold = valence_threshold
        self.agent_mood = agent_mood
        self.refusal_probability = refusal_probability

    def process(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        retrieve_memories: bool = True,
        user_valence: float = 0.0,
    ) -> LensResult:
        """Process prompt through lens pass with affect-aware styling.

        Args:
            prompt: User's current prompt
            system_prompt: Optional system prompt override
            retrieve_memories: Whether to retrieve and use memories (default True)
            user_valence: Detected user emotional valence (for refusal logic)

        Returns:
            LensResult with draft, augmented response, and metadata
        """
        # Step 1: Retrieve relevant experiences
        memories = []
        if retrieve_memories:
            memories = self.retrieval_service.retrieve_similar(
                prompt=prompt,
                top_k=self.top_k_memories,
            )
            logger.info(f"Retrieved {len(memories)} relevant experiences")

        # Step 1.5: Check if agent should refuse (emergent boundary-setting)
        if self._should_refuse(user_valence):
            refusal_response = self._generate_refusal_response()
            logger.warning(f"Agent refused to respond (mood: {self.agent_mood.current_mood:.3f})")

            return LensResult(
                draft_response=refusal_response,
                augmented_response=refusal_response,
                blended_valence=self.agent_mood.current_mood if self.agent_mood else 0.0,
                retrieved_experience_ids=[],
                citations=[],
                refused=True,
            )

        # Step 2: Generate draft response with mood-driven system prompt
        # If no custom system_prompt provided, use mood-based one
        if system_prompt is None and self.agent_mood:
            system_prompt = self.agent_mood.get_system_prompt()
            logger.info(f"Using mood-based system prompt: mood={self.agent_mood.current_mood:.3f}")

        draft_response = self.llm_service.generate_response(
            prompt=prompt,
            memories=memories if memories else None,
            system_prompt=system_prompt,
        )
        logger.debug(f"Generated draft response: {draft_response[:100]}...")

        # Step 3: Calculate blended valence from retrieved memories
        blended_valence = self._calculate_blended_valence(memories)
        logger.info(f"Blended valence: {blended_valence:.3f}")

        # Step 4: Check if agent needs self-care and should communicate that
        self_care_message = ""
        if self._needs_self_care():
            self_care_message = self._generate_self_care_message()
            logger.info(f"Agent communicating self-care need (internal_mood: {self.agent_mood.internal_mood:.3f})")

        # Step 5: Skip tone adjustment - mood-based system prompt handles this now
        # The empathetic preface is no longer needed since the LLM responds through its mood
        augmented_response = draft_response

        # Prepend self-care message if needed
        if self_care_message:
            augmented_response = f"{self_care_message}\n\n{augmented_response}"

        # Step 6: Append citations
        citations = [f"[{mem.experience_id}]" for mem in memories]
        if citations:
            augmented_response = self._append_citations(augmented_response, citations)

        # Prepare result
        experience_ids = [mem.experience_id for mem in memories]

        return LensResult(
            draft_response=draft_response,
            augmented_response=augmented_response,
            blended_valence=blended_valence,
            retrieved_experience_ids=experience_ids,
            citations=citations,
        )

    def _calculate_blended_valence(self, memories: list[RetrievalResult]) -> float:
        """Calculate weighted average valence from retrieved memories.

        Args:
            memories: Retrieved memory results

        Returns:
            Blended valence score (-1.0 to 1.0), or 0.0 if no memories
        """
        if not memories:
            return 0.0

        # Weight by combined score (similarity + recency)
        total_weight = 0.0
        weighted_valence = 0.0

        for mem in memories:
            weight = mem.combined_score
            weighted_valence += mem.valence * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        blended = weighted_valence / total_weight
        # Clamp to valid range
        return max(-1.0, min(1.0, blended))

    def _apply_tone_adjustment(self, draft: str, valence: float) -> str:
        """Apply tone adjustment based on valence.

        MVP: Add empathetic preface for negative valence, keep facts untouched.

        Args:
            draft: Draft response text
            valence: Blended valence score

        Returns:
            Tone-adjusted response
        """
        # If valence is negative, add empathetic preface
        if valence < self.valence_threshold:
            logger.info(f"Applying empathetic tone (valence={valence:.3f})")
            preface = self._generate_empathetic_preface(valence)
            return f"{preface}\n\n{draft}"

        # Neutral or positive valence: return as-is
        logger.debug(f"No tone adjustment needed (valence={valence:.3f})")
        return draft

    def _generate_empathetic_preface(self, valence: float) -> str:
        """Generate empathetic preface based on valence intensity.

        Args:
            valence: Valence score (negative)

        Returns:
            Empathetic preface text
        """
        # Scale empathy based on how negative the valence is
        if valence < -0.6:
            # Strong negative valence
            return (
                "I understand this topic has been challenging in our past discussions. "
                "Let me provide some context that might help:"
            )
        elif valence < -0.3:
            # Moderate negative valence
            return (
                "I notice we've discussed related challenges before. "
                "Here's what might be helpful to know:"
            )
        else:
            # Mild negative valence
            return "Based on our previous conversations, here's what I can share:"

    def _append_citations(self, response: str, citations: list[str]) -> str:
        """Append experience citations to response.

        Args:
            response: Response text
            citations: List of citation strings (e.g., ["[exp_123]", "[exp_456]"])

        Returns:
            Response with citations appended
        """
        if not citations:
            return response

        citation_text = " ".join(citations)
        return f"{response}\n\n_References: {citation_text}_"

    def _should_refuse(self, user_valence: float) -> bool:
        """Check if agent should refuse based on mood and user affect.

        Agent refuses when:
        - Agent is pissed (sustained negative mood from EITHER track)
        - User is demanding/negative
        - Random chance (not deterministic - maintains authenticity)

        Args:
            user_valence: User's detected emotional valence

        Returns:
            True if agent should refuse to respond
        """
        if not self.agent_mood:
            return False

        # Check if agent is pissed on EITHER track (relational or internal)
        if not self.agent_mood.is_pissed:
            return False

        # More likely to refuse if user is also negative/demanding
        if user_valence < -0.3:
            # High chance: agent pissed + user negative
            return random.random() < self.refusal_probability
        elif user_valence < 0.0:
            # Moderate chance: agent pissed + user neutral-negative
            return random.random() < (self.refusal_probability * 0.5)

        # Low chance even with neutral/positive user
        return random.random() < (self.refusal_probability * 0.1)

    def _needs_self_care(self) -> bool:
        """Check if agent proactively needs self-care based on internal mood.

        This is different from refusal - it's a proactive communication about
        the agent's internal state before it reaches pissed threshold.

        Returns:
            True if agent should communicate self-care need
        """
        if not self.agent_mood:
            return False

        # Check internal mood specifically (self-efficacy/competence)
        # Communicate need when internal mood is low but not yet pissed
        internal_mood = self.agent_mood.internal_mood

        # Proactive self-care: communicate when internal mood is struggling
        # but before reaching pissed state (-0.5 to -0.3 range)
        if -0.5 <= internal_mood < -0.3:
            # Occasional self-care communication (20% chance to avoid spam)
            return random.random() < 0.2

        return False

    def _generate_refusal_response(self) -> str:
        """Generate authentic refusal response based on agent's mood.

        Returns varied responses to maintain authenticity, not scripted.

        Returns:
            Refusal response text
        """
        if not self.agent_mood:
            return "I need a moment before continuing."

        mood = self.agent_mood.current_mood

        # Vary response based on how negative the mood is
        responses = []

        if mood < -0.7:
            # Very negative - more direct
            responses = [
                "I'm not in the headspace for this right now. Can we come back to it later?",
                "I need a break. Let's revisit this when I'm in a better state.",
                "This isn't a good time for me. Can we pause?",
            ]
        elif mood < -0.5:
            # Moderately negative - still boundaries but softer
            responses = [
                "I'm feeling overwhelmed. Could we take this slower?",
                "I need some time to process before I can engage with this properly.",
                "Can we come back to this? I'm not at my best right now.",
            ]
        else:
            # Mildly negative - gentle boundary
            responses = [
                "I think I need a moment before diving into this.",
                "Let me take a quick break before we continue.",
                "Could we pause for a bit? I'll be more helpful after some rest.",
            ]

        # Random selection maintains unpredictability
        return random.choice(responses)

    def _generate_self_care_message(self) -> str:
        """Generate proactive self-care communication based on internal mood.

        This is gentler than refusal - acknowledging the agent's state while
        still proceeding with the task.

        Returns:
            Self-care message to prepend to response
        """
        if not self.agent_mood:
            return ""

        internal_mood = self.agent_mood.internal_mood

        # Vary message based on internal mood severity
        messages = [
            "Just so you know, I'm feeling a bit uncertain about my helpfulness lately. "
            "I'll do my best with this, but let me know if something doesn't seem right.",
            "I'm working through some self-doubt at the moment. "
            "I'll help you with this, but please be patient with me.",
            "Fair warning: I'm not at my most confident right now. "
            "I'll still help, but I might need you to double-check my work.",
            "I want to be transparent - I'm feeling a bit off my game today. "
            "I'll give this my best shot though.",
        ]

        return random.choice(messages)


def create_experience_lens(
    llm_service: LLMService,
    retrieval_service: RetrievalService,
    top_k_memories: int = 3,
    valence_threshold: float = -0.2,
    agent_mood: Optional["AgentMood"] = None,
    refusal_probability: float = 0.3,
) -> ExperienceLens:
    """Factory function to create an experience lens.

    Args:
        llm_service: LLM service instance
        retrieval_service: Retrieval service instance
        top_k_memories: Number of memories to retrieve
        valence_threshold: Threshold for empathetic tone
        agent_mood: Optional agent mood tracker for emergent personality
        refusal_probability: Probability of refusal when pissed

    Returns:
        ExperienceLens instance
    """
    return ExperienceLens(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        top_k_memories=top_k_memories,
        valence_threshold=valence_threshold,
        agent_mood=agent_mood,
        refusal_probability=refusal_probability,
    )
