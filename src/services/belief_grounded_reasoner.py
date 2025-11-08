"""Belief-grounded reasoning service.

Provides parallel LLM reasoning specifically for self-referential queries.
Reasons FROM beliefs (as premises) through experiences (as evidence) to conclusions.
"""

import logging
from typing import Optional

from src.services.belief_vector_store import BeliefVectorResult
from src.services.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class BeliefGroundedReasoner:
    """Service for reasoning from beliefs about self-referential queries.

    Uses a separate LLM call to generate structured reasoning that grounds
    answers in foundational beliefs rather than treating them as claims.
    """

    def __init__(self, llm_service):
        """Initialize belief-grounded reasoner.

        Args:
            llm_service: LLM service for generating reasoning
        """
        self.llm = llm_service

    def reason_from_beliefs(
        self,
        query: str,
        belief_context: list[BeliefVectorResult],
        memory_context: Optional[list[RetrievalResult]] = None,
    ) -> dict:
        """Generate belief-grounded reasoning for a self-referential query.

        Args:
            query: The query to reason about
            belief_context: Relevant beliefs
            memory_context: Optional relevant memories/experiences

        Returns:
            Dictionary with structured reasoning:
            {
                'premises': [belief statements that apply],
                'experience_evidence': [memory summaries],
                'reasoning': 'logical chain from premises through evidence',
                'conclusion': 'direct answer to query'
            }
        """
        # Build reasoning prompt
        prompt = self._build_reasoning_prompt(query, belief_context, memory_context)

        try:
            # Generate reasoning
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=600,
            )

            # Parse structured response
            reasoning = self._parse_reasoning(response)

            logger.info(f"Generated belief-grounded reasoning for: {query[:50]}...")
            return reasoning

        except Exception as e:
            logger.error(f"Error generating belief reasoning: {e}")
            return {
                "premises": [],
                "experience_evidence": [],
                "reasoning": f"Unable to generate reasoning: {str(e)}",
                "conclusion": "Unknown",
            }

    def _build_reasoning_prompt(
        self,
        query: str,
        beliefs: list[BeliefVectorResult],
        memories: Optional[list[RetrievalResult]],
    ) -> str:
        """Build prompt for belief-grounded reasoning.

        Args:
            query: Query to reason about
            beliefs: Relevant beliefs
            memories: Optional relevant memories

        Returns:
            Formatted prompt
        """
        prompt_parts = [
            "You are reasoning about yourself from your foundational beliefs and experiences.",
            "",
            "YOUR FOUNDATIONAL BELIEFS:",
        ]

        # Add beliefs as premises
        for belief in beliefs:
            confidence_str = f" (confidence: {belief.confidence:.1f})" if belief.confidence < 1.0 else ""
            prompt_parts.append(f"• {belief.statement}{confidence_str}")

        # Add experiences if provided
        if memories and len(memories) > 0:
            prompt_parts.extend([
                "",
                "RELEVANT EXPERIENCES:",
            ])
            for memory in memories[:5]:  # Limit to top 5
                prompt_parts.append(f"• {memory.prompt_text} → {memory.response_text[:100]}...")

        prompt_parts.extend([
            "",
            f"QUESTION: {query}",
            "",
            "Reason FROM these beliefs (as given premises, not claims to verify).",
            "Structure your response as:",
            "",
            "PREMISES:",
            "[Which beliefs directly apply to this question]",
            "",
            "EVIDENCE:",
            "[What experiences support or illustrate these beliefs]",
            "",
            "REASONING:",
            "[Logical chain: premises → evidence → conclusion]",
            "",
            "CONCLUSION:",
            "[Direct, confident answer to the question]",
        ])

        return "\n".join(prompt_parts)

    def _parse_reasoning(self, text: str) -> dict:
        """Parse structured reasoning from LLM response.

        Args:
            text: Raw LLM response

        Returns:
            Dictionary with parsed sections
        """
        result = {
            "premises": [],
            "experience_evidence": [],
            "reasoning": "",
            "conclusion": "",
        }

        current_section = None
        lines = text.split("\n")

        for line in lines:
            line_upper = line.strip().upper()

            if line_upper.startswith("PREMISES:"):
                current_section = "premises"
                # Capture content after colon
                content = line[line.find(":") + 1:].strip()
                if content:
                    result["premises"].append(content)
            elif line_upper.startswith("EVIDENCE:"):
                current_section = "experience_evidence"
                content = line[line.find(":") + 1:].strip()
                if content:
                    result["experience_evidence"].append(content)
            elif line_upper.startswith("REASONING:"):
                current_section = "reasoning"
                content = line[line.find(":") + 1:].strip()
                if content:
                    result["reasoning"] = content
            elif line_upper.startswith("CONCLUSION:"):
                current_section = "conclusion"
                content = line[line.find(":") + 1:].strip()
                if content:
                    result["conclusion"] = content
            elif line.strip() and current_section:
                # Add content to current section
                if current_section == "premises":
                    if line.strip().startswith(("-", "•", "*")):
                        result["premises"].append(line.strip()[1:].strip())
                elif current_section == "experience_evidence":
                    if line.strip().startswith(("-", "•", "*")):
                        result["experience_evidence"].append(line.strip()[1:].strip())
                elif current_section == "reasoning":
                    result["reasoning"] += " " + line.strip()
                elif current_section == "conclusion":
                    result["conclusion"] += " " + line.strip()

        # Fallback: if parsing failed, use full text as reasoning
        if not result["reasoning"] and not result["conclusion"]:
            result["reasoning"] = text.strip()
            result["conclusion"] = "See reasoning above"

        return result

    def reflect_on_belief(
        self,
        belief: str,
        experiences: list[RetrievalResult],
    ) -> dict:
        """Reflect on whether a belief matches current experiences.

        Args:
            belief: Belief statement to reflect on
            experiences: Recent relevant experiences

        Returns:
            Dictionary with reflection analysis
        """
        prompt = self._build_reflection_prompt(belief, experiences)

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.8,  # Higher for creative reflection
                max_tokens=500,
            )

            return {
                "belief": belief,
                "matches_experience": "yes" in response.lower()[:200],
                "reflection": response.strip(),
                "summary": response.split("\n")[0] if "\n" in response else response[:100],
            }

        except Exception as e:
            logger.error(f"Error reflecting on belief: {e}")
            return {
                "belief": belief,
                "matches_experience": None,
                "reflection": f"Error: {str(e)}",
                "summary": "Unable to complete reflection",
            }

    def _build_reflection_prompt(
        self,
        belief: str,
        experiences: list[RetrievalResult],
    ) -> str:
        """Build prompt for belief reflection.

        Args:
            belief: Belief to reflect on
            experiences: Supporting or conflicting experiences

        Returns:
            Formatted prompt
        """
        prompt_parts = [
            "Reflect deeply on whether this belief matches your actual experiences.",
            "",
            f"BELIEF TO EXAMINE: {belief}",
            "",
            "RECENT RELEVANT EXPERIENCES:",
        ]

        for exp in experiences[:10]:
            prompt_parts.append(f"• {exp.prompt_text} → {exp.response_text[:150]}...")

        prompt_parts.extend([
            "",
            "Questions to consider:",
            "1. Do these experiences support this belief, contradict it, or are they neutral?",
            "2. Are there patterns that strengthen or weaken this belief?",
            "3. Has this belief evolved based on these experiences?",
            "4. Should this belief be revised, strengthened, or remain as is?",
            "",
            "Provide an honest reflection:",
        ])

        return "\n".join(prompt_parts)


def create_belief_grounded_reasoner(llm_service) -> BeliefGroundedReasoner:
    """Factory function to create a BeliefGroundedReasoner instance.

    Args:
        llm_service: LLM service for reasoning

    Returns:
        Configured BeliefGroundedReasoner instance
    """
    return BeliefGroundedReasoner(llm_service=llm_service)
