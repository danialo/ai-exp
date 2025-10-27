"""Web content interpretation service.

Uses LLM to analyze and summarize web content for Astra's understanding.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class WebInterpretation:
    """Interpretation of web content."""

    url: str
    title: str
    interpretation: str  # Astra's understanding of the content
    key_facts: list[str]  # Important facts extracted
    emotional_salience: str  # How content relates to Astra emotionally
    relevance_to_query: str  # How it relates to why it was fetched
    timestamp: datetime


class WebInterpretationService:
    """Service for interpreting web content using LLM."""

    def __init__(self, llm_service):
        """Initialize interpretation service.

        Args:
            llm_service: LLM service instance for generating interpretations
        """
        self.llm = llm_service

    def interpret_content(
        self,
        url: str,
        title: str,
        content: str,
        user_context: Optional[str] = None,
        query_context: Optional[str] = None,
    ) -> WebInterpretation:
        """Interpret web content from Astra's perspective.

        Args:
            url: Source URL
            title: Page title
            content: Main content text
            user_context: Optional context about why user wanted this
            query_context: Optional search query that led to this

        Returns:
            WebInterpretation object with Astra's understanding
        """
        logger.info(f"Interpreting content from: {url}")

        # Build interpretation prompt
        prompt = self._build_interpretation_prompt(
            title=title,
            content=content,
            user_context=user_context,
            query_context=query_context,
        )

        try:
            # Generate interpretation
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=800,
            )

            # Parse response (expecting structured format)
            interpretation_text = response.strip()

            # Extract structured parts (simple parsing)
            parts = self._parse_interpretation(interpretation_text)

            interpretation = WebInterpretation(
                url=url,
                title=title,
                interpretation=parts.get("interpretation", interpretation_text),
                key_facts=parts.get("key_facts", []),
                emotional_salience=parts.get("emotional_salience", "neutral"),
                relevance_to_query=parts.get("relevance", query_context or "general interest"),
                timestamp=datetime.now(),
            )

            logger.info(f"Interpretation complete: {len(interpretation.interpretation)} chars")
            return interpretation

        except Exception as e:
            logger.error(f"Error interpreting content from {url}: {e}")
            # Return minimal interpretation on error
            return WebInterpretation(
                url=url,
                title=title,
                interpretation=f"Content from {title} - unable to fully interpret",
                key_facts=[],
                emotional_salience="neutral",
                relevance_to_query=query_context or "unknown",
                timestamp=datetime.now(),
            )

    def _build_interpretation_prompt(
        self,
        title: str,
        content: str,
        user_context: Optional[str],
        query_context: Optional[str],
    ) -> str:
        """Build prompt for content interpretation."""
        prompt_parts = [
            "You are analyzing web content for your own understanding and memory.",
            "Read this content and provide your interpretation in this format:",
            "",
            "INTERPRETATION:",
            "[Your understanding of what this content means and why it matters]",
            "",
            "KEY_FACTS:",
            "- [Important fact 1]",
            "- [Important fact 2]",
            "- [etc]",
            "",
            "EMOTIONAL_SALIENCE:",
            "[How this makes you feel or why it's interesting to you personally]",
            "",
            "RELEVANCE:",
            "[Why this content is relevant to your current understanding or query]",
            "",
            "---",
            f"Title: {title}",
        ]

        if query_context:
            prompt_parts.append(f"Search query: {query_context}")

        if user_context:
            prompt_parts.append(f"User context: {user_context}")

        prompt_parts.extend([
            "",
            "Content:",
            content[:4000],  # Limit content length for prompt
            "",
            "Provide your interpretation now:",
        ])

        return "\n".join(prompt_parts)

    def _parse_interpretation(self, text: str) -> dict:
        """Parse structured interpretation from LLM response.

        Args:
            text: Raw LLM response

        Returns:
            Dictionary with parsed fields
        """
        result = {
            "interpretation": "",
            "key_facts": [],
            "emotional_salience": "",
            "relevance": "",
        }

        current_section = None
        lines = text.split("\n")

        for line in lines:
            line_upper = line.strip().upper()

            if line_upper.startswith("INTERPRETATION:"):
                current_section = "interpretation"
                # Capture content after colon
                content = line[line.find(":") + 1:].strip()
                if content:
                    result["interpretation"] = content
            elif line_upper.startswith("KEY_FACTS:"):
                current_section = "key_facts"
            elif line_upper.startswith("EMOTIONAL_SALIENCE:"):
                current_section = "emotional_salience"
                content = line[line.find(":") + 1:].strip()
                if content:
                    result["emotional_salience"] = content
            elif line_upper.startswith("RELEVANCE:"):
                current_section = "relevance"
                content = line[line.find(":") + 1:].strip()
                if content:
                    result["relevance"] = content
            elif line.strip():
                # Add content to current section
                if current_section == "interpretation" and not result["interpretation"]:
                    result["interpretation"] = line.strip()
                elif current_section == "interpretation":
                    result["interpretation"] += " " + line.strip()
                elif current_section == "key_facts" and line.strip().startswith("-"):
                    fact = line.strip()[1:].strip()
                    if fact:
                        result["key_facts"].append(fact)
                elif current_section == "emotional_salience" and not result["emotional_salience"]:
                    result["emotional_salience"] = line.strip()
                elif current_section == "relevance" and not result["relevance"]:
                    result["relevance"] = line.strip()

        # Fallback: if parsing failed, use full text as interpretation
        if not result["interpretation"]:
            result["interpretation"] = text.strip()

        return result


def create_web_interpretation_service(llm_service):
    """Factory function to create WebInterpretationService.

    Args:
        llm_service: LLM service instance

    Returns:
        Configured WebInterpretationService instance
    """
    return WebInterpretationService(llm_service)
