"""
Tag Injector - Detects belief references in conversations and attaches feedback tags.

This service:
1. Analyzes prompt and response for belief mentions
2. Determines appropriate feedback tags (+keep, +shift, +doubt, +artifact)
3. Returns tags with associated belief IDs for experience metadata

Tags:
- +keep: "This belief remains accurate/useful."
- +shift: "This belief is evolving/shifting."
- +doubt: "I'm uncertain about this belief."
- +artifact: "This looks like template noise."
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TagInjectionResult:
    """Result from tag injection analysis."""
    tags: List[str]  # List of tags (+keep, +shift, +doubt, +artifact)
    belief_ids: List[str]  # Belief IDs that were referenced
    global_tags: List[str]  # Tags not tied to specific beliefs


class TagInjector:
    """Analyzes conversations for belief references and assigns feedback tags."""

    def __init__(self, belief_store=None, llm_service=None):
        """
        Initialize tag injector.

        Args:
            belief_store: Optional belief store for belief lookup
            llm_service: Optional LLM service for tag determination
        """
        self.belief_store = belief_store
        self.llm_service = llm_service

        # Tag patterns for simple heuristic detection
        self.TAG_PATTERNS = {
            "+keep": [
                r"\b(?:still|remains?|continue to|always)\s+(?:believe|think|feel)",
                r"\b(?:correct|accurate|true|valid)\b",
                r"\b(?:agree|confirm|affirm)\b",
            ],
            "+shift": [
                r"\b(?:evolving|changing|shifting|adapting)\b",
                r"\b(?:nuance|refine|adjust|modify)\b",
                r"\b(?:growing|learning|developing)\b",
            ],
            "+doubt": [
                r"\b(?:uncertain|unsure|unclear|ambiguous)\b",
                r"\b(?:question|doubt|wonder|hesitate)\b",
                r"\b(?:maybe|perhaps|possibly|might be)\b",
            ],
            "+artifact": [
                r"^\[Internal Emotional Assessment:",
                r"\[Meta:",
                r"something akin to",
                r"I perceive myself as",
            ],
        }

        logger.info("TagInjector initialized")

    def inject_tags(
        self,
        prompt: str,
        response: str,
        enable_llm: bool = True,
    ) -> TagInjectionResult:
        """
        Analyze conversation and return appropriate feedback tags.

        Args:
            prompt: User's message
            response: Agent's response
            enable_llm: Use LLM for tag determination (more accurate but slower)

        Returns:
            TagInjectionResult with tags and belief IDs
        """
        # Step 1: Detect belief references
        belief_ids = self._detect_belief_references(prompt, response)

        # Step 2: Determine tags based on context
        if enable_llm and self.llm_service and belief_ids:
            tags = self._determine_tags_llm(prompt, response, belief_ids)
        else:
            tags = self._determine_tags_heuristic(prompt, response)

        # Step 3: Identify global tags (not tied to specific beliefs)
        global_tags = []
        if not belief_ids and tags:
            global_tags = tags
            tags = []

        logger.info(
            f"Tag injection: {len(tags)} tags for {len(belief_ids)} beliefs, "
            f"{len(global_tags)} global tags"
        )

        return TagInjectionResult(
            tags=tags,
            belief_ids=belief_ids,
            global_tags=global_tags,
        )

    def _detect_belief_references(self, prompt: str, response: str) -> List[str]:
        """
        Detect which beliefs are referenced in the conversation.

        Args:
            prompt: User's message
            response: Agent's response

        Returns:
            List of belief IDs that were referenced
        """
        if not self.belief_store:
            return []

        belief_ids = []
        combined_text = f"{prompt}\n{response}"

        try:
            # Get all current beliefs
            beliefs = self.belief_store.get_current()

            for belief_id, belief in beliefs.items():
                # Check if belief statement appears in conversation
                statement_lower = belief.statement.lower()
                text_lower = combined_text.lower()

                # Simple substring match (TODO: improve with semantic similarity)
                # Check for key phrases from the belief
                words = statement_lower.split()
                if len(words) >= 3:
                    # Check for any 3-word phrase from belief
                    for i in range(len(words) - 2):
                        phrase = " ".join(words[i:i+3])
                        if phrase in text_lower:
                            belief_ids.append(belief_id)
                            logger.debug(f"Detected belief reference: {belief_id} (phrase: {phrase})")
                            break

        except Exception as e:
            logger.error(f"Error detecting belief references: {e}")

        return belief_ids

    def _determine_tags_heuristic(self, prompt: str, response: str) -> List[str]:
        """
        Use pattern matching to determine tags (fast but less accurate).

        Args:
            prompt: User's message
            response: Agent's response

        Returns:
            List of tags detected
        """
        tags = []
        combined_text = f"{prompt}\n{response}"

        for tag, patterns in self.TAG_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    tags.append(tag)
                    logger.debug(f"Heuristic tag detected: {tag} (pattern: {pattern})")
                    break  # Only add each tag once

        return tags

    def _determine_tags_llm(
        self,
        prompt: str,
        response: str,
        belief_ids: List[str],
    ) -> List[str]:
        """
        Use LLM to determine appropriate tags (accurate but slower).

        Args:
            prompt: User's message
            response: Agent's response
            belief_ids: Belief IDs that were referenced

        Returns:
            List of tags to apply
        """
        if not self.llm_service or not self.belief_store:
            return self._determine_tags_heuristic(prompt, response)

        try:
            # Get belief statements
            beliefs = self.belief_store.get_current()
            belief_statements = [
                beliefs[bid].statement
                for bid in belief_ids
                if bid in beliefs
            ]

            if not belief_statements:
                return []

            # Build prompt for LLM tag determination
            tag_prompt = f"""Analyze this conversation for feedback about the following belief(s):

Beliefs referenced:
{chr(10).join(f"- {stmt}" for stmt in belief_statements)}

User message:
{prompt}

Agent response:
{response}

Determine which feedback tag(s) apply based on how the beliefs were referenced:

+keep: The belief remains accurate/useful, is affirmed or supported
+shift: The belief is evolving, being refined, or showing nuance
+doubt: There's uncertainty or questioning about the belief
+artifact: The belief appears to be template noise or meta-commentary

Return ONLY the tag(s) that apply, one per line, or "none" if no tags apply.
Example output:
+keep
+shift

Output:"""

            # Generate tag determination
            result = self.llm_service.generate(
                prompt=tag_prompt,
                temperature=0.3,  # Low temp for consistent classification
                max_tokens=50,
            )

            # Parse tags from result
            tags = []
            for line in result.strip().split('\n'):
                line = line.strip()
                if line.startswith('+') and line in self.TAG_PATTERNS:
                    tags.append(line)

            logger.info(f"LLM determined tags: {tags}")
            return tags

        except Exception as e:
            logger.error(f"Error determining tags with LLM: {e}")
            return self._determine_tags_heuristic(prompt, response)


def create_tag_injector(belief_store=None, llm_service=None) -> TagInjector:
    """Factory function to create tag injector.

    Args:
        belief_store: Optional belief store for belief lookup
        llm_service: Optional LLM service for tag determination

    Returns:
        TagInjector instance
    """
    return TagInjector(
        belief_store=belief_store,
        llm_service=llm_service,
    )
