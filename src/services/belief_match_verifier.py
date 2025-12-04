"""
Belief Match Verifier for HTN Self-Belief Decomposer.

LLM-based semantic match verification for uncertain resolution cases.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.utils.belief_config import BeliefSystemConfig, get_belief_config

logger = logging.getLogger(__name__)


def load_prompt(path: str) -> str:
    """Load a prompt template from file."""
    project_root = Path(__file__).resolve().parent.parent.parent
    full_path = project_root / path

    if not full_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {full_path}")

    return full_path.read_text().strip()


@dataclass
class VerifierResult:
    """
    Result of match verification.

    Attributes:
        same_concept: Whether the statements express the same concept
        confidence: Confidence in the determination
        reasoning: Brief explanation
    """
    same_concept: bool
    confidence: float
    reasoning: str


class BeliefMatchVerifier:
    """
    LLM-based semantic match verification.

    Called only when similarity is in the uncertain band
    (between no_match_threshold and match_threshold).
    """

    def __init__(
        self,
        config: Optional[BeliefSystemConfig] = None,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize the verifier.

        Args:
            config: Configuration object
            llm_client: LLM client for generation
        """
        if config is None:
            config = get_belief_config()

        self.config = config
        self.llm = llm_client
        self.prompt_template = load_prompt(config.prompts.verifier)

    def verify(self, atom_text: str, candidate_text: str) -> VerifierResult:
        """
        Verify whether two statements express the same belief concept.

        Args:
            atom_text: The new atom text
            candidate_text: The existing candidate text

        Returns:
            VerifierResult with determination
        """
        if not self.llm:
            # No LLM available - conservative default
            return VerifierResult(
                same_concept=False,
                confidence=0.3,
                reasoning="No LLM available for verification",
            )

        # Build prompt
        prompt = self.prompt_template.replace('{atom_text}', atom_text)
        prompt = prompt.replace('{candidate_text}', candidate_text)

        try:
            response = self._call_llm(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"Verifier call failed: {e}")
            return VerifierResult(
                same_concept=False,
                confidence=0.3,
                reasoning=f"Verification failed: {e}",
            )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the verifier prompt."""
        if hasattr(self.llm, 'generate_response'):
            return self.llm.generate_response(
                prompt=prompt,
                include_self_awareness=False,
            )
        elif hasattr(self.llm, 'generate'):
            return self.llm.generate(prompt=prompt)
        else:
            raise TypeError("LLM client must have generate_response or generate method")

    def _parse_response(self, response: str) -> VerifierResult:
        """
        Parse LLM response into VerifierResult.

        Args:
            response: LLM response text

        Returns:
            VerifierResult
        """
        # Try to extract JSON
        json_data = self._extract_json(response)

        if json_data is None:
            # Try to parse from text
            return self._parse_text_response(response)

        same_concept = json_data.get('same_concept', False)
        if isinstance(same_concept, str):
            same_concept = same_concept.lower() in ('true', 'yes', '1')

        confidence = json_data.get('confidence', 0.5)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        reasoning = json_data.get('reasoning', '')

        return VerifierResult(
            same_concept=same_concept,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _parse_text_response(self, response: str) -> VerifierResult:
        """
        Parse non-JSON response using heuristics.

        Args:
            response: Response text

        Returns:
            VerifierResult
        """
        response_lower = response.lower()

        # Look for clear indicators
        if 'same concept' in response_lower or 'same belief' in response_lower:
            if 'not ' in response_lower or 'different' in response_lower:
                return VerifierResult(
                    same_concept=False,
                    confidence=0.6,
                    reasoning=response[:200],
                )
            else:
                return VerifierResult(
                    same_concept=True,
                    confidence=0.6,
                    reasoning=response[:200],
                )

        if 'yes' in response_lower[:50]:
            return VerifierResult(
                same_concept=True,
                confidence=0.5,
                reasoning=response[:200],
            )

        if 'no' in response_lower[:50]:
            return VerifierResult(
                same_concept=False,
                confidence=0.5,
                reasoning=response[:200],
            )

        # Default: uncertain
        return VerifierResult(
            same_concept=False,
            confidence=0.3,
            reasoning="Could not parse response",
        )

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON object from text."""
        # Try the whole text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Try to find object braces
        brace_pattern = r'\{[\s\S]*\}'
        matches = re.findall(brace_pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None
