"""
Epistemics LLM Fallback for HTN Self-Belief Decomposer.

LLM-based epistemic frame extraction for low-confidence cases.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.belief_config import BeliefSystemConfig, get_belief_config
from src.services.epistemics_rules import EpistemicFrame, EpistemicsResult

logger = logging.getLogger(__name__)


def load_prompt(path: str) -> str:
    """Load a prompt template from file."""
    project_root = Path(__file__).resolve().parent.parent.parent
    full_path = project_root / path

    if not full_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {full_path}")

    return full_path.read_text().strip()


# Valid temporal scope values
VALID_TEMPORAL_SCOPES = {'state', 'ongoing', 'habitual', 'transitional', 'past', 'unknown'}

# Mapping from invalid to valid values
TEMPORAL_SCOPE_ALIASES = {
    'current': 'state',  # "current" is NOT valid, map to "state"
    'present': 'state',
    'now': 'state',
    'always': 'habitual',
    'permanent': 'habitual',
    'temporary': 'transitional',
    'changing': 'transitional',
    'former': 'past',
    'historical': 'past',
}

# Valid modality values
VALID_MODALITIES = {'certain', 'likely', 'possible', 'unsure'}


class EpistemicsLLMFallback:
    """
    LLM-based epistemic frame extraction for low-confidence cases.

    Only called when rules-based extraction confidence is below threshold.
    """

    def __init__(
        self,
        config: Optional[BeliefSystemConfig] = None,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize the LLM fallback.

        Args:
            config: Configuration object. If None, loads from default.
            llm_client: LLM client for generation.
        """
        if config is None:
            config = get_belief_config()

        self.config = config
        self.llm = llm_client
        self.prompt_template = load_prompt(config.prompts.epistemics_fallback)

    def extract(
        self,
        atom_text: str,
        rules_result: Optional[EpistemicsResult] = None
    ) -> EpistemicsResult:
        """
        Extract epistemic frame using LLM.

        Args:
            atom_text: Belief atom text
            rules_result: Optional previous rules-based result

        Returns:
            Updated EpistemicsResult with LLM extraction
        """
        if not self.llm:
            # No LLM available, return rules result or defaults
            if rules_result:
                return rules_result
            return EpistemicsResult(
                frame=EpistemicFrame(),
                confidence=0.3,
                signals=[{'source': 'default', 'reason': 'no_llm_available'}],
            )

        # Build prompt
        prompt = self.prompt_template.replace('{atom_text}', atom_text)

        try:
            response = self._call_llm(prompt)
            frame, confidence = self._parse_response(response)

            # Merge with rules result signals
            signals = rules_result.signals.copy() if rules_result else []
            signals.append({
                'source': 'llm_fallback',
                'llm_override': True,
            })

            return EpistemicsResult(
                frame=frame,
                confidence=confidence,
                signals=signals,
                needs_llm_fallback=False,  # We just did the fallback
                detected_polarity=rules_result.detected_polarity if rules_result else None,
            )

        except Exception as e:
            logger.warning(f"LLM epistemics fallback failed: {e}")
            # Return rules result if available
            if rules_result:
                rules_result.signals.append({
                    'source': 'llm_fallback_error',
                    'error': str(e),
                })
                return rules_result

            return EpistemicsResult(
                frame=EpistemicFrame(),
                confidence=0.3,
                signals=[{'source': 'fallback_error', 'error': str(e)}],
            )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the epistemics prompt."""
        if hasattr(self.llm, 'generate_response'):
            return self.llm.generate_response(
                prompt=prompt,
                include_self_awareness=False,
            )
        elif hasattr(self.llm, 'generate'):
            return self.llm.generate(prompt=prompt)
        else:
            raise TypeError("LLM client must have generate_response or generate method")

    def _parse_response(self, response: str) -> tuple[EpistemicFrame, float]:
        """
        Parse LLM response into epistemic frame.

        Returns (frame, confidence).
        """
        # Try to extract JSON
        json_data = self._extract_json(response)

        if json_data is None:
            logger.warning("Could not parse LLM epistemics response")
            return EpistemicFrame(), 0.3

        # Extract and validate fields
        temporal_scope = json_data.get('temporal_scope', 'ongoing')
        temporal_scope = self._normalize_temporal_scope(temporal_scope)

        modality = json_data.get('modality', 'certain')
        if modality not in VALID_MODALITIES:
            modality = 'certain'

        degree = json_data.get('degree', 0.5)
        try:
            degree = float(degree)
            degree = max(0.0, min(1.0, degree))
        except (TypeError, ValueError):
            degree = 0.5

        confidence = json_data.get('confidence', 0.7)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.7

        frame = EpistemicFrame(
            temporal_scope=temporal_scope,
            modality=modality,
            degree=degree,
        )

        return frame, confidence

    def _normalize_temporal_scope(self, value: str) -> str:
        """
        Normalize temporal scope value.

        Maps invalid values like "current" to valid ones like "state".
        """
        value_lower = value.lower().strip()

        if value_lower in VALID_TEMPORAL_SCOPES:
            return value_lower

        if value_lower in TEMPORAL_SCOPE_ALIASES:
            return TEMPORAL_SCOPE_ALIASES[value_lower]

        logger.debug(f"Unknown temporal_scope '{value}', defaulting to 'ongoing'")
        return 'ongoing'

    def _extract_json(self, text: str) -> Optional[Dict]:
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
