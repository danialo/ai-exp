"""
Belief Atomizer for HTN Self-Belief Decomposer.

LLM-based atomic belief extraction from claim candidates.
Uses prompt templates and handles JSON repair.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.belief_config import BeliefSystemConfig, get_belief_config
from src.services.belief_segmenter import ClaimCandidate

logger = logging.getLogger(__name__)


@dataclass
class RawAtom:
    """
    Raw atom extracted by the LLM.

    Attributes:
        atom_text: The atomic belief text
        belief_type: Ontological category
        polarity: affirm or deny
        confidence: LLM confidence in extraction
        spans: Character spans in source (can be list of tuples after dedup)
        source_candidate_idx: Index of source candidate
    """
    atom_text: str
    belief_type: str
    polarity: str
    confidence: float
    spans: Optional[List[Tuple[int, int]]] = None
    source_candidate_idx: int = 0


@dataclass
class AtomizerResult:
    """
    Result of atomization.

    Attributes:
        atoms: Successfully extracted atoms
        errors: Errors encountered during extraction
    """
    atoms: List[RawAtom]
    errors: List[Dict[str, Any]] = field(default_factory=list)


def load_prompt(path: str) -> str:
    """Load a prompt template from file."""
    project_root = Path(__file__).resolve().parent.parent.parent
    full_path = project_root / path

    if not full_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {full_path}")

    return full_path.read_text().strip()


class BeliefAtomizer:
    """
    LLM-based atomic belief extraction.

    Extracts atomic, first-person belief statements from claim candidates.
    Handles JSON repair if the LLM returns malformed output.
    """

    # Valid belief types
    VALID_BELIEF_TYPES = {
        'TRAIT', 'PREFERENCE', 'VALUE', 'CAPABILITY_LIMIT',
        'FEELING_STATE', 'META_BELIEF', 'RELATIONAL', 'BELIEF_ABOUT_SELF'
    }

    # Valid polarities
    VALID_POLARITIES = {'affirm', 'deny'}

    def __init__(
        self,
        config: Optional[BeliefSystemConfig] = None,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize the atomizer.

        Args:
            config: Configuration object. If None, loads from default.
            llm_client: LLM client for generation. If None, must be provided later.
        """
        if config is None:
            config = get_belief_config()

        self.config = config
        self.llm = llm_client

        # Load prompts
        self.system_prompt = load_prompt(config.prompts.atomizer_system)
        self.user_prompt_template = load_prompt(config.prompts.atomizer_user)
        self.repair_prompt_template = load_prompt(config.prompts.repair_json)

    def atomize(self, candidates: List[ClaimCandidate]) -> AtomizerResult:
        """
        Extract atomic beliefs from claim candidates.

        Args:
            candidates: List of claim candidates to process

        Returns:
            AtomizerResult with atoms and any errors
        """
        all_atoms: List[RawAtom] = []
        errors: List[Dict[str, Any]] = []

        for idx, candidate in enumerate(candidates):
            try:
                atoms = self._extract_atoms(candidate, idx)
                all_atoms.extend(atoms)
            except Exception as e:
                logger.warning(f"Failed to extract atoms from candidate {idx}: {e}")
                errors.append({
                    'candidate_idx': idx,
                    'error_type': type(e).__name__,
                    'details': str(e),
                })

        return AtomizerResult(atoms=all_atoms, errors=errors)

    def _extract_atoms(
        self,
        candidate: ClaimCandidate,
        candidate_idx: int
    ) -> List[RawAtom]:
        """
        Extract atoms from a single candidate.

        Args:
            candidate: Claim candidate to process
            candidate_idx: Index in source list

        Returns:
            List of RawAtom objects
        """
        if not self.llm:
            # If no LLM client, do simple extraction
            return self._simple_extract(candidate, candidate_idx)

        # Build prompt
        user_prompt = self.user_prompt_template.replace('{input_text}', candidate.text)

        # Call LLM
        try:
            response = self._call_llm(user_prompt)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return self._simple_extract(candidate, candidate_idx)

        # Parse response
        atoms = self._parse_response(response, candidate, candidate_idx)
        return atoms

    def _call_llm(self, user_prompt: str) -> str:
        """
        Call the LLM with the atomizer prompts.

        Args:
            user_prompt: User prompt with input text

        Returns:
            LLM response text
        """
        if hasattr(self.llm, 'generate_response'):
            # Standard LLMService interface
            return self.llm.generate_response(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                include_self_awareness=False,
            )
        elif hasattr(self.llm, 'generate'):
            # Alternative interface
            return self.llm.generate(
                prompt=user_prompt,
                system=self.system_prompt,
            )
        else:
            raise TypeError("LLM client must have generate_response or generate method")

    def _parse_response(
        self,
        response: str,
        candidate: ClaimCandidate,
        candidate_idx: int
    ) -> List[RawAtom]:
        """
        Parse LLM response into atoms.

        Args:
            response: LLM response text
            candidate: Source candidate
            candidate_idx: Index in source list

        Returns:
            List of validated RawAtom objects
        """
        # Try to parse as JSON
        json_data = self._extract_json(response)

        if json_data is None:
            # Try JSON repair
            json_data = self._repair_json(response)

        if json_data is None:
            logger.warning(f"Could not parse LLM response as JSON")
            return self._simple_extract(candidate, candidate_idx)

        # Validate and convert to atoms
        atoms = []
        for item in json_data:
            atom = self._validate_atom_data(item, candidate, candidate_idx)
            if atom:
                atoms.append(atom)

        return atoms

    def _extract_json(self, text: str) -> Optional[List[Dict]]:
        """
        Extract JSON array from text.

        Handles markdown code blocks and bare JSON.
        """
        # Try to find JSON array in the text
        # First try the whole text
        try:
            data = json.loads(text.strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue

        # Try to find array brackets
        bracket_pattern = r'\[[\s\S]*\]'
        matches = re.findall(bracket_pattern, text)
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue

        return None

    def _repair_json(self, broken_json: str) -> Optional[List[Dict]]:
        """
        Attempt to repair malformed JSON using LLM.

        Args:
            broken_json: The malformed JSON string

        Returns:
            Parsed JSON list or None if repair failed
        """
        if not self.llm:
            return None

        max_attempts = self.config.extractor.max_json_repair_attempts
        if max_attempts <= 0:
            return None

        repair_prompt = self.repair_prompt_template.replace('{broken_json}', broken_json)

        try:
            response = self._call_llm(repair_prompt)
            return self._extract_json(response)
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}")
            return None

    def _validate_atom_data(
        self,
        data: Dict,
        candidate: ClaimCandidate,
        candidate_idx: int
    ) -> Optional[RawAtom]:
        """
        Validate atom data from LLM and convert to RawAtom.

        Args:
            data: Dictionary from LLM response
            candidate: Source candidate
            candidate_idx: Index in source list

        Returns:
            RawAtom if valid, None otherwise
        """
        # Required fields
        atom_text = data.get('atom_text')
        if not atom_text or not isinstance(atom_text, str):
            logger.debug("Missing or invalid atom_text")
            return None

        # Belief type (with validation)
        belief_type = data.get('belief_type', 'UNKNOWN')
        if belief_type not in self.VALID_BELIEF_TYPES:
            belief_type = 'UNKNOWN'

        # Polarity (with validation)
        polarity = data.get('polarity', 'affirm')
        if polarity not in self.VALID_POLARITIES:
            polarity = 'affirm'

        # Confidence
        confidence = data.get('confidence', 0.8)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.8

        # Spans
        spans = data.get('spans')
        if spans:
            if isinstance(spans, list) and len(spans) >= 2:
                # [start, end] format
                spans = [(spans[0], spans[1])]
            else:
                spans = None

        return RawAtom(
            atom_text=atom_text.strip(),
            belief_type=belief_type,
            polarity=polarity,
            confidence=confidence,
            spans=spans,
            source_candidate_idx=candidate_idx,
        )

    def _simple_extract(
        self,
        candidate: ClaimCandidate,
        candidate_idx: int
    ) -> List[RawAtom]:
        """
        Simple extraction fallback when LLM is unavailable.

        Creates a single atom from the candidate text.
        """
        text = candidate.text.strip()

        # Remove trailing punctuation for the atom
        text = text.rstrip('.,;:!?')

        # Determine polarity from negation words
        text_lower = text.lower()
        polarity = 'deny' if any(neg in text_lower for neg in
            ["don't", "do not", "not", "never", "cannot", "can't"]) else 'affirm'

        return [RawAtom(
            atom_text=text,
            belief_type='UNKNOWN',
            polarity=polarity,
            confidence=0.5,  # Lower confidence for simple extraction
            spans=[candidate.span],
            source_candidate_idx=candidate_idx,
        )]


def atomize_candidates(
    candidates: List[ClaimCandidate],
    llm_client: Optional[Any] = None
) -> AtomizerResult:
    """
    Convenience function to atomize claim candidates.

    Args:
        candidates: Claim candidates to process
        llm_client: Optional LLM client

    Returns:
        AtomizerResult with atoms and errors
    """
    atomizer = BeliefAtomizer(llm_client=llm_client)
    return atomizer.atomize(candidates)
