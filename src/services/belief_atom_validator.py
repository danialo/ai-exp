"""
Belief Atom Validator for HTN Self-Belief Decomposer.

Post-LLM validation filter to reject invalid atoms:
- Not first-person
- Imperative/instructional
- Too short
- Template junk
- Not actually self-claims
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.services.belief_atomizer import RawAtom


@dataclass
class ValidationResult:
    """
    Result of atom validation.

    Attributes:
        valid: Atoms that passed validation
        invalid: Rejected atoms with reasons
    """
    valid: List[RawAtom]
    invalid: List[Dict[str, Any]] = field(default_factory=list)


class BeliefAtomValidator:
    """
    Post-LLM validation filter for belief atoms.

    Rejects atoms that are:
    1. Not first-person (must start with "I")
    2. Imperative/instructional
    3. Too short (<3 words after "I")
    4. Template junk (placeholders, brackets)
    5. Questions
    6. Generic statements about "people" or "one"
    """

    # Imperative patterns - these suggest instructions, not self-beliefs
    IMPERATIVE_PATTERNS = [
        r'^(do|don\'t|always|never|try to|make sure|remember to)\s',
        r'^(engage in|practice|maintain|develop|cultivate)\s',
        r'^(be|become|keep|stay|remain)\s+\w+\s*(when|if|while)',
        r'^(should|must|need to|have to|ought to)\s',
    ]

    # Template/junk patterns
    TEMPLATE_PATTERNS = [
        r'\{[^}]+\}',  # {placeholder}
        r'\[[^\]]+\]',  # [brackets]
        r'<[^>]+>',    # <tags>
        r'_+',         # underscores (fill-in-blank)
        r'\.\.\.',     # ellipsis
    ]

    # Generic patterns (not about self)
    GENERIC_PATTERNS = [
        r'^(people|one|someone|anyone|everyone)\s',
        r'^(we|you)\s+(should|must|can|might)',
        r'^it\s+is\s+(important|good|bad|necessary)',
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self.imperative_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.IMPERATIVE_PATTERNS
        ]
        self.template_patterns = [
            re.compile(p) for p in self.TEMPLATE_PATTERNS
        ]
        self.generic_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.GENERIC_PATTERNS
        ]

    def validate(self, atoms: List[RawAtom]) -> ValidationResult:
        """
        Validate a list of atoms.

        Args:
            atoms: Raw atoms to validate

        Returns:
            ValidationResult with valid and invalid atoms
        """
        valid = []
        invalid = []

        for atom in atoms:
            is_valid, reason = self._validate_atom(atom)
            if is_valid:
                valid.append(atom)
            else:
                invalid.append({
                    'atom_text': atom.atom_text,
                    'reason': reason,
                })

        return ValidationResult(valid=valid, invalid=invalid)

    def _validate_atom(self, atom: RawAtom) -> tuple[bool, Optional[str]]:
        """
        Validate a single atom.

        Returns (is_valid, reason) tuple.
        """
        text = atom.atom_text.strip()
        text_lower = text.lower()

        # REMOVED: first-person check - trust the atomizer LLM to extract
        # first-person beliefs. The atomizer is already instructed to extract
        # self-referential statements. Double-filtering here was too restrictive,
        # rejecting valid beliefs like "My favorite food is sushi".

        # Check for imperative
        if self._is_imperative(text_lower):
            return False, "imperative"

        # Check minimum length
        if self._is_too_short(text):
            return False, "too_short"

        # Check for template junk
        if self._has_template_junk(text):
            return False, "template_junk"

        # Check for questions
        if self._is_question(text):
            return False, "question"

        # Check for generic statements
        if self._is_generic(text_lower):
            return False, "generic"

        return True, None

    def _is_first_person(self, text_lower: str) -> bool:
        """Check if text starts with first-person pronoun."""
        # Must start with "I" (case insensitive)
        first_person_starts = [
            'i ', "i'm ", "i've ", "i'll ", "i'd ", "i'm ", "i've ", "i'll ", "i'd "
        ]
        return any(text_lower.startswith(start) for start in first_person_starts)

    def _is_imperative(self, text_lower: str) -> bool:
        """Check if text is an imperative/instruction."""
        for pattern in self.imperative_patterns:
            if pattern.search(text_lower):
                return True
        return False

    def _is_too_short(self, text: str) -> bool:
        """Check if text is too short (< 3 words after I)."""
        words = text.split()
        if len(words) < 3:
            return True

        # Also check: if starts with "I", need at least verb + something
        if words[0].lower() == 'i' and len(words) < 3:
            return True

        return False

    def _has_template_junk(self, text: str) -> bool:
        """Check for template placeholders and junk."""
        for pattern in self.template_patterns:
            if pattern.search(text):
                return True
        return False

    def _is_question(self, text: str) -> bool:
        """Check if text is a question."""
        return text.strip().endswith('?')

    def _is_generic(self, text_lower: str) -> bool:
        """Check if text is a generic statement (not about self)."""
        for pattern in self.generic_patterns:
            if pattern.search(text_lower):
                return True
        return False

    def _is_empty_after_i(self, text_lower: str) -> bool:
        """Check if there's meaningful content after 'I'."""
        # Remove "I" and common contractions
        stripped = text_lower
        for prefix in ['i am ', "i'm ", 'i have ', "i've ", 'i will ', "i'll ",
                       'i would ', "i'd ", 'i ']:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break

        # Check if what remains is meaningful
        remaining = stripped.strip()
        if not remaining:
            return True

        # Check for just punctuation
        if re.match(r'^[.,;:!?\s]*$', remaining):
            return True

        return False


def validate_atoms(atoms: List[RawAtom]) -> ValidationResult:
    """
    Convenience function to validate atoms.

    Args:
        atoms: Raw atoms to validate

    Returns:
        ValidationResult with valid and invalid atoms
    """
    validator = BeliefAtomValidator()
    return validator.validate(atoms)
