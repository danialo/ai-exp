"""
Belief Canonicalizer for HTN Self-Belief Decomposer.

Normalizes belief atoms for stable matching:
- Lowercase
- Expand contractions
- Normalize whitespace
- Remove trailing punctuation
- Normalize unicode
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, List, Optional


# Contraction expansion map
CONTRACTIONS = {
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "who's": "who is",
    "let's": "let us",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "he's": "he is",
    "she's": "she is",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
}


@dataclass
class CanonicalAtom:
    """
    A canonicalized belief atom.

    Attributes:
        original_text: Original text before canonicalization
        canonical_text: Normalized text
        canonical_hash: SHA256 hash of canonical text (32 chars)
        belief_type: Ontological category
        polarity: affirm or deny
        spans: Character spans in source (can be multiple after dedup)
        confidence: Extraction confidence
    """
    original_text: str
    canonical_text: str
    canonical_hash: str
    belief_type: str
    polarity: str
    spans: Optional[List[tuple]] = None
    confidence: float = 1.0


class BeliefCanonicalizer:
    """
    Normalize belief atoms for stable matching.

    Canonicalization steps:
    1. Lowercase
    2. Expand contractions
    3. Normalize whitespace (collapse to single spaces, strip)
    4. Remove trailing punctuation
    5. Normalize unicode (NFC normalization)
    """

    def __init__(self):
        # Build regex pattern for contraction matching
        # Sort by length (longest first) to handle overlaps
        sorted_contractions = sorted(
            CONTRACTIONS.keys(),
            key=len,
            reverse=True
        )
        pattern = r'\b(' + '|'.join(re.escape(c) for c in sorted_contractions) + r')\b'
        self.contraction_pattern = re.compile(pattern, re.IGNORECASE)

    def canonicalize(self, text: str) -> str:
        """
        Canonicalize a text string.

        Args:
            text: Input text

        Returns:
            Canonicalized text
        """
        if not text:
            return ""

        # 1. Normalize unicode (NFC)
        result = unicodedata.normalize('NFC', text)

        # 2. Lowercase
        result = result.lower()

        # 3. Expand contractions
        def expand_contraction(match):
            contraction = match.group(0).lower()
            return CONTRACTIONS.get(contraction, contraction)

        result = self.contraction_pattern.sub(expand_contraction, result)

        # 4. Normalize whitespace
        result = ' '.join(result.split())

        # 5. Remove trailing punctuation
        result = result.rstrip('.,;:!?')

        return result.strip()

    def compute_hash(self, canonical_text: str) -> str:
        """
        Compute SHA256 hash of canonical text.

        Args:
            canonical_text: Canonicalized text

        Returns:
            32-character hexadecimal hash
        """
        if not canonical_text:
            return hashlib.sha256(b'').hexdigest()[:32]

        hash_bytes = hashlib.sha256(canonical_text.encode('utf-8')).hexdigest()
        return hash_bytes[:32]

    def canonicalize_atom(self, atom: Any) -> CanonicalAtom:
        """
        Canonicalize a raw atom object.

        Args:
            atom: Raw atom with atom_text, belief_type, polarity, confidence, spans

        Returns:
            CanonicalAtom with canonical text and hash
        """
        original = getattr(atom, 'atom_text', str(atom))
        canonical = self.canonicalize(original)
        hash_val = self.compute_hash(canonical)

        return CanonicalAtom(
            original_text=original,
            canonical_text=canonical,
            canonical_hash=hash_val,
            belief_type=getattr(atom, 'belief_type', 'UNKNOWN'),
            polarity=getattr(atom, 'polarity', 'affirm'),
            spans=getattr(atom, 'spans', None),
            confidence=getattr(atom, 'confidence', 1.0),
        )

    def canonicalize_atoms(self, atoms: List[Any]) -> List[CanonicalAtom]:
        """
        Canonicalize all atoms and sort by canonical_text for stability.

        Args:
            atoms: List of raw atoms

        Returns:
            List of CanonicalAtom sorted by canonical_text
        """
        canonical_atoms = [self.canonicalize_atom(a) for a in atoms]

        # Sort by canonical_text for deterministic ordering
        canonical_atoms.sort(key=lambda a: a.canonical_text)

        return canonical_atoms


def canonicalize_text(text: str) -> str:
    """
    Convenience function to canonicalize text.

    Args:
        text: Input text

    Returns:
        Canonicalized text
    """
    canonicalizer = BeliefCanonicalizer()
    return canonicalizer.canonicalize(text)


def compute_canonical_hash(text: str) -> str:
    """
    Convenience function to compute canonical hash.

    Args:
        text: Input text (will be canonicalized first)

    Returns:
        32-character hash
    """
    canonicalizer = BeliefCanonicalizer()
    canonical = canonicalizer.canonicalize(text)
    return canonicalizer.compute_hash(canonical)
