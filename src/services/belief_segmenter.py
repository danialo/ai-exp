"""
Belief Segmenter for HTN Self-Belief Decomposer.

Splits text into claim candidates using deterministic rules:
- Sentence boundaries (.!?)
- Semicolons
- Coordinating conjunctions (when connecting independent clauses)
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ClaimCandidate:
    """
    A candidate claim extracted from text.

    Attributes:
        text: The claim text (stripped)
        span: (start, end) character positions in original text
    """
    text: str
    span: Tuple[int, int]


class BeliefSegmenter:
    """
    Deterministic claim candidate extraction.

    Splits text on:
    1. Sentence-ending punctuation (. ! ?)
    2. Semicolons (;)
    3. Coordinating conjunctions connecting independent clauses
       (and, but, however, although, though, yet)

    Subordinate clauses stay attached to their main clause.
    Only returns self-referential claims (containing I/my/me references).
    """

    # Conjunctions that typically connect independent clauses
    SPLITTING_CONJUNCTIONS = {
        'and', 'but', 'however', 'although', 'though', 'yet',
        'so', 'while', 'whereas'
    }

    # Self-reference markers for belief filtering
    SELF_REFERENCE_MARKERS = {
        'i ', "i'm", "i've", "i'd", "i'll",
        'my ', 'me ', 'myself',
        'mine '
    }

    # Subordinating conjunctions - these typically introduce dependent clauses
    # that should NOT cause a split
    SUBORDINATING_CONJUNCTIONS = {
        'because', 'since', 'unless', 'until', 'when', 'whenever',
        'where', 'wherever', 'whether', 'if', 'after', 'before',
        'as', 'that', 'which', 'who', 'whom', 'whose'
    }

    def segment(self, text: str) -> List[ClaimCandidate]:
        """
        Split text into claim candidates.

        Args:
            text: Input text to segment

        Returns:
            List of ClaimCandidate objects with text and spans
        """
        if not text or not text.strip():
            return []

        candidates = []

        # First split on sentence boundaries and semicolons
        sentence_splits = self._split_on_sentences(text)

        # Then split each sentence on conjunctions
        for sentence_text, sentence_span in sentence_splits:
            clause_splits = self._split_on_conjunctions(
                sentence_text,
                offset=sentence_span[0]
            )
            candidates.extend(clause_splits)

        # Filter out empty candidates and non-self-referential text
        candidates = [c for c in candidates if c.text.strip()]
        candidates = [c for c in candidates if self.is_self_referential(c.text)]

        return candidates

    def is_self_referential(self, text: str) -> bool:
        """
        Check if text contains self-reference (I/my/me).

        Filters out general observations like "The weather is nice"
        that aren't about the speaker.

        Args:
            text: Text to check

        Returns:
            True if text contains self-reference markers
        """
        text_lower = text.lower()
        return any(marker in text_lower for marker in self.SELF_REFERENCE_MARKERS)

    def _split_on_sentences(
        self,
        text: str
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Split text on sentence boundaries and semicolons.

        Returns list of (text, (start, end)) tuples.
        """
        # Pattern: sentence-ending punctuation or semicolon
        # Keep the punctuation with the preceding text
        pattern = r'([.!?;])\s*'

        parts = []
        last_end = 0

        for match in re.finditer(pattern, text):
            end_pos = match.end()
            part_text = text[last_end:end_pos].strip()
            if part_text:
                parts.append((part_text, (last_end, end_pos)))
            last_end = end_pos

        # Don't forget any remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                parts.append((remaining, (last_end, len(text))))

        # If no splits found, return the whole text
        if not parts:
            parts = [(text.strip(), (0, len(text)))]

        return parts

    def _split_on_conjunctions(
        self,
        text: str,
        offset: int = 0
    ) -> List[ClaimCandidate]:
        """
        Split text on coordinating conjunctions.

        Only splits if the conjunction appears to connect independent clauses
        (i.e., both sides have potential subject-verb structure).

        Args:
            text: Text to split
            offset: Character offset in original text

        Returns:
            List of ClaimCandidate objects
        """
        # Build pattern for splitting conjunctions
        conj_pattern = r'\s+(' + '|'.join(self.SPLITTING_CONJUNCTIONS) + r')\s+'

        # Find all conjunction positions
        matches = list(re.finditer(conj_pattern, text, re.IGNORECASE))

        if not matches:
            # No conjunctions found, return as single candidate
            return [ClaimCandidate(
                text=text.strip(),
                span=(offset, offset + len(text))
            )]

        candidates = []
        last_end = 0

        for match in matches:
            conj_start = match.start()
            conj_end = match.end()
            conjunction = match.group(1).lower()

            # Get text before this conjunction
            before_text = text[last_end:conj_start].strip()
            after_text = text[conj_end:].strip()

            # Check if this looks like it connects independent clauses
            # Heuristic: both sides should have at least 2 words and
            # look like they could stand alone
            if self._looks_like_independent_clause(before_text) and \
               self._looks_like_independent_clause(after_text):
                # Split here
                if before_text:
                    candidates.append(ClaimCandidate(
                        text=before_text,
                        span=(offset + last_end, offset + conj_start)
                    ))
                last_end = conj_end
            else:
                # Don't split - subordinate clause or not independent
                continue

        # Add remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                candidates.append(ClaimCandidate(
                    text=remaining,
                    span=(offset + last_end, offset + len(text))
                ))

        # If no splits happened, return original
        if not candidates:
            candidates = [ClaimCandidate(
                text=text.strip(),
                span=(offset, offset + len(text))
            )]

        return candidates

    def _looks_like_independent_clause(self, text: str) -> bool:
        """
        Check if text looks like it could be an independent clause.

        Heuristics:
        - At least 2 words
        - Contains a potential subject (pronoun or noun phrase)
        - Not just a prepositional phrase or fragment
        """
        if not text:
            return False

        words = text.split()
        if len(words) < 2:
            return False

        text_lower = text.lower()

        # Check for common subordinate introductions
        for sub_conj in self.SUBORDINATING_CONJUNCTIONS:
            if text_lower.startswith(sub_conj + ' '):
                return False

        # Check for first-person subject (common in self-beliefs)
        if text_lower.startswith('i ') or text_lower.startswith("i'm ") or \
           text_lower.startswith("i've ") or text_lower.startswith("i'd "):
            return True

        # Check for other common subjects
        subject_starts = ['we ', 'you ', 'he ', 'she ', 'they ', 'it ', 'this ', 'that ']
        for start in subject_starts:
            if text_lower.startswith(start):
                return True

        # If we have at least 3 words and no subordinating intro, likely independent
        if len(words) >= 3:
            return True

        return False


def segment_text(text: str) -> List[ClaimCandidate]:
    """
    Convenience function to segment text into claim candidates.

    Args:
        text: Input text

    Returns:
        List of ClaimCandidate objects
    """
    segmenter = BeliefSegmenter()
    return segmenter.segment(text)
