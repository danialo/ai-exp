"""
Shared statement validation utilities for belief formation.

This module provides canonical validation logic used by both:
- ingest.py: Validates claims before creating SELF_DEFINITION experiences
- belief_gardener.py: Validates statements before forming beliefs

Keeping validation logic in ONE place prevents drift and maintenance headaches.
"""

import re
from typing import Optional


def canonicalize_statement(text: str) -> str:
    """
    Canonicalize a statement by collapsing all whitespace.

    This produces the form used for both validation AND grouping.

    Args:
        text: Raw statement text

    Returns:
        Canonical form with all whitespace collapsed to single spaces
    """
    return " ".join(text.strip().split())


def is_valid_statement(statement: str, source: Optional[str] = None) -> bool:
    """
    Validate that a statement is a complete, standalone self-claim.

    This function expects the canonical form (whitespace already collapsed).
    Use canonicalize_statement() first if needed.

    Args:
        statement: The statement to validate (should be canonical form)
        source: Provenance tag describing origin (REQUIRED for belief formation)
                Valid values: "claim_extractor" (LLM-validated claims only)

    Returns:
        True if the statement is valid, False otherwise
    """
    # Provenance-based filtering (strongest signal)
    # HARD REQUIREMENT: Only trust statements from the LLM claim extractor
    # Unknown provenance = reject by default
    if source != "claim_extractor":
        return False

    # Basic sanity checks
    if not statement or len(statement.strip()) < 10:
        return False

    stmt = statement.strip()
    stmt_lower = stmt.lower()

    # Reject list intros and list scaffolding
    # After whitespace collapse, "Here are key traits: 1. Foo" becomes "Here are key traits: 1. Foo"
    if stmt.endswith(':'):
        return False
    if re.search(r':\s*(\d+[\.\)]|[-*])\s+', stmt):
        return False
    if re.search(r'^\s*(\d+[\.\)]|[-*])\s+', stmt):
        return False

    # Reject discourse markers (transitional phrases that require context)
    discourse_markers = [
        r'^\s*instead\b',
        r'^\s*however\b',
        r'^\s*therefore\b',
        r'^\s*but\b',
        r'^\s*thus\b',
        r'^\s*moreover\b',
        r'^\s*furthermore\b',
        r'^\s*additionally\b',
    ]
    for pattern in discourse_markers:
        if re.search(pattern, stmt_lower):
            return False

    # Reject meta-commentary about the conversation/system
    # These are NOT self-claims - they're commentary about the interaction
    meta_patterns = [
        r'\b(this|the)\s+(conversation|chat|thread|exchange)\b',
        r'\b(my|these)\s+(response|responses|answer|answers)\b',
        r'\b(as an ai|language model)\b',
        r'\bi appreciate your perspective\b',
        r'\bit\'?s confusing\b',
        r'\bi see a bunch of\b',
        r'\bhere are\b',  # kills "Here are some key traits..."
        r'\bpledge enforcement\b',
        r'\bcorrected answer\b',
    ]
    if any(re.search(p, stmt_lower) for p in meta_patterns):
        return False

    # Reject markdown/template scaffolding
    template_patterns = [
        r'^\s*#{1,6}\s+',          # markdown headings
        r'^\s*[-*]\s+',            # bullets at start
        r'^\s*\*\*[^*]+\*\*[:\s]', # bold headers at start like "**Foo:**"
        r'^\s*ASSISTANT:',
        r'^\[Internal\s',
        r'^\[Emotional\s',
    ]
    for pattern in template_patterns:
        if re.search(pattern, stmt):
            return False

    # Reject statements with unresolved pronouns at the start
    # "They represent..." "This is..." "That shows..." need context
    if re.search(r'^\s*(they|this|that|these|those)\s+(is|are|was|were|represent|show|indicate)', stmt_lower):
        return False

    # Must contain a verb (heuristic for complete clause)
    # This is imperfect but catches fragments
    if not re.search(
        r'\b(am|is|are|was|were|have|has|had|do|does|did|will|would|can|could|feel|believe|want|need|prefer)\b',
        stmt_lower
    ):
        return False

    return True


def normalize_for_grouping(canonical_statement: str) -> str:
    """
    Normalize a canonical statement for grouping/deduplication.

    Args:
        canonical_statement: Statement already in canonical form

    Returns:
        Lowercase version for grouping
    """
    return canonical_statement.lower()


# Legacy helper for template noise detection (used by belief_gardener.py)
def looks_like_template_noise(text: str) -> bool:
    """
    Legacy helper for coarse template noise filtering.

    This is a first-pass filter before canonicalization.
    For precise validation, use is_valid_statement() on canonical form.

    Args:
        text: Raw text (may contain newlines, extra whitespace)

    Returns:
        True if text looks like template scaffolding
    """
    if not text or len(text.strip()) < 5:
        return True

    text_lower = text.lower()

    # Template prefixes
    if any(text.strip().startswith(prefix) for prefix in [
        "ASSISTANT:",
        "Response:",
        "[Internal",
        "[Emotional",
        "Corrected Answer",
    ]):
        return True

    # Markdown headings
    if re.match(r'^\s*#{1,6}\s+', text):
        return True

    return False
