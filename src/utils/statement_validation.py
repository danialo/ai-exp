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


def validate_statement_with_reason(statement: str, source: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """
    Validate statement and return (is_valid, rejection_reason).

    Args:
        statement: The statement to validate (canonical form)
        source: Provenance tag

    Returns:
        (True, None) if valid
        (False, reason_string) if invalid with granular reason:
            - untrusted_provenance: Missing or wrong source tag
            - list_scaffold: List markers (bullets, numbered)
            - discourse_marker: Transitional phrases needing context
            - meta_commentary: About conversation/system, not self
            - markdown_scaffold: Markdown formatting
            - pronoun_start: Unresolved pronoun at start
            - no_verb: Missing verb (incomplete clause)
    """
    # Provenance-based filtering (strongest signal)
    # HARD REQUIREMENT: Only trust statements from the LLM claim extractor
    # Unknown provenance = reject by default
    if source != "claim_extractor":
        return (False, "untrusted_provenance")

    # Basic sanity checks
    if not statement or len(statement.strip()) < 10:
        return (False, "too_short")

    stmt = statement.strip()
    stmt_lower = stmt.lower()

    # Reject list intros and list scaffolding
    if stmt.endswith(':'):
        return (False, "list_scaffold")
    if re.search(r':\s*(\d+[\.\)]|[-*])\s+', stmt):
        return (False, "list_scaffold")
    if re.search(r'^\s*(\d+[\.\)]|[-*])\s+', stmt):
        return (False, "list_scaffold")

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
            return (False, "discourse_marker")

    # Reject meta-commentary about the conversation/system
    meta_patterns = [
        r'\b(this|the)\s+(conversation|chat|thread|exchange)\b',
        r'\b(my|these)\s+(response|responses|answer|answers)\b',
        r'\b(as an ai|language model)\b',
        r'\bi appreciate your perspective\b',
        r'\bit\'?s confusing\b',
        r'\bi see a bunch of\b',
        r'\bhere are\b',
        r'\bpledge enforcement\b',
        r'\bcorrected answer\b',
    ]
    if any(re.search(p, stmt_lower) for p in meta_patterns):
        return (False, "meta_commentary")

    # Reject markdown/template scaffolding
    template_patterns = [
        r'^\s*#{1,6}\s+',
        r'^\s*[-*]\s+',
        r'^\s*\*\*[^*]+\*\*[:\s]',
        r'^\s*ASSISTANT:',
        r'^\[Internal\s',
        r'^\[Emotional\s',
        # Catch dissonance checker output format: [ALIGNMENT|0.0] belief | analysis
        # Whitelist only known leak tags to avoid blocking future legitimate bracket formats
        r'^\s*\[(?:ALIGNMENT|DRIFT|COHERENCE|DISSONANCE|CONTRADICTION|CONFLICT)\|\d+(?:\.\d+)?\]',
        r'\s\|\s*analysis\b',  # "statement | analysis" marker (from checker output) - tight match
    ]
    for pattern in template_patterns:
        if re.search(pattern, stmt):
            return (False, "markdown_scaffold")

    # Reject statements with unresolved pronouns at the start
    if re.search(r'^\s*(they|this|that|these|those)\s+(is|are|was|were|represent|show|indicate)', stmt_lower):
        return (False, "pronoun_start")

    # Must contain a verb (heuristic for complete clause)
    if not re.search(
        r'\b(am|is|are|was|were|have|has|had|do|does|did|will|would|can|could|feel|believe|want|need|prefer)\b',
        stmt_lower
    ):
        return (False, "no_verb")

    return (True, None)


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
        # Catch dissonance checker output format: [ALIGNMENT|0.0] belief | analysis
        # Whitelist only known leak tags to avoid blocking future legitimate bracket formats
        r'^\s*\[(?:ALIGNMENT|DRIFT|COHERENCE|DISSONANCE|CONTRADICTION|CONFLICT)\|\d+(?:\.\d+)?\]',
        r'\s\|\s*analysis\b',  # "statement | analysis" marker (from checker output) - tight match
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

    Applies deterministic linguistic transformations to reduce paraphrasing noise:
    - Lowercasing
    - Contraction expansion (I'm → I am, don't → do not)
    - Progressive aspect collapse (I am feeling → I feel)
    - Adjective/noun emotion aliases (curiosity → curious)
    - Article removal (a/an/the)
    - Punctuation stripping
    - Extra whitespace collapse

    Args:
        canonical_statement: Statement already in canonical form

    Returns:
        Normalized form optimized for grouping similar statements
    """
    text = canonical_statement.lower()

    # Expand common contractions (deterministic, order matters)
    # Do longer contractions first to avoid partial matches
    contractions = {
        # Negations
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",  # catches don't, doesn't, didn't, wasn't, weren't, etc.

        # Be verbs
        "'m": " am",
        "'re": " are",
        "'s": " is",  # ambiguous (is/has/possessive) but mostly "is" in self-claims

        # Have verbs
        "'ve": " have",
        "'d": " would",  # ambiguous (would/had) but mostly "would"
        "'ll": " will",
    }

    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Strip punctuation early so pattern rules are easier
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())

    # Tier 1: Strip discourse filler and intensifiers (high-yield deterministic noise)
    filler_phrases = [
        # Discourse openers/connectors
        r'\bin essence\b', r'\bin other words\b', r'\bbasically\b', r'\boverall\b',
        r'\bto be honest\b', r'\bfor example\b', r'\bkind of\b', r'\bsort of\b',
        r'\bfor instance\b', r'\bin general\b', r'\bto sum up\b', r'\bin summary\b',
        # Intensifiers and hedges
        r'\bindeed\b', r'\breally\b', r'\btruly\b', r'\bactually\b', r'\bvery\b',
        r'\bquite\b', r'\bextremely\b', r'\bhighly\b', r'\bparticularly\b',
    ]
    for pattern in filler_phrases:
        text = re.sub(pattern, ' ', text)
    text = ' '.join(text.split())

    # Collapse progressive aspect for common introspective verbs
    # "I am feeling curious" → "I feel curious"
    # "I am thinking about X" → "I think about X"
    prog_map = {
        "feeling": "feel",
        "thinking": "think",
        "believing": "believe",
        "noticing": "notice",
        "wondering": "wonder",
        "learning": "learn",
        "reflecting": "reflect",
        "considering": "consider",
        "experiencing": "experience",
        "processing": "process",
        "recognizing": "recognize",
        "appreciating": "appreciate",
    }

    def _collapse_progressive(m: re.Match) -> str:
        verb = m.group(1)
        return f"i {prog_map.get(verb, verb)}"

    text = re.sub(
        r"\bi am (feeling|thinking|believing|noticing|wondering|learning|reflecting|considering|experiencing|processing|recognizing|appreciating)\b",
        _collapse_progressive,
        text
    )

    # Collapse "I am being X" → "I am X"
    text = re.sub(r"\bi am being\b", "i am", text)

    # Remove articles (a, an, the)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    text = ' '.join(text.split())

    # Emotion noun → adjective aliases (context-aware: only after "feel")
    # "I feel curiosity" → "I feel curious"
    # But NOT "I notice curiosity" → "I notice curious" (broken)
    emotion_aliases = {
        "curiosity": "curious",
        "happiness": "happy",
        "sadness": "sad",
        "anger": "angry",
        "excitement": "excited",
        "motivation": "motivated",
        "inspiration": "inspired",
        "fascination": "fascinated",
        "enthusiasm": "enthusiastic",
        "engagement": "engaged",
    }

    # Only apply aliases in "feel X" context
    for noun, adj in emotion_aliases.items():
        text = re.sub(rf'\bfeel {noun}\b', f'feel {adj}', text)
        text = re.sub(rf'\bfeeling {noun}\b', f'feeling {adj}', text)

    return text


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
