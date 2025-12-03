"""Validation and sanitization helpers for research queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Sequence, Tuple

from config.research_config import FORBIDDEN_ALWAYS, CONTEXT_SUSPECT, STOPWORDS

QUERY_VALIDATION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\s\-']+$")


@dataclass
class SanitizerDiagnostics:
    """Details on how a query string was sanitized."""

    original_query: str
    final_query: str = ""
    original_token_count: int = 0
    final_token_count: int = 0
    removed_tokens: List[str] = field(default_factory=list)
    suspect_tokens: List[str] = field(default_factory=list)
    preserved_proper_nouns: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)


FORBIDDEN_ALWAYS_SET = {w.lower() for w in FORBIDDEN_ALWAYS}
CONTEXT_SUSPECT_SET = {w.lower() for w in CONTEXT_SUSPECT}
STOPWORD_SET = {w.lower() for w in STOPWORDS}


def validate_llm_query(raw: str) -> tuple[str, bool, List[str]]:
    """Validate raw LLM output and return (query, is_current_event, errors)."""
    errors: List[str] = []
    if raw is None:
        return "", False, ["empty_output"]

    text = raw.strip()
    if not text:
        return "", False, ["empty_output"]

    if "\n" in text:
        errors.append("multi_line_output")
        text = text.splitlines()[0].strip()

    is_current_event = False
    if text.endswith("|ce=1"):
        is_current_event = True
        text = text[:-5].strip()
    elif text.endswith("|ce=0"):
        text = text[:-5].strip()
    elif "|ce=" in text:
        errors.append("malformed_ce_flag_position")
        text = re.sub(r"\|ce=[01]", "", text).strip()

    if '?' in text:
        errors.append("contains_question_mark")
        text = text.replace('?', ' ')

    base = text
    if not QUERY_VALIDATION_PATTERN.match(base):
        errors.append("regex_validation_failed")

    tokens = [tok for tok in base.split() if tok]
    if len(tokens) > 10:
        errors.append(f"too_many_tokens:{len(tokens)}")
    if len(tokens) < 2:
        errors.append(f"too_few_tokens:{len(tokens)}")

    lowered = [tok.lower() for tok in tokens]
    for forbidden in FORBIDDEN_ALWAYS_SET:
        if forbidden in lowered:
            errors.append(f"forbidden_word:{forbidden}")
    for suspect in CONTEXT_SUSPECT_SET:
        if suspect in lowered:
            errors.append(f"suspect_word:{suspect}")

    clean_query = " ".join(tokens)
    return clean_query, is_current_event, errors


def is_proper_noun_candidate(token: str) -> bool:
    """Heuristic for detecting tokens that look like proper nouns."""
    if not token or len(token) < 2:
        return False
    if token.isupper() and token.isalpha():
        return True
    if token[0].isupper() and not token.isupper():
        return True
    return False


def _strip_punctuation(query: str) -> str:
    return re.sub(r"[\"',.;:!?()\[\]{}]", "", query)


def sanitize_query(
    query: str,
    *,
    strip_suspect: bool = True,
    known_entities: Sequence[str] | None = None,
    target_token_range: Tuple[int, int] = (3, 7),
) -> tuple[str, SanitizerDiagnostics]:
    """Return sanitized query plus diagnostics."""
    diag = SanitizerDiagnostics(original_query=query)
    cleaned = _strip_punctuation(query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    diag.actions_taken.append("stripped_punctuation")

    tokens = cleaned.split()
    diag.original_token_count = len(tokens)

    known_entities = known_entities or []
    known_entity_set = {ent.lower() for ent in known_entities}

    proper_nouns = []
    for token in tokens:
        if is_proper_noun_candidate(token) or token.lower() in known_entity_set:
            proper_nouns.append(token)
            diag.preserved_proper_nouns.append(token)

    filtered: List[str] = []
    for token in tokens:
        if token.lower() in FORBIDDEN_ALWAYS_SET:
            diag.removed_tokens.append(token)
            continue
        filtered.append(token)
    if diag.removed_tokens:
        diag.actions_taken.append(f"removed_forbidden:{len(diag.removed_tokens)}")

    suspect_marked: List[str] = []
    filtered_suspect: List[str] = []
    for token in filtered:
        lower = token.lower()
        if lower in CONTEXT_SUSPECT_SET:
            suspect_marked.append(token)
            if strip_suspect:
                diag.removed_tokens.append(token)
                continue
        filtered_suspect.append(token)
    if suspect_marked:
        diag.suspect_tokens.extend(suspect_marked)
        diag.actions_taken.append(f"flagged_suspect:{len(suspect_marked)}")
    tokens = filtered_suspect

    deduped: List[str] = []
    seen = set()
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    if len(deduped) < len(tokens):
        diag.actions_taken.append(f"deduped:{len(tokens) - len(deduped)}")
    tokens = deduped

    # Remove stopwords if enough tokens remain
    filtered_stopwords: List[str] = []
    removed_stopwords = 0
    for token in tokens:
        if token.lower() in STOPWORD_SET and len(tokens) - removed_stopwords > target_token_range[0]:
            diag.removed_tokens.append(token)
            removed_stopwords += 1
            continue
        filtered_stopwords.append(token)
    if removed_stopwords:
        diag.actions_taken.append(f"removed_stopwords:{removed_stopwords}")
    tokens = filtered_stopwords

    min_tokens, max_tokens = target_token_range
    if len(tokens) > max_tokens:
        prioritized: List[str] = []
        used = set()
        # First, keep proper nouns and known entities
        for token in tokens:
            if token in prioritized:
                continue
            if token in proper_nouns or token.lower() in known_entity_set:
                prioritized.append(token)
                used.add(token)
        # Next, keep informative tokens (non-stopwords)
        for token in tokens:
            lower = token.lower()
            if token in used:
                continue
            if lower not in STOPWORD_SET:
                prioritized.append(token)
                used.add(token)
        # Finally, fill remaining slots with whatever is left in order
        for token in tokens:
            if token in used:
                continue
            prioritized.append(token)
            used.add(token)
        tokens = prioritized[:max_tokens]
        diag.actions_taken.append(f"collapsed_to_max:{max_tokens}")

    diag.final_token_count = len(tokens)
    diag.final_query = " ".join(tokens)

    if diag.final_token_count < min_tokens and diag.final_query:
        diag.actions_taken.append("below_min_tokens")

    return diag.final_query, diag


def append_year_if_current_event(query: str, is_current_event: bool) -> str:
    """Append current year token when current-event flag is set."""
    if not is_current_event:
        return query

    if re.search(r"\b20[0-9]{2}\b", query):
        return query

    current_year = datetime.now().year
    if query:
        return f"{query} {current_year}"
    return str(current_year)
