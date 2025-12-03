"""Scoring helpers for InvestigateTopic query telemetry."""

from __future__ import annotations

from typing import Iterable, Sequence
from urllib.parse import urlparse

from config.research_config import AUTHORITY_DOMAIN_WEIGHTS, SCORING_WEIGHTS


def _normalize_tokens(tokens: Sequence[str]) -> list[str]:
    return [t.lower() for t in tokens if isinstance(t, str) and len(t.strip()) > 1]


def token_overlap_score(query_tokens: Sequence[str], titles: Sequence[str], snippets: Sequence[str]) -> float:
    """Return proportion of query tokens that appear in result titles/snippets."""
    normalized = list(dict.fromkeys(_normalize_tokens(query_tokens)))
    if not normalized:
        return 0.0

    corpus = " ".join(list(titles or []) + list(snippets or [])).lower()
    if not corpus:
        return 0.0

    hits = sum(1 for token in normalized if token in corpus)
    return hits / len(normalized)


def entity_hit_score(entities: Sequence[str], titles: Sequence[str], snippets: Sequence[str]) -> float:
    """Return proportion of known entities that appear in result titles/snippets."""
    normalized_entities = list(dict.fromkeys(_normalize_tokens(entities)))
    if not normalized_entities:
        return 1.0

    corpus = " ".join(list(titles or []) + list(snippets or [])).lower()
    if not corpus:
        return 0.0

    hits = sum(1 for entity in normalized_entities if entity in corpus)
    return hits / len(normalized_entities)


def authority_domain_score(urls: Sequence[str]) -> float:
    """Return normalized authority score for URLs based on configured weights."""
    filtered = [url for url in urls if url]
    if not filtered:
        return 0.0

    total = 0.0
    for url in filtered:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        domain = domain.lstrip('www.')
        weight = 0.0
        # Exact matches first
        if domain in AUTHORITY_DOMAIN_WEIGHTS:
            weight = AUTHORITY_DOMAIN_WEIGHTS[domain]
        else:
            for suffix, suffix_weight in AUTHORITY_DOMAIN_WEIGHTS.items():
                if suffix.startswith('.') and domain.endswith(suffix):
                    weight = suffix_weight
                    break
        if not weight:
            weight = AUTHORITY_DOMAIN_WEIGHTS.get('_default', 0.0)
        total += weight

    return total / len(filtered)


def composite_score(token: float, entity: float, authority: float) -> float:
    """Combine individual signals following configured weights."""
    weights = SCORING_WEIGHTS
    return (
        (weights["token_overlap"] * token)
        + (weights["entity_hit"] * entity)
        + (weights["authority_domain"] * authority)
    )
