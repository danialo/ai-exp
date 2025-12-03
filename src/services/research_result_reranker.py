"""Authority-based reranker for search results."""

from __future__ import annotations

from typing import List, Sequence

from config.research_config import AUTHORITY_DOMAIN_WEIGHTS
from src.services.web_search_service import SearchResult


def rerank_results(results: Sequence[SearchResult], boost_authority: bool = True) -> List[SearchResult]:
    """Return results sorted by domain authority and original rank."""
    if not boost_authority:
        return list(results)

    def get_weight(url: str) -> float:
        if not url:
            return AUTHORITY_DOMAIN_WEIGHTS.get("_default", 0.0)
        normalized = url.lower()
        for domain, weight in AUTHORITY_DOMAIN_WEIGHTS.items():
            if domain == "_default":
                continue
            if domain.startswith(".") and normalized.endswith(domain):
                return weight
            if domain in normalized:
                return weight
        return AUTHORITY_DOMAIN_WEIGHTS.get("_default", 0.0)

    scored = []
    for idx, result in enumerate(results):
        authority_weight = get_weight(result.url)
        position_weight = 1.0 / (idx + 1)
        final_score = authority_weight * 0.6 + position_weight * 0.4
        scored.append((final_score, result))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [res for _, res in scored]
