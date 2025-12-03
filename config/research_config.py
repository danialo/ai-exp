"""Configuration constants for the research HTN pipeline."""

from __future__ import annotations

from typing import Dict, List

# Authority weights used when evaluating search result quality.
AUTHORITY_DOMAIN_WEIGHTS: Dict[str, float] = {
    # Tier 1 - primary sources
    ".gov": 1.0,
    ".edu": 0.95,
    "whitehouse.gov": 1.0,
    "sec.gov": 1.0,

    # Tier 2 - major news organizations
    "apnews.com": 0.8,
    "reuters.com": 0.8,
    "nytimes.com": 0.75,
    "wsj.com": 0.75,
    "bbc.com": 0.75,
    "npr.org": 0.7,
    "washingtonpost.com": 0.7,

    # Tier 3 - secondary aggregators / knowledge bases
    "wikipedia.org": 0.5,
    "investopedia.com": 0.45,

    # Default weight when no explicit mapping is found
    "_default": 0.2,
}

# Weights for computing composite relevance score.
SCORING_WEIGHTS = {
    "token_overlap": 0.4,
    "entity_hit": 0.3,
    "authority_domain": 0.3,
}

# Fallback configuration will be used later in the pipeline (Phase 3) but kept here
# so other modules can read defaults from a single location.
FALLBACK_CONFIG = {
    "max_attempts": 3,
    "min_acceptable_score": 0.35,
    "strategies": [
        {"name": "strip_suspect_words"},
        {"name": "proper_nouns_only"},
        {"name": "top_keywords"},
    ],
}

# Sanitizer configuration placeholders (will be consumed in Phase 2+3 modules).
FORBIDDEN_ALWAYS: List[str] = [
    "about",
    "regarding",
    "concerning",
    "between",
    "what",
    "how",
    "why",
    "is",
    "are",
    "the",
    "a",
    "an",
]

CONTEXT_SUSPECT: List[str] = [
    "connection",
    "relationship",
    "impact",
    "effect",
    "influence",
    "role",
]

STOPWORDS: List[str] = [
    "of",
    "and",
    "or",
    "for",
    "in",
    "on",
    "to",
    "with",
    "by",
]
