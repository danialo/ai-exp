"""Fallback strategies for research query execution."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from config.research_config import FALLBACK_CONFIG, STOPWORDS
from src.services.research_query_utils import sanitize_query, is_proper_noun_candidate
from src.services.research_query_telemetry import QueryAttempt

SearchFn = Callable[[str], List]
AttemptBuilder = Callable[[str, str, List, dict], QueryAttempt]


def execute_with_fallback(
    initial_query: str,
    search_fn: SearchFn,
    attempt_builder: AttemptBuilder,
    *,
    known_entities: Sequence[str] | None = None,
    config: dict | None = None,
) -> Tuple[QueryAttempt, List]:
    """Run search with fallback strategies and return the winning attempt/results."""

    cfg = config or FALLBACK_CONFIG
    max_attempts = cfg.get("max_attempts", 3)
    min_score = cfg.get("min_acceptable_score", 0.0)
    strategies = cfg.get("strategies", [])

    attempts: List[QueryAttempt] = []

    def record_attempt(query: str, stage: str, diag: dict | None = None) -> Tuple[QueryAttempt, List]:
        results = search_fn(query)
        attempt = attempt_builder(query, stage, results, diag or {})
        attempts.append(attempt)
        return attempt, results

    best_attempt, best_results = record_attempt(initial_query, "initial")

    if best_attempt.composite_score >= min_score or max_attempts <= 1:
        best_attempt.is_winner = True
        return best_attempt, best_results

    for idx, strategy in enumerate(strategies):
        if len(attempts) >= max_attempts:
            break
        name = strategy.get("name", f"strategy_{idx}")
        candidate_query, diag = _apply_strategy(name, initial_query, known_entities)
        if not candidate_query:
            continue
        if candidate_query == attempts[-1].query:
            continue
        attempt, results = record_attempt(candidate_query, f"fallback_{name}", diag)
        if attempt.composite_score > best_attempt.composite_score:
            best_attempt = attempt
            best_results = results
        if attempt.composite_score >= min_score:
            break

    for att in attempts:
        att.is_winner = att is best_attempt

    return best_attempt, best_results


def _apply_strategy(name: str, query: str, known_entities: Sequence[str] | None) -> Tuple[str, dict]:
    known_entities = known_entities or []
    if name == "strip_suspect_words":
        sanitized, diag = sanitize_query(
            query,
            known_entities=known_entities,
            target_token_range=(3, 5),
        )
        return sanitized or query, {
            "removed_tokens": diag.removed_tokens,
            "suspect_tokens": diag.suspect_tokens,
            "actions_taken": diag.actions_taken + ["fallback_strip_suspect"],
        }
    if name == "proper_nouns_only":
        normalized_entities = {entity.lower() for entity in known_entities}
        tokens = [
            tok for tok in query.split()
            if is_proper_noun_candidate(tok) or tok.lower() in normalized_entities
        ]
        candidate = " ".join(tokens) if tokens else query
        return candidate, {"actions_taken": ["fallback_proper_nouns_only"]}
    if name == "top_keywords":
        tokens = [tok for tok in query.split() if tok.lower() not in STOPWORDS and len(tok) > 2]
        tokens.sort(key=len, reverse=True)
        candidate = " ".join(tokens[:3]) if tokens else query
        return candidate, {"actions_taken": ["fallback_top_keywords"]}
    return query, {"actions_taken": [f"fallback_unknown:{name}"]}
