import pytest

from src.services.research_query_scoring import (
    token_overlap_score,
    entity_hit_score,
    authority_domain_score,
    composite_score,
)


def test_token_overlap_score_counts_matches():
    tokens = ["DOGE", "agency", "Elon", "Musk"]
    titles = ["DOGE controversy ties to government agency"]
    snippets = ["Elon Musk mentions DOGE in interview"]

    score = token_overlap_score(tokens, titles, snippets)

    assert 0 < score <= 1
    assert score == pytest.approx(1.0)  # All tokens appear in combined snippets


def test_token_overlap_score_handles_empty_inputs():
    assert token_overlap_score([], [], []) == 0.0
    assert token_overlap_score(["a"], [], []) == 0.0


def test_entity_hit_score_defaults_to_one_without_entities():
    score = entity_hit_score([], ["Title"], ["Snippet"])
    assert score == 1.0


def test_entity_hit_score_counts_matches():
    entities = ["Elon Musk", "DOGE"]
    titles = ["Elon Musk discusses DOGE"]
    snippets = ["No mention of agency"]

    score = entity_hit_score(entities, titles, snippets)
    assert score == pytest.approx(1.0)


def test_entity_hit_score_zero_without_results():
    entities = ["Elon Musk"]
    score = entity_hit_score(entities, [], [])
    assert score == 0.0


def test_authority_domain_score_prefers_known_domains():
    urls = [
        "https://www.reuters.com/article/abc",
        "https://randomblog.net/post",
        "https://whitehouse.gov/news",
    ]

    score = authority_domain_score(urls)

    assert score > 0.2  # better than default average


def test_authority_domain_score_handles_empty():
    assert authority_domain_score([]) == 0.0


def test_composite_score_uses_weights():
    score = composite_score(0.5, 0.25, 0.75)
    assert 0 <= score <= 1
    # Should weight token 0.4, entity 0.3, authority 0.3
    assert score == pytest.approx(0.4 * 0.5 + 0.3 * 0.25 + 0.3 * 0.75)
