import json
from pathlib import Path
from datetime import datetime

import pytest

import src.services.research_query_utils as query_utils

from src.services.research_query_utils import (
    sanitize_query,
    validate_llm_query,
    is_proper_noun_candidate,
    append_year_if_current_event,
)


@pytest.fixture
def known_bad_queries():
    fixture_path = Path(__file__).parent.parent / "fixtures" / "known_bad_queries.json"
    with open(fixture_path, "r", encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def test_sanitize_regressions(known_bad_queries):
    for case in known_bad_queries:
        sanitized, diag = sanitize_query(case["raw_input"])
        assert sanitized == case["expected_sanitized"]
        if case["expected_errors"]:
            # Validate_llm_query will emit errors, but sanitizer captures actions
            _, _, errors = validate_llm_query(case["raw_input"])
            for err in case["expected_errors"]:
                assert err in errors or err in diag.actions_taken


def test_sanitize_preserves_proper_nouns():
    sanitized, _ = sanitize_query("DOGE NASA Elon Musk SpaceX")
    tokens = sanitized.split()
    assert {"DOGE", "NASA", "Elon", "Musk", "SpaceX"}.issubset(set(tokens))


def test_sanitize_removes_forbidden_words():
    sanitized, diag = sanitize_query("what is the connection between A and B")
    assert "what" not in sanitized.lower()
    assert "connection" not in sanitized.lower()
    assert any(action.startswith("removed_forbidden") for action in diag.actions_taken)


def test_validate_llm_query_parses_ce_flag():
    query, is_ce, errors = validate_llm_query("NASA launch |ce=1")
    assert query == "NASA launch"
    assert is_ce is True
    assert errors == []


def test_validate_llm_query_handles_malformed_ce_flag():
    query, is_ce, errors = validate_llm_query("NASA |ce=1 launch")
    assert "malformed_ce_flag_position" in errors
    assert "|ce" not in query
    assert is_ce is False


def test_validate_llm_query_detects_multi_line():
    query, _, errors = validate_llm_query("line one\nline two")
    assert query == "line one"
    assert "multi_line_output" in errors


def test_is_proper_noun_candidate():
    assert is_proper_noun_candidate("NASA") is True
    assert is_proper_noun_candidate("Elon") is True
    assert is_proper_noun_candidate("mission") is False
    assert is_proper_noun_candidate("a") is False


def test_append_year_if_current_event(monkeypatch):
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls):
            return datetime(2024, 1, 1)

    monkeypatch.setattr(query_utils, "datetime", _FixedDatetime)

    query = append_year_if_current_event("NASA mission", True)
    assert query.endswith("2024")
    query = append_year_if_current_event("NASA 2023 mission", True)
    assert query.count("202") == 1
    assert append_year_if_current_event("History", False) == "History"
