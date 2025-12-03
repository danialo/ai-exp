import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.services.research_htn_methods import investigate_topic
from src.services.web_search_service import SearchResult
import src.services.research_session as research_session_module


class DummySession:
    def __init__(self, root_question: str):
        self.root_question = root_question


class DummyResearchSessionStore:
    def __init__(self):
        self.session = DummySession("Root question")

    def get_session(self, session_id: str):
        return self.session


@pytest.fixture(autouse=True)
def patch_research_session_store(monkeypatch):
    monkeypatch.setattr(
        research_session_module,
        "ResearchSessionStore",
        lambda: DummyResearchSessionStore(),
    )


def _build_ctx(search_side_effect):
    llm_mock = MagicMock()
    llm_mock.tool_call.side_effect = [
        "DOGE government agency Elon Musk connection |ce=1",
        json.dumps({
            "claims": ["Claim one"],
            "summary": "Summary",
            "follow_up_questions": ["Question one", "Question two"],
        }),
        json.dumps([1, 2]),
    ]

    web_search_mock = MagicMock(side_effect=search_side_effect)

    fetch_mock = MagicMock(return_value=SimpleNamespace(success=True, main_content="Example article"))

    session_store = SimpleNamespace(create_source_doc=MagicMock())

    ctx = SimpleNamespace(
        llm_service=llm_mock,
        web_search_service=SimpleNamespace(search=web_search_mock),
        url_fetcher_service=SimpleNamespace(fetch_url=fetch_mock),
        session_store=session_store,
        task_store=SimpleNamespace(),
    )
    return ctx, web_search_mock, fetch_mock


def test_investigate_topic_uses_fallback_when_initial_results_empty():
    fallback_result = SearchResult(
        title="Fallback title",
        url="https://www.reuters.com/article/fallback",
        snippet="DOGE related news",
        position=1,
        timestamp=datetime.now(),
    )

    ctx, search_mock, fetch_mock = _build_ctx([
        [],
        [fallback_result],
    ])

    task = SimpleNamespace(
        id="task-1",
        session_id="session-1",
        args={"topic": "DOGE government agency Elon Musk connection"},
        status="pending",
        depth=0,
        parent_id=None,
        created_at=None,
        updated_at=None,
    )

    proposals = investigate_topic(task, ctx)

    assert proposals  # follow-ups generated
    assert search_mock.call_count == 2
    # Second query should be stripped of "connection"
    second_query = search_mock.call_args_list[1].args[0]
    assert "connection" not in second_query.lower()
    # Ensure article fetch used fallback result URL
    fetch_mock.assert_called_once_with(fallback_result.url)
