"""Research session tools for Astra."""

from typing import Optional, Dict, Any
from src.services.task_queue import TaskStore, new_task
from src.services.research_session import ResearchSession, ResearchSessionStore
from src.services.htn_task_executor import HTNTaskExecutor, ExecutionContext


def start_research_session(
    question: str,
    max_tasks: int = 50,
    max_children_per_task: int = 5,
    max_depth: int = 4,
    metadata: Optional[dict] = None,
) -> str:
    """
    Start a new research session with HTN task decomposition.

    Args:
        question: Root research question
        max_tasks: Maximum total tasks allowed (budget control)
        max_children_per_task: Maximum children per parent task
        max_depth: Maximum task decomposition depth
        metadata: Optional metadata dict

    Returns:
        session_id: UUID of created session
    """
    sess_store = ResearchSessionStore()
    task_store = TaskStore()

    # Create session
    sess = ResearchSession(
        root_question=question,
        max_tasks=max_tasks,
        max_children_per_task=max_children_per_task,
        max_depth=max_depth,
        metadata=metadata or {},
    )
    sess_store.create_session(sess)

    # Create root task
    root = new_task(
        session_id=sess.id,
        htn_task_type="ResearchCurrentEvents",
        args={"root_question": question},
        depth=0,
        parent_id=None
    )
    task_store.create_one(root)

    # Increment session counter
    sess_store.increment_tasks_created(sess.id, 1)

    return sess.id


def research_and_summarize(
    question: str,
    max_tasks: int = 30,
    max_children_per_task: int = 3,
    max_depth: int = 4,
    llm_service=None,
    web_search_service=None,
    url_fetcher_service=None,
    metadata: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Complete research pipeline: Start session, execute until complete, return synthesis.

    This is the top-level tool for Astra to autonomously research a question.

    Args:
        question: Root research question
        max_tasks: Maximum total tasks (budget control)
        max_children_per_task: Max children per parent
        max_depth: Max decomposition depth
        llm_service: LLM service for query generation and analysis
        web_search_service: Web search service
        url_fetcher_service: URL content fetcher
        metadata: Optional metadata

    Returns:
        session_summary: {
            "narrative_summary": str,
            "key_events": List[str],
            "contested_claims": List[dict],
            "open_questions": List[str],
            "coverage_stats": dict,
            "session_id": str  # Added for reference
        }

    Example:
        >>> summary = research_and_summarize(
        ...     question="What happened in AI safety this week?",
        ...     llm_service=astra_llm,
        ...     web_search_service=web_search,
        ...     url_fetcher_service=url_fetcher
        ... )
        >>> print(summary["narrative_summary"])
        >>> for event in summary["key_events"]:
        ...     print(f"- {event}")
    """
    # Validate required services
    if not llm_service:
        raise ValueError("llm_service is required")
    if not web_search_service:
        raise ValueError("web_search_service is required")
    if not url_fetcher_service:
        raise ValueError("url_fetcher_service is required")

    # Start session
    session_id = start_research_session(
        question=question,
        max_tasks=max_tasks,
        max_children_per_task=max_children_per_task,
        max_depth=max_depth,
        metadata=metadata
    )

    # Build execution context
    task_store = TaskStore()
    session_store = ResearchSessionStore()

    exec_ctx = ExecutionContext(
        llm_service=llm_service,
        web_search_service=web_search_service,
        url_fetcher_service=url_fetcher_service,
        session_store=session_store,
        task_store=task_store
    )

    # Execute until complete
    executor = HTNTaskExecutor(
        task_store=task_store,
        session_store=session_store,
        ctx=exec_ctx
    )
    executor.run_until_empty(session_id=session_id)

    # Retrieve synthesis
    session = session_store.get_session(session_id)
    if not session or not session.session_summary:
        return {
            "error": "Synthesis failed",
            "session_id": session_id,
            "narrative_summary": "Research completed but synthesis unavailable",
            "key_events": [],
            "contested_claims": [],
            "open_questions": [],
            "coverage_stats": {}
        }

    # Add session_id to summary for reference
    summary = dict(session.session_summary)
    summary["session_id"] = session_id

    return summary
