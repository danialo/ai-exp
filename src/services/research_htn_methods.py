"""HTN methods for autonomous research with session budgets."""

import json
import logging
from typing import Any, Dict, List, Callable
from src.utils.logging_config import get_multi_logger

logger = logging.getLogger(__name__)

# Global method registry
METHODS: Dict[str, Callable] = {}


def method(name: str):
    """Decorator to register HTN method implementations."""
    def decorator(func: Callable):
        METHODS[name] = func
        logger.info(f"Registered HTN method: {name}")
        return func
    return decorator


def get_method(name: str) -> Callable:
    """Get registered method by name."""
    return METHODS.get(name)


def list_methods() -> List[str]:
    """List all registered method names."""
    return list(METHODS.keys())


# === P1: Research HTN Methods ===

@method("ResearchCurrentEvents")
def research_current_events(task, ctx) -> List[Dict[str, Any]]:
    """
    Root method: Generate seed queries from root question and create InvestigateTopic tasks.

    Args:
        task: Task with args {"root_question": str}
        ctx: ExecutionContext with llm_service, session_store, etc.

    Returns:
        List of child task proposals
    """
    root_question = task.args.get("root_question", "What are the most important current events today?")

    # Generate seed topics using LLM
    prompt = f"""Given this research question: "{root_question}"

Generate 3-5 specific topics to investigate. Return as JSON:
{{
  "topics": ["topic1", "topic2", "topic3"]
}}"""

    try:
        response = ctx.llm_service.generate_with_tools(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.7
        )
        seed_topics = json.loads(response["message"].content).get("topics", [])
    except Exception as e:
        logger.error(f"Failed to generate seed topics: {e}")
        seed_topics = [root_question]  # Fallback

    # Return child task proposals
    proposals = []
    for i, topic in enumerate(seed_topics[:5]):
        proposals.append({
            "htn_task_type": "InvestigateTopic",
            "args": {"topic": topic},
            "dedup_key": f"topic::{topic}",
            "depth": task.depth + 1
        })

    logger.info(f"ResearchCurrentEvents: Generated {len(proposals)} InvestigateTopic proposals")
    return proposals


@method("InvestigateTopic")
def investigate_topic(task, ctx) -> List[Dict[str, Any]]:
    """
    Search for topic, read first result, extract claims and follow-up questions.

    Args:
        task: Task with args {"topic": str}
        ctx: ExecutionContext

    Returns:
        List of InvestigateQuestion child proposals
    """
    topic = task.args.get("topic", "")
    if not topic:
        logger.warning("InvestigateTopic called with no topic")
        return []

    try:
        # 1. Generate search query
        query_prompt = f"""Generate a web search query for: {topic}

Rules:
- Use 3-8 keyword phrases that a human would type into Google
- NO quotation marks around the entire query
- NO full sentences or natural language
- NO extra explanation
- Just keywords, separated by spaces

Example good queries:
- Elon Musk DOGE government efficiency 2024
- climate change latest IPCC report
- cryptocurrency regulation SEC

Query:"""
        query_response = ctx.llm_service.generate_with_tools(
            messages=[{"role": "user", "content": query_prompt}],
            tools=None,
            temperature=0.3
        )
        raw_query = query_response["message"].content.strip()

        # Sanitize query: strip wrapping quotes, collapse whitespace, limit tokens
        search_query = raw_query.strip('"').strip("'")
        search_query = " ".join(search_query.split())
        tokens = search_query.split()
        if len(tokens) > 10:
            search_query = " ".join(tokens[:10])

        # 2. Search web (try with more results)
        search_results = ctx.web_search_service.search(search_query, num_results=5)

        # Fallback: if no results, try simpler query with first 4 tokens
        if not search_results and len(tokens) > 4:
            simpler_query = " ".join(tokens[:4])
            logger.warning(f"No results for '{search_query}', trying simpler: '{simpler_query}'")
            search_results = ctx.web_search_service.search(simpler_query, num_results=5)

        if not search_results:
            logger.warning(f"No search results for: {search_query}")
            return []

        first_result = search_results[0]

        # 3. Fetch content
        fetched = ctx.url_fetcher_service.fetch_url(first_result.url)
        if not fetched or not fetched.success or not fetched.main_content:
            logger.warning(f"Failed to fetch URL: {first_result.url}")
            return []

        # 4. Extract claims and follow-up questions
        content = fetched.main_content
        analysis_prompt = f"""Analyze this content about "{topic}":

{content[:4000]}

Extract:
1. Main claims (factual statements)
2. Follow-up questions to investigate further

Return as JSON:
{{
  "summary": "brief summary",
  "claims": [
    {{"claim": "statement", "confidence": "high/medium/low"}}
  ],
  "follow_up_questions": ["question1", "question2"]
}}"""

        analysis_response = ctx.llm_service.generate_with_tools(
            messages=[{"role": "user", "content": analysis_prompt}],
            tools=None,
            temperature=0.3
        )

        analysis = json.loads(analysis_response["message"].content)

        # 5. Create SourceDoc
        from src.services.research_session import SourceDoc
        source_doc = SourceDoc(
            session_id=task.session_id,
            url=first_result.get("url"),
            title=first_result.get("title"),
            claims=analysis.get("claims", []),
            content_summary=analysis.get("summary")
        )
        ctx.session_store.create_source_doc(source_doc)

        # 6. Return follow-up question proposals
        follow_ups = analysis.get("follow_up_questions", [])
        proposals = []
        for question in follow_ups[:5]:
            proposals.append({
                "htn_task_type": "InvestigateQuestion",
                "args": {"question": question},
                "dedup_key": f"q::{question}",
                "depth": task.depth + 1
            })

        logger.info(f"InvestigateTopic: Created SourceDoc {source_doc.id}, proposing {len(proposals)} follow-ups")
        return proposals

    except Exception as e:
        logger.error(f"InvestigateTopic failed for '{topic}': {e}")
        return []


@method("InvestigateQuestion")
def investigate_question(task, ctx) -> List[Dict[str, Any]]:
    """
    Same as InvestigateTopic but keyed off free-form question.

    Args:
        task: Task with args {"question": str}
        ctx: ExecutionContext

    Returns:
        List of child proposals
    """
    # Reuse InvestigateTopic logic with question as topic
    from src.services.task_queue import Task
    task_copy = Task(
        id=task.id,
        session_id=task.session_id,
        htn_task_type=task.htn_task_type,
        args={"topic": task.args.get("question", "")},
        status=task.status,
        depth=task.depth,
        parent_id=task.parent_id,
        created_at=task.created_at,
        updated_at=task.updated_at
    )
    return investigate_topic(task_copy, ctx)


@method("SynthesizeFindings")
def synthesize_findings(task, ctx) -> List[Dict[str, Any]]:
    """
    Terminal HTN method - synthesizes all findings for a session.

    Loads all SourceDocs and tasks, asks LLM for global narrative,
    detects contested claims and unanswered questions, persists to session_summary.

    Args:
        task: Task with session_id
        ctx: ExecutionContext with llm_service, session_store, task_store

    Returns:
        Empty list (no child proposals - this is terminal)
    """
    session_store = ctx.session_store
    task_store = ctx.task_store

    # Load session
    sess = session_store.get_session(task.session_id)
    if not sess:
        logger.error(f"SynthesizeFindings: session {task.session_id} not found")
        return []

    # Load all research artifacts
    docs = session_store.load_source_docs_for_session(task.session_id)
    tasks = task_store.list_tasks_for_session(task.session_id)

    # Convert tasks to simple dicts for LLM
    task_dicts = [{
        "id": t.id,
        "htn_task_type": t.htn_task_type,
        "args": t.args,
        "status": t.status,
        "depth": t.depth,
    } for t in tasks]

    # Generate synthesis via LLM
    try:
        summary_obj = ctx.llm_service.summarize_research_session(
            root_question=sess.root_question,
            docs=docs,
            tasks=task_dicts,
        )

        # Persist summary
        session_store.save_session_summary(task.session_id, summary_obj)
        logger.info(f"SynthesizeFindings: Summary written for session {task.session_id}")

        # Log synthesis completion with stats
        get_multi_logger().log_research_event(
            event_type="synthesis_complete",
            session_id=task.session_id,
            data={
                "docs": len(docs),
                "claims": sum(len(d.get("claims", [])) for d in docs),
                "key_events": len(summary_obj.get("key_events", [])),
                "contested_claims": len(summary_obj.get("contested_claims", [])),
                "open_questions": len(summary_obj.get("open_questions", [])),
            }
        )

        # Propose belief updates (minimal scaffold)
        try:
            from src.services.research_to_belief_adapter import propose_updates, store_belief_updates
            updates = propose_updates(sess, summary_obj)
            store_belief_updates(updates)
            logger.info(f"SynthesizeFindings: Created {len(updates)} belief updates for session {task.session_id}")
        except Exception as belief_err:
            logger.warning(f"Belief update creation failed (non-fatal): {belief_err}")

        # Create research anchor for session reuse
        try:
            from src.services.research_anchor_store import create_anchor_from_session
            create_anchor_from_session(task.session_id, summary_obj)
            logger.info(f"SynthesizeFindings: Created research anchor for session {task.session_id}")
        except Exception as anchor_err:
            logger.warning(f"Research anchor creation failed (non-fatal): {anchor_err}")

    except Exception as e:
        logger.error(f"SynthesizeFindings failed: {e}")
        # Save error state
        session_store.save_session_summary(task.session_id, {
            "error": str(e),
            "narrative_summary": "Synthesis failed",
            "key_events": [],
            "contested_claims": [],
            "open_questions": []
        })

    return []  # Terminal method - no children
