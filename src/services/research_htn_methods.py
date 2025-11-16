"""HTN methods for autonomous research with session budgets."""

import json
import logging
import re
from typing import Any, Dict, List, Callable
from src.utils.logging_config import get_multi_logger

logger = logging.getLogger(__name__)

# JSON extraction regex
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

# Cold system prompt for research tool LLM (NO persona bleed)
RESEARCH_TOOL_SYSTEM = """You are a strict research microservice.

Rules:
- Output ONLY valid JSON matching the requested schema.
- No preamble, no explanations, no markdown.
- Do not mention being an AI or having feelings.
- Be deterministic and factual."""


def coerce_json_object(raw: str) -> Dict[str, Any]:
    """Extract and parse JSON object from potentially messy LLM output.

    Handles:
    - Pure JSON
    - JSON wrapped in markdown code blocks
    - JSON buried in prose
    """
    raw = raw.strip()

    # Fast path: raw is already valid JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Strip markdown code blocks
    if raw.startswith("```"):
        # Remove opening ```json or ```
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        # Remove closing ```
        raw = re.sub(r"\s*```$", "", raw)
        try:
            return json.loads(raw.strip())
        except Exception:
            pass

    # Try to extract first JSON object substring
    m = JSON_OBJECT_RE.search(raw)
    if not m:
        raise ValueError("No JSON object found in LLM output")

    snippet = m.group(0)
    return json.loads(snippet)


def coerce_json_array(raw: str) -> List[Any]:
    """Extract and parse JSON array from potentially messy LLM output."""
    raw = raw.strip()

    # Fast path: raw is already valid JSON
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        # If it's an object with a "topics" or similar key, extract that
        if isinstance(result, dict):
            for key in ["topics", "queries", "questions", "items"]:
                if key in result and isinstance(result[key], list):
                    return result[key]
        raise ValueError(f"Parsed JSON but got {type(result)}, not array")
    except Exception:
        pass

    # Strip markdown code blocks
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            return json.loads(raw.strip())
        except Exception:
            pass

    # Try to extract first JSON array substring
    m = JSON_ARRAY_RE.search(raw)
    if not m:
        raise ValueError("No JSON array found in LLM output")

    snippet = m.group(0)
    return json.loads(snippet)

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

    # Generate seed topics using cold tool LLM (no persona bleed)
    prompt = f"""Generate 3 to 5 short web search queries that would help investigate this question:

"{root_question}"

Output:
- Return ONLY a JSON array of strings.
- No explanation, no markdown, no extra text.
- Example: ["query one", "query two", "query three"]
""".strip()

    try:
        raw = ctx.llm_service.tool_call(
            system=RESEARCH_TOOL_SYSTEM,
            user=prompt,
            temperature=0.1,
            max_tokens=500
        )
        try:
            seed_topics = coerce_json_array(raw)
        except Exception as parse_err:
            logger.warning(f"Failed to parse seed topics as array, trying object: {parse_err}")
            data = coerce_json_object(raw)
            seed_topics = data.get("topics", [])
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
        # 1. Generate search query using cold tool LLM
        query_prompt = f"""Generate a web search query for: {topic}

Rules:
- Use 3-8 keyword phrases that a human would type into Google
- NO quotation marks around the entire query
- NO full sentences or natural language
- NO extra explanation
- Just keywords, separated by spaces

Query patterns:
- For scientific/technical topics: Add "study", "research", "paper", "measurement", "data"
  Example: "PETG PLA emissions study VOC measurement"
- For policy/economics: Add official sources like "EPI", "BLS", "CBO", "IPCC"
  Example: "wage stagnation productivity EPI data"
- For current events: Add year and official sources
  Example: "Elon Musk DOGE government efficiency 2024"

Query:"""
        raw_query = ctx.llm_service.tool_call(
            system=RESEARCH_TOOL_SYSTEM,
            user=query_prompt,
            temperature=0.3,
            max_tokens=100
        ).strip()

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

        # 4. Extract claims and follow-up questions using cold tool LLM
        content = fetched.main_content
        analysis_prompt = f"""Task:
- Read the following article excerpt about "{topic}".
- Extract:
  - "claims": a list of concise factual claims made in the text.
    CRITICAL: For each claim, preserve ALL concrete details:
    * Numerical values (measurements, percentages, counts, amounts)
    * Specific names (chemicals, products, institutions, studies, authors)
    * Dates and time periods
    * Experimental conditions (temperature, materials, setups)
    * Comparisons between entities (X vs Y showed A vs B)

    Examples of GOOD claims:
    - "PLA emits lactide at 210°C, measured at 50 μg/m³ (Smith et al 2020)"
    - "PETG showed 2x higher UFP count than PLA in enclosed printer tests"
    - "ABS releases styrene (suspected carcinogen) at typical print temps"

    Examples of BAD claims (too vague):
    - "PLA releases VOCs" (missing: WHICH VOCs, at what levels?)
    - "PETG is safer than ABS" (missing: quantitative comparison)
    - "Studies show health risks" (missing: WHICH studies, what risks?)

  - "summary": a 2-4 sentence neutral summary.
  - "follow_up_questions": 2-4 concrete questions that would advance understanding.

Output:
- Return ONLY a single JSON object.
- Do NOT include any explanation or markdown.
- JSON keys: "claims", "summary", "follow_up_questions".

Text:
{content[:4000]}
""".strip()

        raw_analysis = ctx.llm_service.tool_call(
            system=RESEARCH_TOOL_SYSTEM,
            user=analysis_prompt,
            temperature=0.0,
            max_tokens=1500
        )

        try:
            analysis = coerce_json_object(raw_analysis)
        except Exception as e:
            logger.error(f"InvestigateTopic JSON parse failed for '{topic}': {e}")
            logger.debug(f"Raw LLM response: {raw_analysis[:500]}")
            return []

        # 5. Create SourceDoc
        from src.services.research_session import SourceDoc
        source_doc = SourceDoc(
            session_id=task.session_id,
            url=first_result.url,
            title=first_result.title,
            claims=analysis.get("claims", []),
            content_summary=analysis.get("summary")
        )
        ctx.session_store.create_source_doc(source_doc)

        # 6. Filter follow-up questions for topic relevance
        # Prevent topic drift by only accepting questions related to root question
        from src.services.research_session import ResearchSessionStore
        sess_store = ResearchSessionStore()
        session = sess_store.get_session(task.session_id)
        root_question = session.root_question if session else ""

        follow_ups = analysis.get("follow_up_questions", [])

        # Filter follow-ups for relevance to root question
        if root_question and follow_ups:
            relevance_prompt = f"""Root question: "{root_question}"

Candidate follow-up questions:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(follow_ups))}

Task: Filter out questions that drift too far from the root question topic.
- Keep questions that directly investigate aspects of the root question
- Reject questions about tangential concerns, risks, or side topics

Return JSON array of question numbers to KEEP (e.g., [1, 3, 5]).
If all questions are relevant, return all numbers.
If all drift off-topic, return empty array []."""

            try:
                raw_filter = ctx.llm_service.tool_call(
                    system=RESEARCH_TOOL_SYSTEM,
                    user=relevance_prompt,
                    temperature=0.0,
                    max_tokens=100
                )
                keep_indices = coerce_json_array(raw_filter)
                filtered_questions = [follow_ups[i-1] for i in keep_indices if 0 < i <= len(follow_ups)]
                logger.info(f"InvestigateTopic: Filtered {len(follow_ups)} → {len(filtered_questions)} questions (topic relevance)")
                follow_ups = filtered_questions
            except Exception as e:
                logger.warning(f"InvestigateTopic: Follow-up filtering failed, using all: {e}")

        # 7. Return follow-up question proposals
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

    # Guard rail: Don't create beliefs or anchors for empty research
    if not docs:
        logger.warning(
            f"SynthesizeFindings: No docs for session {task.session_id}, "
            "skipping belief updates and anchor creation."
        )
        # Still log synthesis for observability
        get_multi_logger().log_research_event(
            event_type="synthesis_complete",
            session_id=task.session_id,
            data={
                "docs": 0,
                "claims": 0,
                "key_events": 0,
                "contested_claims": 0,
                "open_questions": 0,
            }
        )
        # Save minimal summary
        session_store.save_session_summary(task.session_id, {
            "narrative_summary": "No sources found for this question",
            "key_events": [],
            "contested_claims": [],
            "open_questions": [],
            "coverage_stats": {"total_docs": 0}
        })
        return []

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

        # Integrate with memory: reflection → experience → curiosity queue
        try:
            from src.services.research_memory_integration import integrate_research_with_memory
            integration_result = integrate_research_with_memory(
                session_id=task.session_id,
                root_question=sess.root_question,
                synthesis=summary_obj,
                llm_service=ctx.llm_service,
                embedding_service=None  # Will use default embedding service
            )
            logger.info(
                f"SynthesizeFindings: Memory integration complete - "
                f"experience_id={integration_result['experience_id']}, "
                f"questions_queued={integration_result['questions_queued']}"
            )
        except Exception as memory_err:
            logger.warning(f"Memory integration failed (non-fatal): {memory_err}")

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
