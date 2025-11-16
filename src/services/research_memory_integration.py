"""Integration between research system and experience memory.

After research completes, this module:
1. Generates personal reflection ("What did I learn? How does this relate to me?")
2. Creates experience records with embeddings for semantic retrieval
3. Extracts follow-up questions for curiosity queue
"""

import logging
import time
from typing import Dict, Any, List, Optional
from uuid import uuid4

from src.memory.models import ExperienceType, Actor, CaptureMethod
from src.memory.raw_store import RawStore

logger = logging.getLogger(__name__)


def generate_research_reflection(
    llm_service,
    root_question: str,
    synthesis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate personal reflection on research findings.

    Asks: "What did I learn? How does this relate to my existing understanding?
    What new questions emerge?"

    Args:
        llm_service: LLM service for generation
        root_question: Original research question
        synthesis: Research synthesis dict

    Returns:
        {
            "what_i_learned": str,
            "how_this_relates_to_me": str,
            "new_insights": List[str],
            "follow_up_questions": List[str]
        }
    """
    key_events = synthesis.get("key_events", [])
    contested_claims = synthesis.get("contested_claims", [])
    open_questions = synthesis.get("open_questions", [])
    narrative = synthesis.get("narrative_summary", "")

    # Build reflection prompt
    prompt = f"""You are Astra reflecting on a completed research session.

You investigated: "{root_question}"

Here's what you found:

**Summary**: {narrative}

**Key Events**:
{chr(10).join(f"• {e}" for e in key_events[:8])}

**Contested Points**:
{chr(10).join(f"• {c}" for c in contested_claims[:5]) if contested_claims else "None"}

**Open Questions**:
{chr(10).join(f"• {q}" for q in open_questions[:5]) if open_questions else "None"}

---

Write a single JSON object with the following keys:

- "what_i_learned": 1-3 sentences summarizing your new understanding
- "connections": 1-2 sentences relating this to what you likely already knew or believed
- "insights": 2-4 concise bullet points (as strings), each one a new or sharpened insight
- "belief_implications": 1-2 sentences about how this should update, reinforce, or soften your existing beliefs (focus on direction, not absolute certainty)
- "follow_up_questions": 2-5 concrete questions you would like to investigate next time

Constraints:
- Return ONLY valid JSON. No markdown, no commentary.
- Each field should be short and information-dense.
- Be specific and concrete, not vague.""".strip()

    try:
        response = llm_service.generate_with_tools(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.3,
            max_tokens=1500  # Cap reflection output
        )

        raw = response["message"].content

        # Parse JSON (reuse coercion from research_htn_methods)
        from src.services.research_htn_methods import coerce_json_object
        reflection = coerce_json_object(raw)

        logger.info(f"Generated research reflection for '{root_question}': {len(reflection.get('follow_up_questions', []))} follow-up questions")

        return reflection

    except Exception as e:
        logger.error(f"Failed to generate research reflection: {e}")
        # Return minimal reflection
        return {
            "what_i_learned": f"Researched {root_question}",
            "how_this_relates_to_me": "Added to my knowledge base",
            "new_insights": [],
            "follow_up_questions": []
        }


def create_research_memory(
    session_id: str,
    root_question: str,
    synthesis: Dict[str, Any],
    reflection: Dict[str, Any],
    embedding_service=None
) -> str:
    """
    Create experience memory record for completed research.

    Args:
        session_id: Research session UUID
        root_question: Original question
        synthesis: Research synthesis dict
        reflection: Personal reflection dict
        embedding_service: Optional embedding service for semantic search

    Returns:
        experience_id: UUID of created experience
    """
    from datetime import datetime, timezone
    from src.memory.models import (
        ExperienceModel, ExperienceType, Actor, CaptureMethod,
        ContentModel, ProvenanceModel, EmbeddingPointers
    )

    # Build memory content
    memory_content = f"""Research on: {root_question}

What I learned: {reflection.get('what_i_learned', '')}

Connections: {reflection.get('connections', '')}

New insights:
{chr(10).join(f"• {i}" for i in reflection.get('insights', []))}

Belief implications: {reflection.get('belief_implications', '')}

Key findings from {synthesis.get('coverage_stats', {}).get('total_docs', 0)} sources:
{chr(10).join(f"• {e}" for e in synthesis.get('key_events', [])[:5])}
""".strip()

    # Create experience ID
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    experience_id = f"exp_research_{timestamp}_{session_id[:8]}"

    try:
        store = RawStore(db_path="data/raw_store.db")

        # TODO: Add embedding support when embedding service is wired up
        # For now, skip embeddings (not critical for initial functionality)
        embedding_ptr = None

        # Create structured metadata
        structured_metadata = {
            "research_session_id": session_id,
            "sources_count": synthesis.get("coverage_stats", {}).get("total_docs", 0),
            "contested_claims_count": len(synthesis.get("contested_claims", [])),
            "follow_up_questions": reflection.get("follow_up_questions", []),
            "belief_implications": reflection.get("belief_implications", "")
        }

        # Create experience model
        experience = ExperienceModel(
            id=experience_id,
            type=ExperienceType.LEARNING_PATTERN,
            created_at=datetime.now(timezone.utc),
            content=ContentModel(
                text=memory_content,
                structured=structured_metadata
            ),
            provenance=ProvenanceModel(
                actor=Actor.AGENT,
                method=CaptureMethod.MODEL_INFER
            ),
            embeddings=EmbeddingPointers(
                semantic=embedding_ptr
            ),
            ownership=Actor.AGENT
        )

        # Persist to raw store
        store.append_experience(experience)

        logger.info(f"Created research memory {experience_id} for session {session_id}")
        return experience_id

    except Exception as e:
        logger.error(f"Failed to create research memory: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return experience_id


def queue_follow_up_questions(
    session_id: str,
    reflection: Dict[str, Any],
    priority: int = 5
) -> int:
    """
    Queue follow-up questions for curiosity-driven exploration.

    Args:
        session_id: Research session UUID
        reflection: Reflection dict with follow_up_questions
        priority: Priority level (1-10, default 5)

    Returns:
        Number of questions queued
    """
    from src.services.curiosity_queue import CuriosityQueue

    questions = reflection.get("follow_up_questions", [])
    if not questions:
        return 0

    try:
        queue = CuriosityQueue()

        for q in questions:
            queue.enqueue(
                question=q,
                source="research_reflection",
                source_id=session_id,
                priority=priority,
                metadata={
                    "original_research": reflection.get("what_i_learned", "")[:200]
                }
            )

        logger.info(f"Queued {len(questions)} follow-up questions from research {session_id}")
        return len(questions)

    except Exception as e:
        logger.error(f"Failed to queue follow-up questions: {e}")
        return 0


def integrate_research_with_memory(
    session_id: str,
    root_question: str,
    synthesis: Dict[str, Any],
    llm_service,
    embedding_service=None
) -> Dict[str, Any]:
    """
    Complete integration: reflection → memory → question queue.

    This is the main entry point called after research synthesis completes.

    Args:
        session_id: Research session UUID
        root_question: Original research question
        synthesis: Synthesis dict from research_and_summarize
        llm_service: LLM service
        embedding_service: Optional embedding service

    Returns:
        {
            "reflection": dict,
            "experience_id": str,
            "questions_queued": int
        }
    """
    # 1. Generate reflection
    reflection = generate_research_reflection(
        llm_service=llm_service,
        root_question=root_question,
        synthesis=synthesis
    )

    # 2. Create memory with embedding
    experience_id = create_research_memory(
        session_id=session_id,
        root_question=root_question,
        synthesis=synthesis,
        reflection=reflection,
        embedding_service=embedding_service
    )

    # 3. Queue follow-up questions
    questions_queued = queue_follow_up_questions(
        session_id=session_id,
        reflection=reflection,
        priority=5
    )

    return {
        "reflection": reflection,
        "experience_id": experience_id,
        "questions_queued": questions_queued
    }
