"""Test runner for research HTN system with stub services."""

from dataclasses import dataclass
from typing import Any, List, Dict
import logging

from src.services.task_queue import TaskStore
from src.services.htn_task_executor import HTNTaskExecutor, ExecutionContext
from src.services.research_session import ResearchSessionStore
from src.tools.research_tools import start_research_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# === Stub Services for Testing ===

class StubLLMService:
    """Stub LLM that returns canned responses matching real LLM format."""

    def generate_with_tools(self, messages, tools=None, temperature=0.5):
        prompt = messages[0]["content"]

        # Detect prompt type and return appropriate JSON
        if "Generate 3-5 specific topics" in prompt:
            content = '{"topics": ["AI safety developments", "Tech layoffs 2025", "Climate policy updates"]}'

        elif "Generate a concise web search query" in prompt:
            content = "AI safety recent developments"

        elif "Analyze this content" in prompt:
            content = '''{
  "summary": "Article discusses recent AI safety research",
  "claims": [
    {"claim": "New alignment techniques show 30% improvement", "confidence": "medium"},
    {"claim": "Several labs committed to safety standards", "confidence": "high"}
  ],
  "follow_up_questions": ["What specific alignment techniques were developed?", "Which labs made commitments?"]
}'''

        else:
            content = "{}"

        # Return dict matching real LLM service format
        return {
            "message": type('Message', (), {"content": content})()
        }

    def summarize_research_session(self, root_question: str, docs: list, tasks: list) -> dict:
        """Generate session synthesis summary."""
        logger.info(f"Synthesizing {len(docs)} docs and {len(tasks)} tasks for question: {root_question}")

        # Extract all claims
        all_claims = []
        for doc in docs:
            for claim in doc.get("claims", []):
                all_claims.append(claim)

        # Simple synthesis
        return {
            "narrative_summary": f"Research on '{root_question}' yielded {len(docs)} sources with {len(all_claims)} total claims. "
                                f"Key findings include alignment technique improvements and industry commitments to safety standards. "
                                f"Investigation covered {len(tasks)} tasks across multiple depth levels.",
            "key_events": [
                "New alignment techniques showing measurable improvements",
                "Multiple AI labs committing to safety protocols",
                "Increased focus on safety benchmarking"
            ],
            "contested_claims": [
                {
                    "claim": "30% improvement in alignment techniques",
                    "reason": "Multiple sources report different improvement percentages",
                    "sources": [doc.get("url") for doc in docs[:2]]
                }
            ],
            "open_questions": [
                "What are the specific methodologies behind alignment improvements?",
                "How are safety commitments being measured and enforced?",
                "What gaps remain in current safety approaches?"
            ],
            "coverage_stats": {
                "sources_investigated": len(docs),
                "claims_extracted": len(all_claims),
                "tasks_executed": len(tasks),
                "depth_reached": max([t.get("depth", 0) for t in tasks]) if tasks else 0
            }
        }


class StubWebSearchService:
    """Stub search that returns fake results."""

    def search(self, query: str, num_results: int = 1) -> List[Dict[str, Any]]:
        logger.info(f"StubWebSearch: Searching for '{query}'")
        return [
            {
                "url": f"https://example.com/article/{hash(query) % 1000}",
                "title": f"Article about {query}",
                "snippet": "Example snippet..."
            }
        ]


class StubURLFetcherService:
    """Stub fetcher that returns fake content."""

    def fetch(self, url: str) -> str:
        logger.info(f"StubURLFetcher: Fetching {url}")
        return f"""Example article content from {url}.

This is a test article about AI safety developments. Recent research shows significant
progress in alignment techniques, with several major labs committing to new safety
standards and protocols. The field is evolving rapidly with both technical and policy
developments occurring in parallel.

Key findings include improved interpretability methods, better red-teaming practices,
and stronger coordination between research institutions."""


# === Main Test Runner ===

def main():
    """Run research HTN test with stub services."""

    logger.info("=== Starting Research HTN Test ===")

    # Create session
    session_id = start_research_session(
        question="What are the latest developments in AI safety?",
        max_tasks=10,
        max_children_per_task=2,
        max_depth=3
    )
    logger.info(f"Created research session: {session_id}")

    # Build execution context
    exec_ctx = ExecutionContext(
        llm_service=StubLLMService(),
        web_search_service=StubWebSearchService(),
        url_fetcher_service=StubURLFetcherService(),
        session_store=ResearchSessionStore(),
        task_store=TaskStore()
    )

    # Run executor
    executor = HTNTaskExecutor(
        task_store=TaskStore(),
        session_store=ResearchSessionStore(),
        ctx=exec_ctx
    )

    logger.info("Starting task execution...")
    executor.run_until_empty(session_id=session_id)

    logger.info("=== Execution Complete ===")

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    sess_store = ResearchSessionStore()
    session = sess_store.get_session(session_id)
    print(f"\nSession Status: {session.status}")
    print(f"Tasks Created: {session.tasks_created} / {session.max_tasks}")

    # Print source docs
    source_docs = sess_store.list_source_docs(session_id)
    print(f"\nSource Documents Found: {len(source_docs)}")
    for i, doc in enumerate(source_docs, 1):
        print(f"\n{i}. {doc.title}")
        print(f"   URL: {doc.url}")
        print(f"   Summary: {doc.content_summary}")
        print(f"   Claims: {len(doc.claims)}")
        for claim in doc.claims:
            print(f"     - {claim.get('claim')} ({claim.get('confidence')})")

    # Print task statistics
    task_store = TaskStore()
    print(f"\nTask Queue Statistics:")
    print(f"  Queued: {task_store.queued_count(session_id)}")

    # Print synthesis summary
    if session.session_summary:
        print(f"\n{'='*60}")
        print("SESSION SYNTHESIS")
        print("="*60)
        summary = session.session_summary
        print(f"\nNarrative Summary:")
        print(f"  {summary.get('narrative_summary', 'N/A')}")

        print(f"\nKey Events ({len(summary.get('key_events', []))}):")
        for event in summary.get("key_events", []):
            print(f"  • {event}")

        print(f"\nContested Claims ({len(summary.get('contested_claims', []))}):")
        for claim in summary.get("contested_claims", []):
            print(f"  • {claim.get('claim')}")
            print(f"    Reason: {claim.get('reason')}")

        print(f"\nOpen Questions ({len(summary.get('open_questions', []))}):")
        for question in summary.get("open_questions", []):
            print(f"  • {question}")

        stats = summary.get("coverage_stats", {})
        print(f"\nCoverage Stats:")
        print(f"  Sources: {stats.get('sources_investigated', 0)}")
        print(f"  Claims: {stats.get('claims_extracted', 0)}")
        print(f"  Tasks: {stats.get('tasks_executed', 0)}")
        print(f"  Max Depth: {stats.get('depth_reached', 0)}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
