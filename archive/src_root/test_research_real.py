"""End-to-end test with real services (LLM, web search, URL fetcher)."""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.llm import create_llm_service
from src.services.web_search_service import create_web_search_service
from src.services.url_fetcher_service import create_url_fetcher_service
from src.tools.research_tools import research_and_summarize
from src.config import get_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run real-world research test."""

    settings = get_settings()

    # Initialize real services
    logger.info("Initializing services...")

    llm_service = create_llm_service(
        api_key=settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.7,
        max_tokens=4000
    )

    web_search_service = create_web_search_service()
    url_fetcher_service = create_url_fetcher_service()

    if not web_search_service.is_available():
        logger.error("Web search service not available")
        return

    logger.info("Services initialized ✓")

    # Test question
    question = "What is actually going on with the Epstein files story?"

    logger.info(f"\n{'='*60}")
    logger.info(f"RESEARCH QUESTION: {question}")
    logger.info(f"{'='*60}\n")

    # Run research
    try:
        summary = research_and_summarize(
            question=question,
            max_tasks=15,  # Keep it manageable for first real test
            max_children_per_task=2,
            max_depth=3,
            llm_service=llm_service,
            web_search_service=web_search_service,
            url_fetcher_service=url_fetcher_service
        )

        # Print results
        print("\n" + "="*60)
        print("SYNTHESIS RESULTS")
        print("="*60)

        print(f"\nSession ID: {summary.get('session_id')}")

        print(f"\nNarrative Summary:")
        print(f"  {summary.get('narrative_summary', 'N/A')}")

        print(f"\nKey Events ({len(summary.get('key_events', []))}):")
        for event in summary.get("key_events", []):
            print(f"  • {event}")

        print(f"\nContested Claims ({len(summary.get('contested_claims', []))}):")
        for claim in summary.get("contested_claims", []):
            print(f"  • {claim.get('claim')}")
            print(f"    Reason: {claim.get('reason')}")
            if claim.get('sources'):
                print(f"    Sources: {len(claim.get('sources'))} URLs")

        print(f"\nOpen Questions ({len(summary.get('open_questions', []))}):")
        for q in summary.get("open_questions", []):
            print(f"  • {q}")

        stats = summary.get("coverage_stats", {})
        print(f"\nCoverage Stats:")
        print(f"  Sources Investigated: {stats.get('sources_investigated', 0)}")
        print(f"  Claims Extracted: {stats.get('claims_extracted', 0)}")
        print(f"  Tasks Executed: {stats.get('tasks_executed', 0)}")
        print(f"  Max Depth Reached: {stats.get('depth_reached', 0)}")

        print("\n" + "="*60)

        # Analysis questions
        print("\nANALYSIS:")
        print("1. Which domains were hit?")
        print("   (Check source_docs table for URLs)")
        print("2. Are contested claims meaningful?")
        print("   (Review above)")
        print("3. Are open questions good research prompts?")
        print("   (Review above)")
        print("\nThis tells us where P2 (dedup, drift, source quality) needs work.")
        print("="*60)

    except Exception as e:
        logger.error(f"Research failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
