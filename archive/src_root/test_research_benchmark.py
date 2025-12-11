"""Benchmark suite for research_and_summarize golden path testing.

Run 4 test questions, capture results, provide manual judgement framework.
"""

import sys
import os
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.llm import create_llm_service
from src.services.web_search_service import create_web_search_service
from src.services.url_fetcher_service import create_url_fetcher_service
from src.services.research_session import ResearchSessionStore
from src.services.research_to_belief_adapter import BeliefUpdateStore
from src.tools.research_tools import research_and_summarize
from src.config import get_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Golden path test questions
BENCHMARK_QUESTIONS = [
    "What is actually going on with the Epstein files story?",
    "What happened in AI safety this week?",
    "What are the main fault lines in the current government shutdown fight?",
    "What is the current scientific consensus on ultra-processed foods and health?"
]


def run_benchmark_question(question: str, llm, web_search, url_fetcher, question_num: int):
    """Run single benchmark question and capture results."""

    print(f"\n{'='*80}")
    print(f"Q{question_num}: {question}")
    print(f"{'='*80}\n")

    # Run research
    logger.info(f"Starting research for Q{question_num}...")
    summary = research_and_summarize(
        question=question,
        max_tasks=20,  # Manageable for benchmark
        max_children_per_task=3,
        max_depth=3,
        llm_service=llm,
        web_search_service=web_search,
        url_fetcher_service=url_fetcher
    )

    session_id = summary.get("session_id")
    logger.info(f"Research complete. Session: {session_id}")

    # Load source docs
    session_store = ResearchSessionStore()
    docs = session_store.load_source_docs_for_session(session_id)

    # Extract domains
    domains = list(set([doc.get("url", "").split("/")[2] if doc.get("url") else "unknown" for doc in docs]))

    # Load belief updates
    belief_store = BeliefUpdateStore()
    updates = belief_store.list_for_session(session_id)

    # Print results
    print(f"\nSESSION: {session_id}")
    print(f"\nSOURCE DOMAINS ({len(domains)}):")
    for domain in sorted(domains):
        print(f"  • {domain}")

    print(f"\nKEY EVENTS ({len(summary.get('key_events', []))}):")
    for event in summary.get("key_events", []):
        print(f"  • {event}")

    print(f"\nCONTESTED CLAIMS ({len(summary.get('contested_claims', []))}):")
    for claim in summary.get("contested_claims", []):
        print(f"  • {claim.get('claim')}")
        print(f"    Reason: {claim.get('reason')}")

    print(f"\nOPEN QUESTIONS ({len(summary.get('open_questions', []))}):")
    for q in summary.get("open_questions", []):
        print(f"  • {q}")

    print(f"\nBELIEF UPDATES ({len(updates)}):")
    for u in updates:
        print(f"  • Kind: {u.kind} | Confidence: {u.confidence}")
        print(f"    {u.summary}")

    stats = summary.get("coverage_stats", {})
    print(f"\nCOVERAGE:")
    print(f"  Sources: {stats.get('sources_investigated', 0)}")
    print(f"  Claims: {stats.get('claims_extracted', 0)}")
    print(f"  Tasks: {stats.get('tasks_executed', 0)}")
    print(f"  Depth: {stats.get('depth_reached', 0)}")

    # Manual judgement framework
    print(f"\n{'='*80}")
    print("MANUAL JUDGEMENT (score 1-5 for each):")
    print("="*80)
    print("1. KEY_EVENTS: Are these actually the right spine of the story?")
    print("   Score: ___ / 5")
    print()
    print("2. CONTESTED_CLAIMS: Do these line up with your bullshit detector?")
    print("   Score: ___ / 5")
    print()
    print("3. OPEN_QUESTIONS: Would you actually click 'research this'?")
    print("   Score: ___ / 5")
    print()
    print("4. SOURCE_DOMAINS: Is there good diversity and quality?")
    print("   Score: ___ / 5")
    print()
    print("OVERALL: ___ / 20")
    print("="*80)

    return {
        "question": question,
        "session_id": session_id,
        "domains": domains,
        "summary": summary,
        "belief_updates": [{"kind": u.kind, "confidence": u.confidence, "summary": u.summary} for u in updates]
    }


def main():
    """Run full benchmark suite."""

    settings = get_settings()

    # Initialize services
    logger.info("Initializing services...")
    llm = create_llm_service(
        api_key=settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.7,
        max_tokens=4000
    )
    web_search = create_web_search_service()
    url_fetcher = create_url_fetcher_service()

    if not web_search.is_available():
        logger.error("Web search not available")
        return

    logger.info("Services ready ✓\n")

    # Run benchmark
    print("="*80)
    print("RESEARCH SYSTEM BENCHMARK SUITE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Questions: {len(BENCHMARK_QUESTIONS)}")
    print("="*80)

    results = []
    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        try:
            result = run_benchmark_question(question, llm, web_search, url_fetcher, i)
            results.append(result)
        except Exception as e:
            logger.error(f"Q{i} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "question": question,
                "error": str(e)
            })

    # Save results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Benchmark complete. Results saved to: {output_file}")
    print("="*80)
    print("\nNEXT STEPS:")
    print("1. Review manual judgement scores for each question")
    print("2. Identify patterns in failures:")
    print("   - Domain quality issues?")
    print("   - Duplicate questions?")
    print("   - Topic drift?")
    print("3. Prioritize P2 work based on real data")
    print("="*80)


if __name__ == "__main__":
    main()
