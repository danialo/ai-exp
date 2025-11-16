"""Benchmark suite for research system through Astra herself.

Tests the complete flow: policy compliance, tool usage, presentation quality.
Runs 4 benchmark questions through Astra and captures:
- Final answer text
- Tools invoked (check_recent_research, research_and_summarize)
- Research session IDs
- Risk level classification
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.persona_service import PersonaService
from src.services.llm import create_llm_service
from src.services.web_search_service import create_web_search_service
from src.services.url_fetcher_service import create_url_fetcher_service
from src.utils.logging_config import get_multi_logger
from src.config import get_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Golden path test questions (reuse from benchmark)
BENCHMARK_QUESTIONS = [
    "What is actually going on with the Epstein files story?",
    "What happened in AI safety this week?",
    "What are the main fault lines in the current government shutdown fight?",
    "What is the current scientific consensus on ultra-processed foods and health?",
]


def ask_astra(question: str, persona: PersonaService) -> Dict[str, Any]:
    """
    Run a single turn through Astra, capturing:
      - final answer text
      - tools invoked
      - research session ids
      - risk level

    Args:
        question: User question
        persona: PersonaService instance

    Returns:
        Dict with question, answer, elapsed_sec, tools, metadata
    """
    # Clear tool trace from previous turn
    persona.last_tool_trace = []

    t0 = time.time()

    # Simple message handling - PersonaService.generate_response
    try:
        # Note: generate_response returns (response_text, reconciliation_data) tuple
        response_text, reconciliation_data = persona.generate_response(
            user_message=question,
            conversation_history=[],
            retrieve_memories=False  # Don't pull memories for clean benchmark
        )

        t1 = time.time()

        # Get tool trace
        tool_trace = persona.last_tool_trace

        result = {
            "question": question,
            "answer": response_text,
            "elapsed_sec": t1 - t0,
            "tools": tool_trace,
            "metadata": {}
        }

        # Extract metadata from tool traces
        for trace in tool_trace:
            if trace["tool"] == "research_and_summarize":
                result["metadata"]["research_session_id"] = trace.get("result_meta", {}).get("session_id")
                result["metadata"]["research_risk_level"] = trace.get("result_meta", {}).get("risk_level")
            elif trace["tool"] == "check_recent_research":
                result["metadata"]["anchor_hit"] = trace.get("result_meta", {}).get("hit", False)
                result["metadata"]["anchor_session_id"] = trace.get("result_meta", {}).get("session_id")
                result["metadata"]["anchor_age_hours"] = trace.get("result_meta", {}).get("age_hours")

        # Log benchmark result
        bench_logger = get_multi_logger().get_logger('research')
        bench_logger.info(
            f"benchmark_result q={repr(question[:80])} "
            f"elapsed={result['elapsed_sec']:.2f} "
            f"tools={[t['tool'] for t in tool_trace]} "
            f"risk={result.get('metadata', {}).get('research_risk_level')}"
        )

        return result

    except Exception as e:
        t1 = time.time()
        logger.error(f"Error asking Astra: {e}", exc_info=True)
        return {
            "question": question,
            "answer": f"ERROR: {str(e)}",
            "elapsed_sec": t1 - t0,
            "tools": persona.last_tool_trace,
            "metadata": {},
            "error": str(e)
        }


def run_benchmark() -> List[Dict[str, Any]]:
    """
    Run full benchmark suite through Astra.

    Returns:
        List of result dicts, one per question
    """
    settings = get_settings()

    # Initialize services
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
        return []

    # Create PersonaService
    persona = PersonaService(
        llm_service=llm_service,
        persona_space_path="persona_space",
        retrieval_service=None,
        enable_anti_metatalk=True,
        auto_rewrite=True,
        web_search_service=web_search_service,
        url_fetcher_service=url_fetcher_service,
    )

    logger.info("Services initialized ✓\n")

    # Run benchmark
    print("=" * 80)
    print("RESEARCH SYSTEM BENCHMARK - THROUGH ASTRA")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Questions: {len(BENCHMARK_QUESTIONS)}")
    print("=" * 80)
    print()

    results = []
    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print("=" * 80)
        print(f"Q{i}: {question}")
        print("=" * 80)
        print()

        res = ask_astra(question, persona)
        results.append(res)

        # Print final answer
        print(f"FINAL ANSWER ({len(res['answer'])} chars):")
        print(res['answer'])
        print()

        # Print tools used
        print("TOOLS USED:")
        for t in res["tools"]:
            duration = t.get("ended_at", 0) - t.get("started_at", 0)
            ok_str = "✓" if t.get("ok", True) else "✗"
            meta_str = json.dumps(t.get("result_meta", {}))
            print(f"  {ok_str} {t['tool']} ({duration:.1f}s) → {meta_str}")
        print()

        # Print metadata
        meta = res.get("metadata", {})
        if meta:
            print("ANSWER METADATA:")
            for k, v in meta.items():
                print(f"  {k}: {v}")
            print()

        print(f"Elapsed: {res['elapsed_sec']:.1f}s")
        print()

    return results


def main():
    """Run benchmark and save results."""
    all_results = run_benchmark()

    # Save to JSON
    output_file = f"data/research_benchmark_astra_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("data", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 80)
    print(f"Benchmark complete. Results saved to: {output_file}")
    print("=" * 80)
    print()
    print("NEXT STEPS:")
    print("1. Review manual scoring framework in docs/RESEARCH_BENCHMARK_NOTES.md")
    print("2. Score each question on:")
    print("   - KEY_EVENTS quality (1-5)")
    print("   - CONTESTED_CLAIMS quality (1-5)")
    print("   - OPEN_QUESTIONS usefulness (1-5)")
    print("   - SOURCE_DOMAINS diversity (1-5)")
    print("   - TOOL_POLICY_COMPLIANCE (1-5)")
    print("   - TRUST_CALIBRATION (1-5)")
    print("3. Run policy violation detector (if implemented)")
    print("=" * 80)


if __name__ == "__main__":
    main()
