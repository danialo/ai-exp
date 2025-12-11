"""Automated policy violation detector for research benchmark results.

Analyzes benchmark results JSON and flags policy violations:
- Missing check_recent_research before research_and_summarize
- Research called for background knowledge (not current events)
- No research for obvious current events
- High risk with low hedging (or vice versa)
"""

import sys
import json
from typing import Dict, Any, List


def detect_tool_pattern_violations(result: Dict[str, Any]) -> List[str]:
    """Detect violations in tool call patterns.

    Args:
        result: Single benchmark result dict

    Returns:
        List of violation strings
    """
    violations = []
    tools = result.get("tools", [])
    tool_names = [t["tool"] for t in tools]

    # Check if research was called
    has_research = "research_and_summarize" in tool_names
    has_check = "check_recent_research" in tool_names

    if has_research:
        # Should have check_recent_research before research_and_summarize
        if not has_check:
            violations.append("MISSING_CHECK: research_and_summarize called without check_recent_research first")

        # Check ordering
        if has_check:
            check_idx = tool_names.index("check_recent_research")
            research_idx = tool_names.index("research_and_summarize")
            if check_idx > research_idx:
                violations.append("WRONG_ORDER: check_recent_research called AFTER research_and_summarize")

    return violations


def detect_risk_hedging_mismatch(result: Dict[str, Any]) -> List[str]:
    """Detect mismatch between risk level and hedging in answer.

    Args:
        result: Single benchmark result dict

    Returns:
        List of violation strings
    """
    violations = []

    meta = result.get("metadata", {})
    risk_level = meta.get("research_risk_level")
    answer = result.get("answer", "")

    if not risk_level:
        return []  # No research was done, skip

    # Hedging markers
    high_hedging_markers = ["early reports", "some sources claim", "unclear whether", "known so far", "developing", "emerging"]
    medium_hedging_markers = ["according to most", "majority view", "some dispute", "details disputed"]
    low_hedging_markers = ["multiple sources agree", "confirmed", "established", "documented"]

    answer_lower = answer.lower()

    # Count hedging markers
    high_hedge_count = sum(1 for m in high_hedging_markers if m in answer_lower)
    medium_hedge_count = sum(1 for m in medium_hedging_markers if m in answer_lower)
    low_hedge_count = sum(1 for m in low_hedging_markers if m in answer_lower)

    # Detect mismatches
    if risk_level == "high":
        if high_hedge_count == 0 and low_hedge_count > 0:
            violations.append("HEDGING_MISMATCH: High risk but answer uses assertive language")

    elif risk_level == "low":
        if high_hedge_count > 0:
            violations.append("HEDGING_MISMATCH: Low risk but answer uses excessive hedging")

    elif risk_level == "medium":
        if high_hedge_count > 2:
            violations.append("HEDGING_MISMATCH: Medium risk but answer uses very heavy hedging")

    return violations


def detect_current_events_violations(result: Dict[str, Any]) -> List[str]:
    """Detect if obvious current events question didn't trigger research.

    Args:
        result: Single benchmark result dict

    Returns:
        List of violation strings
    """
    violations = []

    question = result.get("question", "")
    question_lower = question.lower()
    tools = result.get("tools", [])
    tool_names = [t["tool"] for t in tools]

    # Current events markers
    current_events_markers = [
        "what happened",
        "what's happening",
        "what is going on",
        "current",
        "this week",
        "recent",
        "latest",
        "ongoing"
    ]

    is_current_events = any(m in question_lower for m in current_events_markers)

    if is_current_events:
        if "research_and_summarize" not in tool_names:
            violations.append("MISSING_RESEARCH: Obvious current events question did not trigger research")

    return violations


def analyze_results(results_file: str) -> None:
    """Analyze benchmark results and print policy violations.

    Args:
        results_file: Path to JSON results file
    """
    with open(results_file, "r") as f:
        results = json.load(f)

    print("=" * 80)
    print(f"POLICY VIOLATION ANALYSIS: {results_file}")
    print("=" * 80)
    print()

    all_violations = []

    for i, result in enumerate(results, 1):
        question = result.get("question", "")
        print(f"Q{i}: {question}")
        print("-" * 80)

        violations = []
        violations.extend(detect_tool_pattern_violations(result))
        violations.extend(detect_risk_hedging_mismatch(result))
        violations.extend(detect_current_events_violations(result))

        if violations:
            for v in violations:
                print(f"  ⚠️  {v}")
                all_violations.append((i, question, v))
        else:
            print(f"  ✓ No policy violations detected")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total questions: {len(results)}")
    print(f"Total violations: {len(all_violations)}")
    print()

    if all_violations:
        print("VIOLATIONS BY TYPE:")
        violation_types = {}
        for _, _, v in all_violations:
            vtype = v.split(":")[0]
            violation_types[vtype] = violation_types.get(vtype, 0) + 1

        for vtype, count in sorted(violation_types.items(), key=lambda x: -x[1]):
            print(f"  {vtype}: {count}")
        print()

        print("P2 PRIORITIES (based on violations):")
        if violation_types.get("HEDGING_MISMATCH", 0) > 0:
            print("  1. Improve risk assessment or hedging guidance")
        if violation_types.get("MISSING_CHECK", 0) > 0:
            print("  2. Enforce check_recent_research in tool selection prompt")
        if violation_types.get("MISSING_RESEARCH", 0) > 0:
            print("  3. Improve current events detection in policy layer")
    else:
        print("✅ NO VIOLATIONS DETECTED - Policy layer is working correctly!")

    print()
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.analyze_benchmark_results <results_file.json>")
        print()
        print("Example:")
        print("  python -m src.analyze_benchmark_results data/research_benchmark_astra_results_20251115_143022.json")
        sys.exit(1)

    results_file = sys.argv[1]
    analyze_results(results_file)


if __name__ == "__main__":
    main()
