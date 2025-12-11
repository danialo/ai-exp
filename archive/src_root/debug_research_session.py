"""Debug script to inspect research session details.

Usage:
    python -m src.debug_research_session <session_id>

Displays:
- Session metadata (root question, status, budgets)
- Source documents with claims
- Task execution tree
- Synthesis summary
- Belief updates
"""

import sys
import json
from typing import Dict, Any, List

from src.services.research_session import ResearchSessionStore
from src.services.task_queue import TaskStore
from src.services.research_to_belief_adapter import BeliefUpdateStore


def format_task_tree(tasks: List[Dict], indent_level: int = 0) -> str:
    """Format tasks as tree structure based on parent_id."""
    indent = "  " * indent_level
    output = ""

    # Group by parent
    by_parent: Dict[str, List[Dict]] = {}
    for t in tasks:
        parent = t.get("parent_id") or "ROOT"
        if parent not in by_parent:
            by_parent[parent] = []
        by_parent[parent].append(t)

    # Print root tasks first
    root_tasks = by_parent.get("ROOT", [])
    for task in root_tasks:
        status_icon = {"queued": "⏸", "running": "▶", "done": "✓", "error": "✗"}.get(task["status"], "?")
        output += f"{indent}{status_icon} [{task['htn_task_type']}] {json.dumps(task.get('args', {}))}\n"

        # Recursively print children
        children = by_parent.get(task["id"], [])
        for child in children:
            child_status = {"queued": "⏸", "running": "▶", "done": "✓", "error": "✗"}.get(child["status"], "?")
            output += f"{indent}  {child_status} [{child['htn_task_type']}] {json.dumps(child.get('args', {}))}\n"

    return output


def inspect_session(session_id: str):
    """Inspect and print details of a research session."""
    session_store = ResearchSessionStore()
    task_store = TaskStore()
    belief_store = BeliefUpdateStore()

    # Load session
    session = session_store.get_session(session_id)
    if not session:
        print(f"❌ Session not found: {session_id}")
        return

    print("=" * 80)
    print(f"RESEARCH SESSION: {session_id}")
    print("=" * 80)

    # Session metadata
    print(f"\n📋 METADATA:")
    print(f"  Root Question: {session.root_question}")
    print(f"  Status: {session.status}")
    print(f"  Max Tasks: {session.max_tasks}")
    print(f"  Max Children/Task: {session.max_children_per_task}")
    print(f"  Max Depth: {session.max_depth}")
    print(f"  Tasks Created: {session.tasks_created}")

    # Source documents
    docs = session_store.load_source_docs_for_session(session_id)
    print(f"\n📄 SOURCE DOCUMENTS ({len(docs)}):")
    if docs:
        # Extract unique domains
        domains = set()
        for doc in docs:
            url = doc.get("url", "")
            if url:
                try:
                    domain = url.split("/")[2]
                    domains.add(domain)
                except IndexError:
                    pass

        print(f"  Unique Domains: {len(domains)}")
        for domain in sorted(domains):
            count = sum(1 for d in docs if domain in d.get("url", ""))
            print(f"    • {domain} ({count} doc{'s' if count > 1 else ''})")

        print(f"\n  Documents:")
        for doc in docs[:10]:  # Limit to first 10
            print(f"    • {doc.get('title', 'Untitled')}")
            print(f"      URL: {doc.get('url', 'N/A')}")
            claims = doc.get("claims", [])
            print(f"      Claims: {len(claims)}")
            if claims:
                for claim in claims[:3]:  # First 3 claims
                    print(f"        - {claim.get('claim', 'N/A')} (confidence: {claim.get('confidence', 'N/A')})")
        if len(docs) > 10:
            print(f"    ... and {len(docs) - 10} more")
    else:
        print("  (None)")

    # Task execution tree
    tasks_list = task_store.list_tasks_for_session(session_id)
    tasks_dicts = [{
        "id": t.id,
        "parent_id": t.parent_id,
        "htn_task_type": t.htn_task_type,
        "args": t.args,
        "status": t.status,
        "depth": t.depth
    } for t in tasks_list]

    print(f"\n🌳 TASK TREE ({len(tasks_dicts)} tasks):")
    if tasks_dicts:
        # Count by status
        status_counts = {}
        for t in tasks_dicts:
            status = t["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        print(f"  Status: {', '.join([f'{k}={v}' for k, v in status_counts.items()])}")
        print()
        print(format_task_tree(tasks_dicts))
    else:
        print("  (None)")

    # Synthesis summary
    if session.session_summary:
        summary = session.session_summary
        print(f"\n📊 SYNTHESIS SUMMARY:")

        if summary.get("narrative_summary"):
            print(f"\n  Narrative:")
            print(f"    {summary['narrative_summary']}")

        key_events = summary.get("key_events", [])
        if key_events:
            print(f"\n  Key Events ({len(key_events)}):")
            for event in key_events:
                print(f"    • {event}")

        contested = summary.get("contested_claims", [])
        if contested:
            print(f"\n  Contested Claims ({len(contested)}):")
            for claim in contested:
                print(f"    • {claim.get('claim')}")
                print(f"      Reason: {claim.get('reason')}")

        open_qs = summary.get("open_questions", [])
        if open_qs:
            print(f"\n  Open Questions ({len(open_qs)}):")
            for q in open_qs:
                print(f"    • {q}")

        stats = summary.get("coverage_stats", {})
        if stats:
            print(f"\n  Coverage Stats:")
            print(f"    Sources Investigated: {stats.get('sources_investigated', 0)}")
            print(f"    Claims Extracted: {stats.get('claims_extracted', 0)}")
            print(f"    Tasks Executed: {stats.get('tasks_executed', 0)}")
            print(f"    Depth Reached: {stats.get('depth_reached', 0)}")
    else:
        print(f"\n📊 SYNTHESIS SUMMARY: (Not yet synthesized)")

    # Belief updates
    updates = belief_store.list_for_session(session_id)
    print(f"\n💡 BELIEF UPDATES ({len(updates)}):")
    if updates:
        for u in updates:
            print(f"  • Kind: {u.kind} | Confidence: {u.confidence:.0%}")
            print(f"    {u.summary}")
    else:
        print("  (None)")

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.debug_research_session <session_id>")
        print("\nExample:")
        print("  python -m src.debug_research_session abc123-def456-ghi789")
        sys.exit(1)

    session_id = sys.argv[1]
    inspect_session(session_id)


if __name__ == "__main__":
    main()
