#!/usr/bin/env python3
"""Quick metrics from research system logs.

Parses logs/research/research_system.log and prints summary stats:
- Total sessions completed
- Average tasks per session
- Average docs per session
- Risk level distribution
- Average contested claims per session
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_log_line(line: str) -> Dict[str, Any]:
    """Parse structured log line into dict."""
    parsed = {}

    # Extract session_id
    session_match = re.search(r'session=(\S+)', line)
    if session_match:
        parsed['session_id'] = session_match.group(1)

    # Extract event type
    event_match = re.search(r'event=(\S+)', line)
    if event_match:
        parsed['event'] = event_match.group(1)

    # Extract all key=value pairs
    kv_pattern = r'(\w+)=([^\s]+)'
    for match in re.finditer(kv_pattern, line):
        key, value = match.groups()
        if key not in ['session', 'event']:
            # Try to parse as int/float
            try:
                if '.' in value:
                    parsed[key] = float(value)
                else:
                    parsed[key] = int(value)
            except ValueError:
                parsed[key] = value

    return parsed


def compute_metrics(log_path: str) -> Dict[str, Any]:
    """Compute summary metrics from log file."""

    sessions = defaultdict(lambda: {
        'tasks_created': 0,
        'docs': 0,
        'claims': 0,
        'key_events': 0,
        'contested_claims': 0,
        'open_questions': 0,
        'risk': None
    })

    task_counts = []
    doc_counts = []
    claim_counts = []
    risk_distribution = defaultdict(int)

    with open(log_path, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)

            if not parsed.get('event'):
                continue

            session_id = parsed.get('session_id')
            event = parsed['event']

            if event == 'session_complete':
                sessions[session_id]['tasks_created'] = parsed.get('tasks_created', 0)
                if parsed.get('tasks_created'):
                    task_counts.append(parsed['tasks_created'])

            elif event == 'synthesis_complete':
                sessions[session_id]['docs'] = parsed.get('docs', 0)
                sessions[session_id]['claims'] = parsed.get('claims', 0)
                sessions[session_id]['key_events'] = parsed.get('key_events', 0)
                sessions[session_id]['contested_claims'] = parsed.get('contested_claims', 0)
                sessions[session_id]['open_questions'] = parsed.get('open_questions', 0)

                if parsed.get('docs'):
                    doc_counts.append(parsed['docs'])
                if parsed.get('claims'):
                    claim_counts.append(parsed['claims'])

            elif event == 'benchmark_result':
                risk = parsed.get('risk')
                if risk and risk != 'None':
                    risk_distribution[risk] += 1

    # Compute averages
    metrics = {
        'sessions_completed': len([s for s in sessions.values() if s['tasks_created'] > 0]),
        'avg_tasks_per_session': sum(task_counts) / len(task_counts) if task_counts else 0,
        'avg_docs_per_session': sum(doc_counts) / len(doc_counts) if doc_counts else 0,
        'avg_claims_per_session': sum(claim_counts) / len(claim_counts) if claim_counts else 0,
        'risk_distribution': dict(risk_distribution),
        'total_benchmark_runs': sum(risk_distribution.values())
    }

    # Add per-session breakdown if available
    if sessions:
        contested = [s['contested_claims'] for s in sessions.values() if s['contested_claims'] > 0]
        if contested:
            metrics['avg_contested_claims'] = sum(contested) / len(contested)

    return metrics


def print_metrics(metrics: Dict[str, Any]):
    """Pretty-print metrics table."""
    print("=" * 60)
    print("RESEARCH SYSTEM METRICS")
    print("=" * 60)
    print()

    print(f"Sessions completed:        {metrics['sessions_completed']}")
    print(f"Avg tasks per session:     {metrics['avg_tasks_per_session']:.1f}")
    print(f"Avg docs per session:      {metrics['avg_docs_per_session']:.1f}")
    print(f"Avg claims per session:    {metrics['avg_claims_per_session']:.1f}")

    if 'avg_contested_claims' in metrics:
        print(f"Avg contested claims:      {metrics['avg_contested_claims']:.1f}")

    print()
    print("RISK DISTRIBUTION:")
    risk_dist = metrics['risk_distribution']
    if risk_dist:
        for risk_level in ['low', 'medium', 'high']:
            count = risk_dist.get(risk_level, 0)
            print(f"  {risk_level:8s}: {count}")
        print(f"  Total benchmark runs: {metrics['total_benchmark_runs']}")
    else:
        print("  (no benchmark data)")

    print()
    print("=" * 60)


if __name__ == '__main__':
    log_path = 'logs/research/research_system.log'

    if not Path(log_path).exists():
        print(f"Error: Log file not found at {log_path}")
        print("Run some research sessions first to generate logs.")
        sys.exit(1)

    metrics = compute_metrics(log_path)
    print_metrics(metrics)
