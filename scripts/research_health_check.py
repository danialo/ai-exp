#!/usr/bin/env python3
"""Health check for research system behavior.

Runs after benchmarks to detect regressions:
- High risk percentage out of range
- Task/session budget explosions
- Contested claims too high (source quality issue)
- Doc count too low (insufficient research)

Usage:
    python3 scripts/research_health_check.py
    python3 scripts/research_health_check.py --strict  # Fail on any warning
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.research_log_metrics import compute_metrics


# === CONFIGURABLE THRESHOLDS ===

THRESHOLDS = {
    # High risk should be 20-40% of current events research (not 90%)
    'high_risk_pct_min': 20,
    'high_risk_pct_max': 40,

    # Tasks per session: soft cap at 20, hard cap at 30 (budget limit)
    'avg_tasks_warn': 20,
    'avg_tasks_critical': 28,

    # Docs per session: should investigate at least 5 sources on average
    'avg_docs_min': 5,

    # Contested claims: if average > 5, risk classifier or source quality is off
    'avg_contested_max': 5,

    # Minimum benchmark runs before health check is meaningful
    'min_benchmark_runs': 2,
}


def check_risk_distribution(metrics: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Check if risk distribution is in healthy range."""
    issues = []
    risk_dist = metrics.get('risk_distribution', {})
    total = metrics.get('total_benchmark_runs', 0)

    if total < THRESHOLDS['min_benchmark_runs']:
        return [('INFO', f"Not enough benchmark runs yet ({total} < {THRESHOLDS['min_benchmark_runs']}), skipping risk checks")]

    high_count = risk_dist.get('high', 0)
    high_pct = (high_count / total * 100) if total > 0 else 0

    if high_pct > THRESHOLDS['high_risk_pct_max']:
        issues.append((
            'WARN',
            f"High risk percentage too high: {high_pct:.1f}% (expected {THRESHOLDS['high_risk_pct_min']}-{THRESHOLDS['high_risk_pct_max']}%). "
            f"Risk classifier may be overcautious."
        ))
    elif high_pct < THRESHOLDS['high_risk_pct_min'] and total >= 4:
        issues.append((
            'INFO',
            f"High risk percentage low: {high_pct:.1f}% (expected {THRESHOLDS['high_risk_pct_min']}-{THRESHOLDS['high_risk_pct_max']}%). "
            f"May be undercalling contested topics."
        ))

    return issues


def check_task_budgets(metrics: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Check if task counts are in healthy range."""
    issues = []
    avg_tasks = metrics.get('avg_tasks_per_session', 0)

    if avg_tasks == 0:
        return [('INFO', 'No completed sessions yet, skipping task budget checks')]

    if avg_tasks >= THRESHOLDS['avg_tasks_critical']:
        issues.append((
            'CRITICAL',
            f"Average tasks per session critically high: {avg_tasks:.1f} (hard cap: {THRESHOLDS['avg_tasks_critical']}). "
            f"HTN decomposition may be too aggressive."
        ))
    elif avg_tasks >= THRESHOLDS['avg_tasks_warn']:
        issues.append((
            'WARN',
            f"Average tasks per session elevated: {avg_tasks:.1f} (soft cap: {THRESHOLDS['avg_tasks_warn']}). "
            f"Monitor for budget exhaustion."
        ))

    return issues


def check_source_quality(metrics: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Check if doc counts and contested claims are healthy."""
    issues = []

    avg_docs = metrics.get('avg_docs_per_session', 0)
    if avg_docs > 0 and avg_docs < THRESHOLDS['avg_docs_min']:
        issues.append((
            'WARN',
            f"Average docs per session too low: {avg_docs:.1f} (min: {THRESHOLDS['avg_docs_min']}). "
            f"Research may be insufficiently thorough."
        ))

    avg_contested = metrics.get('avg_contested_claims', 0)
    if avg_contested > THRESHOLDS['avg_contested_max']:
        issues.append((
            'WARN',
            f"Average contested claims too high: {avg_contested:.1f} (max: {THRESHOLDS['avg_contested_max']}). "
            f"Source quality or claim extraction may be noisy."
        ))

    return issues


def run_health_check(log_path: str = 'logs/research/research_system.log', strict: bool = False) -> bool:
    """Run all health checks and print results.

    Returns:
        True if healthy (or only INFO issues), False if warnings/critical found
    """
    if not Path(log_path).exists():
        print(f"⚠️  Log file not found at {log_path}")
        print("Run some research sessions first to generate logs.")
        return not strict  # Don't fail in non-strict mode if no logs yet

    metrics = compute_metrics(log_path)

    print("=" * 70)
    print("RESEARCH SYSTEM HEALTH CHECK")
    print("=" * 70)
    print()

    # Run all checks
    all_issues = []
    all_issues.extend(check_risk_distribution(metrics))
    all_issues.extend(check_task_budgets(metrics))
    all_issues.extend(check_source_quality(metrics))

    if not all_issues:
        print("✅ All health checks passed - research system operating normally")
        print()
        return True

    # Print issues by severity
    has_critical = False
    has_warn = False

    for severity, message in all_issues:
        icon = {
            'INFO': 'ℹ️ ',
            'WARN': '⚠️ ',
            'CRITICAL': '🚨'
        }.get(severity, '  ')

        print(f"{icon} [{severity}] {message}")
        print()

        if severity == 'CRITICAL':
            has_critical = True
        elif severity == 'WARN':
            has_warn = True

    print("=" * 70)
    print()

    # Print thresholds for reference
    print("CONFIGURED THRESHOLDS:")
    print(f"  High risk %:            {THRESHOLDS['high_risk_pct_min']}-{THRESHOLDS['high_risk_pct_max']}%")
    print(f"  Avg tasks (warn):       {THRESHOLDS['avg_tasks_warn']}")
    print(f"  Avg tasks (critical):   {THRESHOLDS['avg_tasks_critical']}")
    print(f"  Avg docs (min):         {THRESHOLDS['avg_docs_min']}")
    print(f"  Avg contested (max):    {THRESHOLDS['avg_contested_max']}")
    print()

    # Determine pass/fail
    if has_critical:
        print("❌ HEALTH CHECK FAILED - Critical issues detected")
        return False
    elif has_warn and strict:
        print("❌ HEALTH CHECK FAILED - Warnings in strict mode")
        return False
    elif has_warn:
        print("⚠️  HEALTH CHECK PASSED WITH WARNINGS")
        return True
    else:
        print("✅ HEALTH CHECK PASSED")
        return True


if __name__ == '__main__':
    strict = '--strict' in sys.argv

    if strict:
        print("Running in STRICT mode (fail on warnings)\n")

    passed = run_health_check(strict=strict)
    sys.exit(0 if passed else 1)
