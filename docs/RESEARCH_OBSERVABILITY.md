# Research System Observability - Quick Reference

**Date**: 2025-11-15
**Status**: ✅ Complete observability suite

## Overview

Three scripts for monitoring and validating research system behavior:
1. **Log views** - Quick queries for common questions
2. **Metrics** - Summary statistics
3. **Health check** - Regression detection with tunable thresholds

---

## Quick Start

### After first benchmark run:

```bash
# 1. View recent activity
./scripts/research_log_views.sh recent

# 2. Get baseline metrics
python3 scripts/research_log_metrics.py

# 3. Run health check
python3 scripts/research_health_check.py
```

### After making changes:

```bash
# Run benchmark
python3 src/test_research_benchmark_astra.py

# Check for regressions
python3 scripts/research_health_check.py --strict
```

---

## 1. Log Views (`scripts/research_log_views.sh`)

Quick one-liners for common triage questions.

### Commands

```bash
# Last 50 synthesis completions
./scripts/research_log_views.sh recent

# Full trace for specific session
./scripts/research_log_views.sh session abc-123

# Last 20 benchmark results
./scripts/research_log_views.sh benchmark

# All high-risk research sessions
./scripts/research_log_views.sh high-risk
```

### When to use

- **After weird answer**: Check session trace to see what she actually did
- **After benchmark**: Quick scan of recent/high-risk to spot patterns
- **Debugging**: Full session trace shows task decomposition and budget usage

---

## 2. Metrics (`scripts/research_log_metrics.py`)

Summary statistics from all logged research activity.

### Usage

```bash
python3 scripts/research_log_metrics.py
```

### Output

```
============================================================
RESEARCH SYSTEM METRICS
============================================================

Sessions completed:        4
Avg tasks per session:     11.5
Avg docs per session:      8.3
Avg claims per session:    42.1
Avg contested claims:      2.3

RISK DISTRIBUTION:
  low     : 1
  medium  : 2
  high    : 1
  Total benchmark runs: 4

============================================================
```

### When to use

- **After first benchmark**: Establish baseline numbers
- **After changes**: Verify metrics stayed stable
- **Before tuning**: Understand current behavior

### What to look for

- **Tasks per session**: Should be ≤ 20 on average (soft cap), never consistently > 28
- **Docs per session**: Should be ≥ 5 (sufficient research depth)
- **Risk distribution**: High risk should be ~20-40% of current events questions
- **Contested claims**: If average > 5, source quality or extraction is noisy

---

## 3. Health Check (`scripts/research_health_check.py`)

Automated regression detection with configurable thresholds.

### Usage

```bash
# Normal mode (warnings don't fail)
python3 scripts/research_health_check.py

# Strict mode (warnings also fail - for CI)
python3 scripts/research_health_check.py --strict
```

### What it checks

#### Risk Distribution
- **WARN** if high risk > 40% (overcautious)
- **INFO** if high risk < 20% (may be undercalling)

#### Task Budgets
- **WARN** if avg tasks ≥ 20 (soft cap)
- **CRITICAL** if avg tasks ≥ 28 (hard cap approach)

#### Source Quality
- **WARN** if avg docs < 5 (insufficient research)
- **WARN** if avg contested claims > 5 (noisy extraction)

### Exit codes

- `0` - Healthy (only INFO or passed)
- `1` - Failed (CRITICAL or WARN in strict mode)

### Example output

```
======================================================================
RESEARCH SYSTEM HEALTH CHECK
======================================================================

⚠️  [WARN] High risk percentage too high: 75.0% (expected 20-40%).
Risk classifier may be overcautious.

✅ Task budgets healthy
✅ Source quality healthy

======================================================================

CONFIGURED THRESHOLDS:
  High risk %:            20-40%
  Avg tasks (warn):       20
  Avg tasks (critical):   28
  Avg docs (min):         5
  Avg contested (max):    5

⚠️  HEALTH CHECK PASSED WITH WARNINGS
```

---

## Tuning Thresholds

Edit `scripts/research_health_check.py` at the top:

```python
THRESHOLDS = {
    'high_risk_pct_min': 20,      # Lower bound for high risk %
    'high_risk_pct_max': 40,      # Upper bound for high risk %
    'avg_tasks_warn': 20,         # Soft cap for tasks/session
    'avg_tasks_critical': 28,     # Hard cap (near budget limit)
    'avg_docs_min': 5,            # Minimum docs for thorough research
    'avg_contested_max': 5,       # Max contested claims before noisy
    'min_benchmark_runs': 2,      # Min runs before checks meaningful
}
```

### How to set thresholds

1. **Run baseline**: Get initial metrics from first few benchmarks
2. **Define normal**: Based on baseline, decide what's acceptable
3. **Set thresholds**: Add ~20% buffer above/below normal
4. **Iterate**: Adjust based on false positives/negatives

Example:
- Baseline shows avg tasks = 12
- Set warn at 20 (plenty of headroom)
- Set critical at 28 (near 30 budget limit)

---

## Workflow Examples

### After first benchmark

```bash
# 1. Run benchmark
python3 src/test_research_benchmark_astra.py

# 2. Get baseline metrics
python3 scripts/research_log_metrics.py

# 3. Document baseline in notes
# Sessions: 4, Tasks: 11.5, Docs: 8.3, High risk: 25%

# 4. Run health check (should pass or have minor warnings)
python3 scripts/research_health_check.py
```

### After changing risk classifier

```bash
# 1. Run benchmark
python3 src/test_research_benchmark_astra.py

# 2. Check risk distribution changed as expected
./scripts/research_log_views.sh high-risk

# 3. Verify no other regressions
python3 scripts/research_health_check.py

# 4. If high-risk % out of range, adjust classifier or thresholds
```

### After changing HTN decomposition

```bash
# 1. Run benchmark
python3 src/test_research_benchmark_astra.py

# 2. Check task budgets
python3 scripts/research_log_metrics.py

# 3. Verify tasks didn't spike
python3 scripts/research_health_check.py --strict

# 4. If tasks spiked, check session traces
./scripts/research_log_views.sh recent
```

### Daily triage

```bash
# Quick scan of recent activity
./scripts/research_log_views.sh recent | tail -20

# Any high-risk sessions?
./scripts/research_log_views.sh high-risk | tail -10

# Metrics in normal range?
python3 scripts/research_log_metrics.py
```

---

## Integration with CI/CD

Add to test script or pre-commit hook:

```bash
#!/bin/bash
set -e

# Run benchmark
python3 src/test_research_benchmark_astra.py

# Run automated violation detector
python -m src.analyze_benchmark_results data/research_benchmark_astra_results_*.json

# Run health check in strict mode
python3 scripts/research_health_check.py --strict

echo "✅ All research system checks passed"
```

---

## Files

- `scripts/research_log_views.sh` - Quick log queries
- `scripts/research_log_metrics.py` - Summary statistics
- `scripts/research_health_check.py` - Regression detection
- `logs/research/research_system.log` - Source data

---

## Bottom Line

**Triage**: Use log views to understand what happened
**Baseline**: Use metrics to define "normal"
**Regression detection**: Use health check to catch drift

All three scripts are standalone and require no dependencies beyond Python stdlib.
