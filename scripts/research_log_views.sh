#!/bin/bash
# Quick log views for research system observability

LOG_FILE="logs/research/research_system.log"

case "$1" in
  recent)
    # Last 50 synthesis completions
    echo "=== RECENT RESEARCH SESSIONS (last 50) ==="
    grep "event=synthesis_complete" "$LOG_FILE" | tail -50
    ;;

  session)
    # Per-session trace
    if [ -z "$2" ]; then
      echo "Usage: $0 session <session_id>"
      exit 1
    fi
    echo "=== SESSION TRACE: $2 ==="
    grep "session=$2" "$LOG_FILE"
    ;;

  benchmark)
    # Benchmark runs summary
    echo "=== BENCHMARK RUNS (last 20) ==="
    grep "benchmark_result" "$LOG_FILE" | tail -20
    ;;

  high-risk)
    # High-risk research for sanity checks
    echo "=== HIGH-RISK RESEARCH SESSIONS ==="
    grep "event=synthesis_complete" "$LOG_FILE" | grep "risk=high"
    ;;

  *)
    echo "Usage: $0 {recent|session <id>|benchmark|high-risk}"
    echo ""
    echo "Commands:"
    echo "  recent       - Last 50 synthesis completions"
    echo "  session <id> - Full trace for specific session"
    echo "  benchmark    - Last 20 benchmark results"
    echo "  high-risk    - All high-risk research sessions"
    exit 1
    ;;
esac
