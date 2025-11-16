#!/bin/bash
# Run complete test battery for autonomous coding pipeline
#
# Tests all three phases:
#   Phase 1: TaskExecutionEngine + Executors
#   Phase 2: GoalExecutionService (HTN Planning)
#   Phase 3: GoalStore Integration + Full Pipeline

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  AUTONOMOUS CODING PIPELINE - COMPLETE TEST BATTERY                ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PHASE1_PASS=0
PHASE1_FAIL=0
PHASE2_PASS=0
PHASE2_FAIL=0
PHASE3_PASS=0
PHASE3_FAIL=0

# Test output directory
mkdir -p test_results

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  PHASE 1: TaskExecutionEngine + Executors${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

if pytest tests/integration/test_phase1_execution_engine.py -v --tb=short --color=yes 2>&1 | tee test_results/phase1.log; then
    echo -e "${GREEN}✓ Phase 1 tests PASSED${NC}"
    PHASE1_PASS=$(grep -c "PASSED" test_results/phase1.log || echo 0)
else
    echo -e "${RED}✗ Phase 1 tests FAILED${NC}"
    PHASE1_FAIL=$(grep -c "FAILED" test_results/phase1.log || echo 0)
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  PHASE 2: GoalExecutionService (HTN Planning)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

if pytest tests/integration/test_phase2_goal_execution_service.py -v --tb=short --color=yes 2>&1 | tee test_results/phase2.log; then
    echo -e "${GREEN}✓ Phase 2 tests PASSED${NC}"
    PHASE2_PASS=$(grep -c "PASSED" test_results/phase2.log || echo 0)
else
    echo -e "${RED}✗ Phase 2 tests FAILED${NC}"
    PHASE2_FAIL=$(grep -c "FAILED" test_results/phase2.log || echo 0)
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  PHASE 3: GoalStore Integration + Full Pipeline${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

if pytest tests/integration/test_phase3_goalstore_integration.py -v --tb=short --color=yes 2>&1 | tee test_results/phase3.log; then
    echo -e "${GREEN}✓ Phase 3 tests PASSED${NC}"
    PHASE3_PASS=$(grep -c "PASSED" test_results/phase3.log || echo 0)
else
    echo -e "${RED}✗ Phase 3 tests FAILED${NC}"
    PHASE3_FAIL=$(grep -c "FAILED" test_results/phase3.log || echo 0)
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  TEST SUMMARY                                                       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL_PASS=$((PHASE1_PASS + PHASE2_PASS + PHASE3_PASS))
TOTAL_FAIL=$((PHASE1_FAIL + PHASE2_FAIL + PHASE3_FAIL))
TOTAL_TESTS=$((TOTAL_PASS + TOTAL_FAIL))

echo "Phase 1 (Execution Engine):"
echo -e "  ${GREEN}Passed: $PHASE1_PASS${NC}  ${RED}Failed: $PHASE1_FAIL${NC}"
echo ""
echo "Phase 2 (HTN Planning):"
echo -e "  ${GREEN}Passed: $PHASE2_PASS${NC}  ${RED}Failed: $PHASE2_FAIL${NC}"
echo ""
echo "Phase 3 (Full Pipeline):"
echo -e "  ${GREEN}Passed: $PHASE3_PASS${NC}  ${RED}Failed: $PHASE3_FAIL${NC}"
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "TOTAL:"
echo -e "  ${GREEN}Passed: $TOTAL_PASS${NC}  ${RED}Failed: $TOTAL_FAIL${NC}  Total: $TOTAL_TESTS"
echo ""

if [ $TOTAL_FAIL -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ ALL TESTS PASSED - AUTONOMOUS CODING PIPELINE VERIFIED          ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ✗ SOME TESTS FAILED - SEE LOGS FOR DETAILS                         ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Test logs saved to: test_results/"
    exit 1
fi
