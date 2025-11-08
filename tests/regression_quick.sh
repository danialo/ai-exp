#!/bin/bash
# Astra Quick Regression Test Suite
# Version: Baseline Stable (Nov 2025)
# Run before: merges, deploys, major changes

echo "=========================================="
echo "Astra Regression Tests - Quick Suite"
echo "=========================================="
echo ""

PASS_COUNT=0
FAIL_COUNT=0

# Helper functions
pass_test() {
    echo "✅ $1"
    ((PASS_COUNT++))
}

fail_test() {
    echo "❌ $1"
    ((FAIL_COUNT++))
}

warn_test() {
    echo "⚠️  $1"
}

# Test 1: Server Running
echo "[1/10] Checking server status..."
if curl -s --max-time 5 http://localhost:8000/api/awareness/status > /dev/null 2>&1; then
    pass_test "Server is running"
else
    fail_test "Server not responding on port 8000"
    echo "       Run: python app.py"
    exit 1
fi

# Test 2: Awareness Loop Running
echo "[2/10] Checking awareness loop..."
STATUS=$(curl -s http://localhost:8000/api/awareness/status)
RUNNING=$(echo $STATUS | python3 -c "import sys, json; print(json.load(sys.stdin)['running'])" 2>/dev/null)
if [ "$RUNNING" = "True" ]; then
    pass_test "Awareness loop running"
else
    fail_test "Awareness loop not running"
fi

# Test 3: Text Percepts Check (CRITICAL BUG FIX VERIFICATION)
echo "[3/10] Checking text percept deduplication..."
TEXT=$(echo $STATUS | python3 -c "import sys, json; print(json.load(sys.stdin)['buffer']['text_percepts'])" 2>/dev/null)
TIME=$(echo $STATUS | python3 -c "import sys, json; print(json.load(sys.stdin)['buffer']['by_kind'].get('time', 0))" 2>/dev/null)
if [ "$TIME" -le 2 ]; then
    pass_test "Time percept deduplication working (time=$TIME)"
else
    fail_test "Time percepts accumulating: $TIME (should be ~1) - REGRESSION!"
fi

# Test 4: Basic Chat
echo "[4/10] Testing basic chat endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Regression test", "retrieve_memories": false}' 2>/dev/null)
if echo $RESPONSE | grep -q "response"; then
    pass_test "Basic chat working"
else
    fail_test "Chat endpoint failed"
fi

# Test 5: Memory Retrieval
echo "[5/10] Testing memory retrieval..."
MEM_RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Test with memories", "retrieve_memories": true, "top_k": 3}' 2>/dev/null)
if echo $MEM_RESPONSE | grep -q "memories"; then
    pass_test "Memory retrieval working"
else
    fail_test "Memory retrieval failed"
fi

# Test 6: Belief System
echo "[6/10] Checking belief system..."
BELIEFS=$(curl -s http://localhost:8000/api/beliefs 2>/dev/null)
CORE_COUNT=$(echo $BELIEFS | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_core', 0))" 2>/dev/null)
if [ "$CORE_COUNT" = "5" ]; then
    pass_test "Belief system loaded (5 core beliefs)"
else
    fail_test "Belief system error (expected 5 core beliefs, got $CORE_COUNT)"
fi

# Test 7: Events Dropped Check
echo "[7/10] Checking for queue overflows..."
DROPPED=$(echo $STATUS | python3 -c "import sys, json; print(json.load(sys.stdin)['metrics']['events_dropped'])" 2>/dev/null)
if [ "$DROPPED" = "0" ]; then
    pass_test "No events dropped"
else
    warn_test "$DROPPED events dropped (queue overflow)"
fi

# Test 8: Performance Check
echo "[8/10] Checking awareness loop performance..."
MEAN_MS=$(echo $STATUS | python3 -c "import sys, json; print(json.load(sys.stdin)['metrics']['tick_ms']['mean'])" 2>/dev/null)
if [ $(echo "$MEAN_MS < 5.0" | bc -l) -eq 1 ] 2>/dev/null; then
    pass_test "Tick performance good (${MEAN_MS}ms mean)"
else
    warn_test "Tick performance slow (${MEAN_MS}ms mean)"
fi

# Test 9: Empty Message Validation
echo "[9/10] Testing input validation..."
EMPTY_RESP=$(curl -s -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "", "retrieve_memories": false}' 2>/dev/null)
if echo $EMPTY_RESP | grep -q "Message cannot be empty"; then
    pass_test "Input validation working"
else
    fail_test "Empty message not rejected"
fi

# Test 10: Application Errors Check
echo "[10/10] Checking for application errors..."
if [ -f "logs/errors/errors.log" ]; then
    ERROR_COUNT=$(grep -v "chromadb.telemetry" logs/errors/errors.log 2>/dev/null | wc -l)
    if [ "$ERROR_COUNT" -eq 0 ]; then
        pass_test "No application errors in logs"
    else
        warn_test "$ERROR_COUNT non-chromadb errors found in logs"
    fi
else
    warn_test "Error log file not found"
fi

# Summary
echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - System is healthy"
    exit 0
else
    echo "❌ $FAIL_COUNT TEST(S) FAILED - Investigation required"
    exit 1
fi
