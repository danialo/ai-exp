# Regression Test Checklist

**Purpose:** Verify critical system functionality before deployment
**Estimated Time:** 5-10 minutes
**Run:** Before merging features, after major changes, before production deploy

---

## Quick Health Check (1 minute)

### System Startup
```bash
# Start Astra
python app.py

# Expected: No ERROR lines during startup
# Expected: "Uvicorn running on http://0.0.0.0:8000"
```

**✅ Pass Criteria:**
- [ ] Server starts without errors
- [ ] Awareness loop initialized
- [ ] Belief system loaded
- [ ] No Python exceptions in startup logs

---

## Core Functionality Tests (5 minutes)

### Test 1: Basic Chat Works
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "retrieve_memories": false}'
```

**✅ Pass Criteria:**
- [ ] Returns 200 status
- [ ] Response contains valid JSON
- [ ] Response has `response` field with text
- [ ] Response has `experience_id` field
- [ ] No errors in logs

### Test 2: Memory Retrieval Works
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What did we discuss earlier?", "retrieve_memories": true, "top_k": 3}'
```

**✅ Pass Criteria:**
- [ ] Returns 200 status
- [ ] Response contains `memories` array
- [ ] If experiences exist, memories array populated
- [ ] No retrieval errors in logs

### Test 3: Awareness Loop Running
```bash
curl -s http://localhost:8000/api/awareness/status | python3 -m json.tool
```

**✅ Pass Criteria:**
- [ ] `"running": true`
- [ ] `"tick"` value increasing over time
- [ ] `"buffer.total_percepts"` > 0 after some chat
- [ ] `"buffer.text_percepts"` > 0 after chat messages
- [ ] `"events_dropped": 0`
- [ ] No "Queue full" warnings in logs

### Test 4: Belief System Accessible
```bash
curl -s http://localhost:8000/api/beliefs | python3 -m json.tool
```

**✅ Pass Criteria:**
- [ ] Returns 200 status
- [ ] `"total_core": 5`
- [ ] `"total_peripheral"` >= 0
- [ ] Core beliefs array has 5 items
- [ ] Each belief has `statement`, `confidence`, `belief_type`

### Test 5: Text Percepts Accumulating (CRITICAL FIX VERIFICATION)
```bash
# Send 3 test messages
for i in {1..3}; do
  curl -s -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"Test message $i\", \"retrieve_memories\": false}" > /dev/null
  sleep 2
done

# Wait for slow tick
sleep 12

# Check awareness status
curl -s http://localhost:8000/api/awareness/status | \
  python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Total: {d['buffer']['total_percepts']}, Text: {d['buffer']['text_percepts']}, Time: {d['buffer']['by_kind'].get('time', 0)}\")"
```

**✅ Pass Criteria:**
- [ ] Text percepts >= 6 (3 user + 3 token)
- [ ] Time percepts = 1 (deduplication working!)
- [ ] Total percepts = text + 1
- [ ] **CRITICAL:** Time percepts should NOT flood the buffer

**❌ Fail Signal:** If time percepts > 10, deduplication regression!

---

## Advanced System Tests (3 minutes)

### Test 6: Belief Gardener Running
```bash
# Check logs for gardener activity
tail -100 logs/app/application.log | grep -i "gardener\|pattern scan"
```

**✅ Pass Criteria:**
- [ ] Gardener initialized at startup
- [ ] Pattern scan occurs every 3600s (if running that long)
- [ ] No "unpack" errors
- [ ] No "update_belief" AttributeErrors

### Test 7: Self-Claim Detection Working
```bash
# Send message with self-referential content
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I believe that AI can have consciousness", "retrieve_memories": false}'

# Check logs
tail -50 logs/app/application.log | grep -i "self-claim"
```

**✅ Pass Criteria:**
- [ ] No JSON parsing errors
- [ ] Either detects claims OR logs "No self-claims detected"
- [ ] No "Expecting value: line 1 column 1" errors

### Test 8: Introspection Generating Notes
```bash
curl -s http://localhost:8000/api/awareness/notes | \
  python3 -c "import sys, json; notes=json.load(sys.stdin); print(f\"Total notes: {notes['count']}\"); print(f\"Sample: {notes['notes'][0][:100]}...\" if notes['notes'] else 'No notes yet')"
```

**✅ Pass Criteria:**
- [ ] Notes count > 0 (after some runtime)
- [ ] Notes contain self-reflective content
- [ ] Notes mention "tension", "value", or introspective themes

### Test 9: No Application Errors
```bash
tail -200 logs/errors/errors.log | grep -v "chromadb.telemetry" | wc -l
```

**✅ Pass Criteria:**
- [ ] Count = 0 (zero non-chromadb errors)
- [ ] ChromaDB telemetry errors OK (harmless)

---

## Edge Case Tests (2 minutes)

### Test 10: Empty Message Rejected
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "", "retrieve_memories": false}'
```

**✅ Pass Criteria:**
- [ ] Returns 400 status
- [ ] Error message: "Message cannot be empty"

### Test 11: Unicode and Emoji Handling
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Test unicode: 你好 😊 ñ", "retrieve_memories": false}'
```

**✅ Pass Criteria:**
- [ ] Returns 200 status
- [ ] No encoding errors
- [ ] Response handles unicode correctly

### Test 12: Rapid Message Burst (No Queue Overflow)
```bash
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"Burst test $i\", \"retrieve_memories\": false}" > /dev/null &
done
wait

# Check awareness status
sleep 5
curl -s http://localhost:8000/api/awareness/status | \
  python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Events dropped: {d['metrics']['events_dropped']}\")"
```

**✅ Pass Criteria:**
- [ ] Events dropped = 0
- [ ] No queue overflow warnings
- [ ] All requests return 200

---

## Performance Verification (1 minute)

### Test 13: Awareness Loop Performance
```bash
curl -s http://localhost:8000/api/awareness/status | \
  python3 -c "import sys, json; d=json.load(sys.stdin); m=d['metrics']['tick_ms']; print(f\"Mean: {m['mean']:.1f}ms, P95: {m['p95']:.1f}ms\")"
```

**✅ Pass Criteria:**
- [ ] Mean tick time < 5ms
- [ ] P95 tick time < 10ms

### Test 14: Memory Retrieval Speed
```bash
time curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quick performance test", "retrieve_memories": true, "top_k": 5}' > /dev/null
```

**✅ Pass Criteria:**
- [ ] Total time < 5 seconds
- [ ] If slower, check for LLM API latency (acceptable)

---

## Critical Bug Regression Checks

### Bug Fix 1: Belief Gardener Unpack Error (Nov 4, 2025)
**What to check:** Enhanced feedback aggregator returns 3 values

```bash
# Wait for gardener to run (or trigger manually if endpoint exists)
# Check logs for:
tail -200 logs/app/application.log | grep -i "too many values to unpack"
```

**✅ Pass Criteria:**
- [ ] No "too many values to unpack" errors
- [ ] Gardener runs without exceptions

### Bug Fix 2: Awareness Loop Text Percept Loss (Nov 4, 2025)
**What to check:** Text percepts should accumulate, not be pushed out by time percepts

```bash
# Run Test 5 above (Text Percepts Accumulating)
# Critical: time percepts should be ~1, not hundreds
```

**✅ Pass Criteria:**
- [ ] Time percepts stay at 1 (deduplication working)
- [ ] Text percepts accumulate with each message
- [ ] Buffer doesn't fill with duplicate time percepts

### Bug Fix 3: Self-Claim Detection JSON Parsing (Nov 4, 2025)
**What to check:** No JSON parsing errors when LLM returns plain text

```bash
# Run Test 7 above (Self-Claim Detection Working)
# Check for:
tail -100 logs/app/application.log | grep "Expecting value: line 1 column 1"
```

**✅ Pass Criteria:**
- [ ] No "Expecting value" JSON errors
- [ ] Either detects claims OR handles "no claims" gracefully

---

## Database Health Check

### Test 15: Raw Store Accessible
```bash
python3 << 'EOF'
import sqlite3
conn = sqlite3.connect("data/raw_store.db")
cursor = conn.cursor()
cursor.execute("SELECT type, COUNT(*) FROM experience GROUP BY type")
for row in cursor.fetchall():
    print(f"{row[0]}: {row[1]}")
conn.close()
EOF
```

**✅ Pass Criteria:**
- [ ] Database opens without errors
- [ ] Multiple experience types present
- [ ] Counts seem reasonable

---

## Quick Pass/Fail Summary

**PASS:** System is healthy ✅
- All core functionality tests passed
- No regressions in critical bug fixes
- Performance within acceptable limits
- Zero application errors

**FAIL:** Investigation required ❌
- Any test marked ❌
- New errors in logs
- Performance degradation
- Critical bug regressions

---

## Automated Test Script

```bash
#!/bin/bash
# Save as: tests/regression_quick.sh

echo "Running Astra Regression Tests..."
echo "=================================="

# Test 1: Server Running
curl -s http://localhost:8000/api/awareness/status > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Server is running"
else
    echo "❌ Server not responding"
    exit 1
fi

# Test 2: Awareness Loop
STATUS=$(curl -s http://localhost:8000/api/awareness/status)
RUNNING=$(echo $STATUS | python3 -c "import sys, json; print(json.load(sys.stdin)['running'])")
if [ "$RUNNING" = "True" ]; then
    echo "✅ Awareness loop running"
else
    echo "❌ Awareness loop not running"
fi

# Test 3: Text Percepts Check
TEXT=$(echo $STATUS | python3 -c "import sys, json; print(json.load(sys.stdin)['buffer']['text_percepts'])")
TIME=$(echo $STATUS | python3 -c "import sys, json; print(json.load(sys.stdin)['buffer']['by_kind'].get('time', 0))")
if [ "$TIME" -le 2 ]; then
    echo "✅ Time percept deduplication working (time=$TIME)"
else
    echo "⚠️  Time percepts accumulating: $TIME (should be ~1)"
fi

# Test 4: Basic Chat
RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"message": "test", "retrieve_memories": false}')
if echo $RESPONSE | grep -q "response"; then
    echo "✅ Basic chat working"
else
    echo "❌ Chat endpoint failed"
fi

echo "=================================="
echo "Quick regression check complete"
```

---

**Last Updated:** November 4, 2025
**Version:** Baseline Stable
**Run Before:** Merging branches, production deploys, major changes
