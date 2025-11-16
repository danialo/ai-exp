#!/bin/bash
# Monitor Goal Generator Activity

echo "=========================================="
echo "Goal Generator Monitoring Dashboard"
echo "=========================================="
echo ""

# Check if logs exist
if [ ! -f "logs/app/astra.log" ]; then
    echo "⚠️  No logs found at logs/app/astra.log"
    echo "   App may not be running yet."
    exit 1
fi

# 1. Startup Status
echo "📋 Goal Generator Startup Status:"
echo "---"
grep "Goal generator background task started" logs/app/astra.log | tail -1
grep "Goal generator loop started" logs/app/astra.log | tail -1
echo ""

# 2. Recent Activity (last 24 hours)
echo "🔄 Recent Pattern Detection Runs:"
echo "---"
grep -E "Goal generator: [0-9]+ created, [0-9]+ rejected" logs/app/astra.log | tail -5
echo ""

# 3. Telemetry
echo "📊 Latest Telemetry:"
echo "---"
grep "Goal generator telemetry:" logs/app/astra.log | tail -1
echo ""

# 4. System-Generated Goals
echo "🎯 System-Generated Goals (last 10):"
echo "---"
sqlite3 data/raw_store.db <<SQL
.mode column
.headers on
SELECT
    substr(id, 1, 8) as id,
    substr(text, 1, 50) as goal_text,
    source,
    created_by,
    state,
    auto_approved,
    datetime(created_at, 'unixepoch', 'localtime') as created
FROM goals
WHERE source='system'
ORDER BY created_at DESC
LIMIT 10;
SQL
echo ""

# 5. Pattern Detection Errors
echo "❌ Recent Errors:"
echo "---"
grep -E "(Goal generation failed|Failed to initialize goal generator)" logs/app/astra.log | tail -3
echo ""

# 6. Live Monitor Instructions
echo "=========================================="
echo "📡 Live Monitoring Commands:"
echo "=========================================="
echo ""
echo "1. Watch goal generator activity:"
echo "   tail -f logs/app/astra.log | grep -i 'goal generator'"
echo ""
echo "2. Watch identity ledger for goal creation:"
echo "   tail -f data/identity_ledger.ndjson | jq 'select(.event==\"goal_created\")'"
echo ""
echo "3. Watch ALL pattern detection activity:"
echo "   tail -f logs/app/astra.log | grep -E '(Goal generator|Pattern|Detector)'"
echo ""
