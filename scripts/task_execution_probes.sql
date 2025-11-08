-- SQL probes for monitoring task execution tracking
-- Run these against data/core.db to observe task execution patterns
--
-- Usage:
--   sqlite3 data/core.db < scripts/task_execution_probes.sql
--   sqlite3 data/core.db ".read scripts/task_execution_probes.sql"

.headers on
.mode column

-- ============================================================
-- BASIC STATS
-- ============================================================

.print ""
.print "=========================================="
.print "TASK EXECUTION SUMMARY"
.print "=========================================="

-- Total count by status
SELECT
    json_extract(content, '$.structured.status') as status,
    COUNT(*) as count
FROM experience
WHERE type = 'task_execution'
GROUP BY status
ORDER BY count DESC;

.print ""
.print "Task counts by task_id:"
SELECT
    json_extract(content, '$.structured.task_id') as task_id,
    COUNT(*) as executions
FROM experience
WHERE type = 'task_execution'
GROUP BY task_id
ORDER BY executions DESC
LIMIT 20;

-- ============================================================
-- PERFORMANCE METRICS
-- ============================================================

.print ""
.print "=========================================="
.print "PERFORMANCE METRICS"
.print "=========================================="

-- Duration stats by task
SELECT
    json_extract(content, '$.structured.task_id') as task_id,
    COUNT(*) as runs,
    ROUND(AVG(CAST(json_extract(content, '$.structured.duration_ms') AS REAL)), 2) as mean_ms,
    MIN(json_extract(content, '$.structured.duration_ms')) as min_ms,
    MAX(json_extract(content, '$.structured.duration_ms')) as max_ms
FROM experience
WHERE type = 'task_execution'
GROUP BY task_id
ORDER BY mean_ms DESC;

-- ============================================================
-- FAILURE ANALYSIS
-- ============================================================

.print ""
.print "=========================================="
.print "RECENT FAILURES"
.print "=========================================="

-- Recent failures with error details
SELECT
    id,
    json_extract(content, '$.structured.task_id') as task_id,
    json_extract(content, '$.structured.error.type') as error_type,
    json_extract(content, '$.structured.error.stack_hash') as stack_hash,
    datetime(created_at) as occurred_at
FROM experience
WHERE type = 'task_execution'
  AND json_extract(content, '$.structured.status') = 'failed'
ORDER BY created_at DESC
LIMIT 20;

.print ""
.print "Distinct error patterns (by stack hash):"
SELECT
    json_extract(content, '$.structured.task_id') as task_id,
    json_extract(content, '$.structured.error.type') as error_type,
    json_extract(content, '$.structured.error.stack_hash') as stack_hash,
    COUNT(*) as occurrences,
    MAX(datetime(created_at)) as last_seen
FROM experience
WHERE type = 'task_execution'
  AND json_extract(content, '$.structured.status') = 'failed'
GROUP BY task_id, error_type, stack_hash
ORDER BY occurrences DESC
LIMIT 20;

-- ============================================================
-- RETRY ANALYSIS
-- ============================================================

.print ""
.print "=========================================="
.print "RETRY PATTERNS"
.print "=========================================="

-- Traces with multiple attempts (retries)
SELECT
    json_extract(content, '$.structured.trace_id') as trace_id,
    json_extract(content, '$.structured.task_id') as task_id,
    COUNT(*) as attempts
FROM experience
WHERE type = 'task_execution'
GROUP BY trace_id
HAVING attempts > 1
ORDER BY attempts DESC
LIMIT 20;

-- ============================================================
-- SUCCESS RATE TRENDS
-- ============================================================

.print ""
.print "=========================================="
.print "SUCCESS RATE BY TASK"
.print "=========================================="

SELECT
    json_extract(content, '$.structured.task_id') as task_id,
    COUNT(*) as total_runs,
    SUM(CASE WHEN json_extract(content, '$.structured.status') = 'success' THEN 1 ELSE 0 END) as successes,
    SUM(CASE WHEN json_extract(content, '$.structured.status') = 'failed' THEN 1 ELSE 0 END) as failures,
    ROUND(
        100.0 * SUM(CASE WHEN json_extract(content, '$.structured.status') = 'success' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) as success_rate_pct
FROM experience
WHERE type = 'task_execution'
GROUP BY task_id
ORDER BY total_runs DESC;

-- ============================================================
-- RECENT EXECUTIONS
-- ============================================================

.print ""
.print "=========================================="
.print "LAST 10 TASK EXECUTIONS"
.print "=========================================="

SELECT
    substr(id, 1, 40) as experience_id,
    json_extract(content, '$.structured.task_id') as task,
    json_extract(content, '$.structured.status') as status,
    json_extract(content, '$.structured.duration_ms') || 'ms' as duration,
    datetime(created_at) as executed_at
FROM experience
WHERE type = 'task_execution'
ORDER BY created_at DESC
LIMIT 10;

-- ============================================================
-- PARENT MEMORY ANALYSIS
-- ============================================================

.print ""
.print "=========================================="
.print "MEMORY RETRIEVAL STATS"
.print "=========================================="

-- Retrieval stats by task
SELECT
    json_extract(content, '$.structured.task_id') as task_id,
    COUNT(*) as runs,
    ROUND(AVG(CAST(json_extract(content, '$.structured.retrieval.memory_count') AS REAL)), 2) as avg_memories,
    MAX(json_extract(content, '$.structured.retrieval.memory_count')) as max_memories
FROM experience
WHERE type = 'task_execution'
GROUP BY task_id
ORDER BY avg_memories DESC;

-- ============================================================
-- BACKFILLED DATA
-- ============================================================

.print ""
.print "=========================================="
.print "BACKFILLED vs LIVE DATA"
.print "=========================================="

SELECT
    CASE
        WHEN json_extract(content, '$.structured.backfilled') = 'true'
        THEN 'Backfilled'
        ELSE 'Live'
    END as source,
    COUNT(*) as count
FROM experience
WHERE type = 'task_execution'
GROUP BY source;

.print ""
.print "=========================================="
.print "END OF PROBES"
.print "=========================================="
.print ""
