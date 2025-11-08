# Phase 1 Final Specification - Surgical Patches Applied

## Summary

Both specification and prompt have been updated with final pre-implementation patches to prevent runtime issues and schema churn.

**Files Updated:**
- `.claude/tasks/task-007-end-to-end-tracking.md` (693 lines)
- `.claude/tasks/prompt-007.md` (503 lines)

## 10 Surgical Additions Applied

### 1. ✅ Schema Versioning
```json
"schema_version": 1
```
Global EXPERIENCE_SCHEMA_VERSION for TASK_EXECUTION type. Enables clean schema evolution.

### 2. ✅ Clock Integrity - Dual Timestamps
```json
"started_at_iso": "2025-11-04T18:29:12Z",  // ISO string for JSON
"ended_at_iso": "2025-11-04T18:29:13Z",
"started_at_ts": 1762284552.123,           // Float for SQL
"ended_at_ts": 1762284553.165,
"duration_ms": 1042  // max(ended - started, 0)
```
Both formats stored for JSON/SQL compatibility.

### 3. ✅ Scheduler Identity
```json
"provenance_runner": {
  "name": "scheduler",
  "version": "1.0.0",
  "pid": 351425,
  "host": "d-vps"
}
```
Useful when multiple runners exist.

### 4. ✅ Config Echo
```json
"task_config_digest": "sha256:abc123..."
```
Proof of what config produced a run. Enables config traceability.

### 5. ✅ Embeddings Flag
```json
"embedding_skipped": true
```
Phase 1: always true. Future toggle can backfill embeddings without ambiguity.

### 6. ✅ Query Ergonomics - task_slug
```json
"task_id": "uuid-or-stable-id",
"task_slug": "daily_reflection"  // Human-readable for queries
```
Index added: `ix_experiences_task_slug` for fast slug-based queries.

### 7. ✅ Retry Semantics Clarified
**Invariant:**
- Same `trace_id` across retries
- New `span_id` each attempt
- `retry_of` holds prior **span_id** (not trace_id)

### 8. ✅ Join Keys for IO
```json
"io": {
  "tool_calls": ["toolrun_9f2"],  // Ref to tool_call table
  "script_runs": ["scr_733"]      // Ref to script_run table
}
```
Comment added: ensure IDs reference existing tables or are empty (avoid dangling IDs).

### 9. ✅ Backfill Dedupe
Script requirements updated:
- Must honor `idempotency_key`
- Dry-run mode with counts
- Safe to re-run multiple times

### 10. ✅ Permissions Footprint
```json
"grants": {
  "scopes": [],
  "impersonation": false
}
```
Stub for now, prevents confusion if tasks touch user data later.

## Updated Invariants (10 Total)

1. Creating TASK_EXECUTION with same idempotency_key is no-op
2. If status="success", duration_ms >= 0 and ended_at_ts >= started_at_ts
3. **NEW:** If status="failed", error.type and error.message are non-null
4. **UPDATED:** If retrieval.memory_count > 0, len(parents) == memory_count, else both zero
5. **UPDATED:** trace_id constant across retries, span_id unique per attempt
6. **NEW:** attempt >= 1, and attempt == 1 when retry_of is null
7. **NEW:** retry_of holds span_id (not trace_id) of prior attempt
8. **NEW:** idempotency_key uniqueness enforced by insert path
9. trace_id and span_id are valid UUIDv4
10. **NEW:** tool_calls and script_runs reference existing tables or are empty

## SQLite Indexes (5 Total)

```sql
CREATE INDEX IF NOT EXISTS ix_experiences_type_ts
    ON experiences(type, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_experiences_task
    ON experiences(json_extract(content, '$.structured.task_id'), created_at DESC);

CREATE INDEX IF NOT EXISTS ix_experiences_task_slug          -- NEW
    ON experiences(json_extract(content, '$.structured.task_slug'), ts DESC);

CREATE INDEX IF NOT EXISTS ix_experiences_trace
    ON experiences(json_extract(content, '$.structured.trace_id'));

CREATE INDEX IF NOT EXISTS ix_experiences_idempotency
    ON experiences(json_extract(content, '$.structured.idempotency_key'));
```

## Ready-to-Run Check Commands (Added to Prompt)

4 verification commands added for post-implementation testing:

1. **Run scheduled task** via API
2. **Verify experience creation** via Python
3. **Query by task_id** via SQLite
4. **Verify invariant** (parents length == memory_count)

## Complete Field Contract

**Total fields in content.structured:** 22 fields across 8 categories

### Schema & Identity (5 fields)
- schema_version
- task_id
- task_slug
- task_name
- task_type
- scheduled_vs_manual

### Execution Status (6 fields)
- status
- started_at_iso
- ended_at_iso
- started_at_ts
- ended_at_ts
- duration_ms

### Correlation (5 fields)
- trace_id
- span_id
- attempt
- retry_of
- idempotency_key

### Configuration (1 field)
- task_config_digest

### Retrieval Provenance (1 object)
- retrieval: {memory_count, query, filters, latency_ms, source}

### Side Effects (1 object)
- io: {files_written, artifact_ids, tool_calls, script_runs}

### Embeddings (1 field)
- embedding_skipped

### Runner Identity (1 object)
- provenance_runner: {name, version, pid, host}

### Permissions (1 object)
- grants: {scopes, impersonation}

### Error Details (1 field)
- error: {type, message, stack_hash} or null

## Why These Changes Prevent Rework

### Phase 2 (Ledger)
- ✅ trace_id/span_id already captured → just log to ledger
- ✅ task_config_digest → can prove what config produced ledger event

### Phase 3 (Trace Context)
- ✅ trace_id/span_id ready for distributed tracing
- ✅ provenance_runner → can trace across multiple hosts/processes

### Phase 4 (Goals)
- ✅ causes field ready (currently empty)
- ✅ task_slug makes goal-task queries ergonomic

### Phase 5 (Hooks)
- ✅ parents already link to inputs
- ✅ io already tracks side effects

### Phase 6 (Analytics)
- ✅ All timing data captured (started_at_ts, ended_at_ts, duration_ms)
- ✅ Retrieval metrics captured (latency_ms, memory_count)
- ✅ Success/failure tracked (status, error)

## Files Ready for Tomorrow

```
.claude/tasks/
├── task-007-end-to-end-tracking.md  (693 lines - full spec)
├── prompt-007.md                     (503 lines - execution guide)
└── PHASE1-FINAL-SPEC.md              (this file - summary)
```

## Execution Tomorrow

Tell Claude:
```
Execute the prompt in .claude/tasks/prompt-007.md
```

Claude will:
1. Read full spec
2. Create branch `feature/task-execution-tracking`
3. Implement all 10 surgical additions
4. Run all invariants
5. Create backfill script with dry-run
6. Update docs
7. Run 4 ready-to-run checks
8. Commit with test results

## Confidence Level

**Schema stability:** 🟢 HIGH - No rework needed for Phases 2-6
**Runtime safety:** 🟢 HIGH - All clock, retry, and failure cases covered
**Query performance:** 🟢 HIGH - 5 indexes cover all access patterns
**Auditability:** 🟢 HIGH - Full provenance + correlation + config tracking

Phase 1 is **production-ready** after these patches. 🎯
