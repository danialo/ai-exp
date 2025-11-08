# PROMPT: Implement End-to-End Task Tracking for Auditability (Phase 1)

## Context

Astra has a task scheduler system that executes scheduled tasks (reflections, goal assessments, etc.), but task executions are not integrated into the experience/memory system. This creates gaps in auditability - we can't trace what context informed a task, what side effects occurred, or how tasks contribute to goals.

## Your Task

Implement **Phase 1: Link Task Executions to Experiences** from the specification in `.claude/tasks/task-007-end-to-end-tracking.md`.

**Read the full specification first** to understand the context and the surgical changes based on expert review.

## Critical Design Decisions (DO NOT DEVIATE)

These are precise requirements from the expert review to avoid schema churn in later phases:

### 1. Use `TASK_EXECUTION` type, NOT `OBSERVATION`
- Keeps query semantics clean
- OBSERVATION is for external events, TASK_EXECUTION is for agent actions

### 2. Session ID Format
```
session_id = f"task:{task_id}:{exec_ts_iso}:{short_uuid}"
```
Example: `"task:daily_reflection:2025-11-04T18:29:12Z:9f1c"`

### 3. Separate `parents` vs `causes`
- `parents` = immediate inputs (retrieved experience IDs)
- `causes` = causal triggers (goal IDs, prior task IDs) - empty for Phase 1, used in Phase 4

### 4. Correlation and Idempotency Fields (REQUIRED)
- `trace_id`: UUIDv4 per execution
- `span_id`: UUIDv4 for this span
- `attempt`: 1-n for retries (always 1 in Phase 1)
- `idempotency_key`: hash of task_id + scheduled_at + attempt

### 5. Full Execution Status Model
- `status`: "success" | "failed" | "partial"
- `error`: null or {"type": str, "message": str, "stack_hash": str}
- `started_at`, `ended_at`, `duration_ms`
- `retry_of`: trace_id of prior attempt (null for Phase 1)

### 6. Resource and Side-Effects Tracking
```python
"io": {
  "files_written": [paths],
  "artifact_ids": [],
  "tool_calls": [ids],
  "script_runs": [ids]
}
```

### 7. Retrieval Provenance
```python
"retrieval": {
  "memory_count": int,
  "query": str,
  "filters": dict,
  "latency_ms": int,
  "source": ["experiences", "beliefs"]
}
```
If no retrieval, set memory_count=0 and omit query.

### 8. Always Create Experience on Failure
- Same structure, but status="failed" and error populated
- Parents still included if retrieval happened before failure

### 9. PII Scrubbing
- Use same scrubber as identity ledger
- Add `meta.scrubbed=true` flag

### 10. Idempotent Inserts
- Check idempotency_key before insert
- If exists, return existing experience_id (no duplicate)

## Specific Implementation Steps

### Step 1: Review Current Code

Read and understand:
- `src/services/task_scheduler.py` - Task execution flow
- `src/memory/raw_store.py` - Experience storage
- `src/memory/models.py` - Experience model
- `src/pipeline/ingest.py` - How experiences are created, find PII scrubber
- `src/services/identity_ledger.py` - PII scrubbing implementation

### Step 2: Add New Types to Models

**File: `src/memory/models.py`**

Add to ExperienceType enum:
```python
class ExperienceType(str, Enum):
    # ... existing ...
    TASK_EXECUTION = "task_execution"
```

Add to CaptureMethod enum:
```python
class CaptureMethod(str, Enum):
    # ... existing ...
    SCHEDULED_TASK = "scheduled_task"
    MANUAL_TASK = "manual_task"
```

Add `causes` field to ExperienceModel:
```python
class ExperienceModel(BaseModel):
    # ... existing fields ...
    parents: List[str] = Field(default_factory=list)
    causes: List[str] = Field(default_factory=list)  # NEW
```

### Step 3: Create Task Experience Helper

**File: `src/pipeline/task_experience.py` (NEW)**

Create helper function that:
1. Takes task execution data (task_id, response, timing, retrieval info, etc.)
2. Generates trace_id, span_id, idempotency_key
3. Builds TASK_EXECUTION experience with all required fields
4. Runs PII scrubber on content.text
5. Returns ExperienceModel

Reference the concrete field contract in the specification for exact structure.

### Step 4: Add Idempotent Insert to Raw Store

**File: `src/memory/raw_store.py`**

Add method:
```python
def append_experience_idempotent(self, experience: ExperienceModel, idempotency_key: str) -> str:
    """
    Insert experience only if idempotency_key not seen before.

    Returns:
        experience_id (existing or newly created)
    """
    # 1. Query for existing experience with this idempotency_key
    #    SELECT id FROM experience WHERE json_extract(content, '$.structured.idempotency_key') = ?
    # 2. If found, return existing id
    # 3. If not found, call append_experience() and return new id
```

Add query helpers:
```python
def list_task_executions(self, task_id: Optional[str] = None, limit: int = 20) -> List[ExperienceModel]:
    """List TASK_EXECUTION experiences, optionally filtered by task_id."""

def get_by_trace_id(self, trace_id: str) -> Optional[ExperienceModel]:
    """Get experience by trace_id."""
```

### Step 5: Add SQLite Indexes

**File: `src/memory/raw_store.py` or startup script in `app.py`**

Add index creation (idempotent with `IF NOT EXISTS`):
```sql
CREATE INDEX IF NOT EXISTS ix_experiences_type_ts
    ON experience(type, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_experiences_task
    ON experience((json_extract(content, '$.structured.task_id')), created_at DESC);

CREATE INDEX IF NOT EXISTS ix_experiences_task_slug
    ON experience((json_extract(content, '$.structured.task_slug')), ts DESC);

CREATE INDEX IF NOT EXISTS ix_experiences_trace
    ON experience((json_extract(content, '$.structured.trace_id')));

CREATE INDEX IF NOT EXISTS ix_experiences_idempotency
    ON experience((json_extract(content, '$.structured.idempotency_key')));
```

### Step 6: Modify Task Scheduler

**File: `src/services/task_scheduler.py`**

Wrap task execution with:

**Before execution:**
1. Record `started_at` timestamp
2. Generate `trace_id`, `span_id` (both UUIDs)
3. Capture retrieval metadata if task retrieves memories

**After execution (success or failure):**
1. Record `ended_at` timestamp
2. Calculate `duration_ms`
3. Build task execution data structure
4. Call `create_task_execution_experience()` helper
5. Call `raw_store.append_experience_idempotent()`
6. Log creation to logger

**On failure:**
- Capture exception type, message, stack hash
- Set status="failed", populate error field
- Still create experience

**Retrieval tracking:**
- If task calls retrieval service, capture:
  - memory_count
  - query text
  - filters
  - latency_ms
  - parent experience IDs

**Side-effects tracking:**
- Track files written (if task writes to persona_space)
- Track tool calls (if any)
- Track script runs (if any)

### Step 7: Create Backfill Script

**File: `scripts/backfill_task_executions.py` (NEW)**

Script to convert existing task result JSONs to TASK_EXECUTION experiences:
1. Scan `persona_space/tasks/results/*.json`
2. Parse each result file
3. Create TASK_EXECUTION experience with `meta.backfilled=true`
4. Honor idempotency_key (no duplicates on re-run)
5. Support dry-run mode with counts before actual insert
6. Use idempotent insert (safe to re-run)
7. Log: `legacy_result_file → experience_id`

### Step 8: Update Documentation

**File: `docs/SYSTEM_ARCHITECTURE.md`**

Add new section under "Memory Systems":

```markdown
### Task Execution Tracking

Task executions are stored as first-class TASK_EXECUTION experiences in the raw store.

**Structure:**
- Type: `TASK_EXECUTION`
- Parents: Retrieved memories used as context
- Causes: Goals or prior tasks that triggered execution (Phase 4)
- Correlation: trace_id, span_id for distributed tracing
- Full execution status: success/failed/partial with timing
- Resource tracking: files written, tool calls, script runs
- Retrieval provenance: what memories were retrieved and how

**Querying task history:**

```python
# Last 20 runs of specific task
task_runs = raw_store.list_task_executions(task_id="daily_reflection", limit=20)

# Get specific execution by trace_id
execution = raw_store.get_by_trace_id("550e8400-...")

# Show DAG: parents → this task → effects
for parent_id in execution.parents:
    parent = raw_store.get(parent_id)
    print(f"Input: {parent.id}")
```

**Idempotency:**
Re-running a task with the same schedule window won't create duplicate experiences.
```

## Concrete Field Contract (COPY THIS)

This is the exact structure you must create:

```python
{
  "id": "task_exec_{task_id}_{timestamp}_{short_uuid}",
  "type": "TASK_EXECUTION",
  "content": {
    "text": "<task response or summary>",  # PII scrubbed
    "structured": {
      # Schema versioning
      "schema_version": 1,  # TASK_EXECUTION schema version

      # Core identity
      "task_id": "daily_reflection",  # UUID or stable ID
      "task_slug": "daily_reflection",  # Human-readable for queries
      "task_name": "Daily Self-Reflection",
      "task_type": "reflection",  # reflection|assessment|ingest|custom
      "scheduled_vs_manual": "scheduled",

      # Status with dual timestamps
      "status": "success",  # success|failed|partial
      "started_at_iso": "2025-11-04T18:29:12Z",  # ISO string
      "ended_at_iso": "2025-11-04T18:29:13Z",
      "started_at_ts": 1762284552.123,  # Float for SQL
      "ended_at_ts": 1762284553.165,
      "duration_ms": 1042,  # max(ended - started, 0)

      # Correlation
      "trace_id": "uuid4",  # Stable across retries
      "span_id": "uuid4",  # Unique per attempt
      "attempt": 1,
      "retry_of": null,  # span_id (not trace_id) of prior attempt
      "idempotency_key": "sha256(task_id+scheduled_at+attempt)",

      # Config provenance
      "task_config_digest": "sha256:abc123...",

      # Retrieval
      "retrieval": {
        "memory_count": 3,
        "query": "text query used",
        "filters": {"type": "OBSERVATION"},
        "latency_ms": 28,
        "source": ["experiences", "beliefs"]
      },

      # Side effects (ensure IDs reference real tables)
      "io": {
        "files_written": ["path/to/file.json"],
        "artifact_ids": [],
        "tool_calls": [],  # Ref to tool_call table
        "script_runs": []  # Ref to script_run table
      },

      # Embeddings flag
      "embedding_skipped": true,  # Phase 1: always true

      # Scheduler identity
      "provenance_runner": {
        "name": "scheduler",
        "version": "1.0.0",
        "pid": 351425,
        "host": "d-vps"
      },

      # Permissions (stub for now)
      "grants": {
        "scopes": [],
        "impersonation": false
      },

      # Error (non-null on failure)
      "error": null  # or {"type": "ValueError", "message": "...", "stack_hash": "abc123"}
    }
  },
  "provenance": {
    "actor": "AGENT",
    "method": "SCHEDULED_TASK",
    "sources": []
  },
  "parents": ["exp_123", "exp_456"],  # Retrieved memories
  "causes": [],  # Empty for Phase 1, used in Phase 4
  "session_id": "task:daily_reflection:2025-11-04T18:29:12Z:9f1c",
  "created_at": "2025-11-04T18:29:12Z",
  "ts": 1730744952.123,
  "meta": {"scrubbed": true}
}
```

## Testing Checklist (ALL MUST PASS)

### Invariants:
- [ ] Creating TASK_EXECUTION with same idempotency_key is no-op (returns existing id)
- [ ] If status="success", duration_ms >= 0 and ended_at_ts >= started_at_ts
- [ ] If status="failed", error.type and error.message are non-null
- [ ] If retrieval.memory_count > 0, len(parents) == memory_count, else both zero
- [ ] trace_id constant across retries, span_id unique per attempt
- [ ] attempt >= 1, and attempt == 1 when retry_of is null
- [ ] retry_of holds span_id (not trace_id) of prior attempt
- [ ] idempotency_key uniqueness enforced by insert path
- [ ] trace_id and span_id are valid UUIDv4
- [ ] tool_calls and script_runs reference existing tables or are empty

### Test Cases:
1. [ ] Execute scheduled task → verify success experience created
2. [ ] Force task failure (inject exception) → verify failed experience with error populated
3. [ ] Re-run with same schedule → verify no duplicate (idempotent)
4. [ ] Verify parents length matches retrieval.memory_count
5. [ ] Verify trace_id, span_id, attempt all populated
6. [ ] Query by task_id → returns all runs of that task
7. [ ] Query by trace_id → returns specific execution
8. [ ] Verify indexes exist (check with `.indexes` on sqlite connection)
9. [ ] Verify PII scrubbing ran (meta.scrubbed=true)
10. [ ] Run backfill script → verify creates experiences for existing results

### Query Examples to Test:
```python
# Get last 20 task executions
exps = raw_store.list_recent(limit=20, experience_type=ExperienceType.TASK_EXECUTION)

# Get all runs of daily_reflection
runs = raw_store.list_task_executions(task_id="daily_reflection", limit=20)

# Get specific execution
exec = raw_store.get_by_trace_id("550e8400-e29b-41d4-a716-446655440000")

# Verify index usage (should use index, not full scan)
# Run EXPLAIN QUERY PLAN on above queries
```

### Ready-to-Run Check Commands:

After implementation, verify with these commands:

**1. Run one scheduled task:**
```bash
curl -s -X POST http://172.239.66.45:8000/api/tasks/run \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"daily_reflection","mode":"scheduled"}' >/dev/null
```

**2. Verify experience creation:**
```bash
python - <<'PY'
from src.memory.raw_store import RawStore
rs = RawStore()
exps = rs.list_recent(limit=5)
print([(e.type, e.content.structured.get("task_id"), e.content.structured.get("status")) for e in exps])
PY
```

**3. Query by task_id (SQLite):**
```bash
sqlite3 data/core.db \
'SELECT ts, json_extract(content,"$.structured.task_id"),
        json_extract(content,"$.structured.status"),
        json_extract(content,"$.structured.duration_ms")
 FROM experience
 WHERE type="TASK_EXECUTION"
 ORDER BY ts DESC LIMIT 10;'
```

**4. Verify parents length equals retrieval.memory_count:**
```bash
sqlite3 data/core.db \
'SELECT json_extract(content,"$.structured.retrieval.memory_count") AS m,
        json_array_length(parents) AS p
 FROM experience
 WHERE type="TASK_EXECUTION"
 ORDER BY ts DESC LIMIT 10;'
```

All queries should return valid data with no errors.

## Important Constraints

1. **DO NOT break existing task functionality** - tasks must still execute and store result JSONs
2. **Backwards compatibility** - keep existing result JSON files for now
3. **Follow existing patterns** - use same patterns as ingestion pipeline
4. **PII scrubbing** - use existing scrubber from identity ledger
5. **Error handling** - always create experience, even on failure
6. **Idempotency** - safe to re-run task scheduler without creating duplicates

## Files Summary

### Create:
- `src/pipeline/task_experience.py` - Helper to build TASK_EXECUTION experiences
- `scripts/backfill_task_executions.py` - Backfill script for existing results

### Modify:
- `src/memory/models.py` - Add TASK_EXECUTION, SCHEDULED_TASK, MANUAL_TASK, causes field
- `src/memory/raw_store.py` - Add idempotent insert, query helpers, indexes
- `src/services/task_scheduler.py` - Wrap execution, create experiences
- `docs/SYSTEM_ARCHITECTURE.md` - Document task execution tracking

## Success Criteria

After implementation, you should be able to:

1. ✅ **Trace any task execution** → query by task_id or trace_id
2. ✅ **See full context** → parents show what memories informed the task
3. ✅ **Track side effects** → io.files_written shows what was created
4. ✅ **Handle failures gracefully** → failed tasks create experiences with error details
5. ✅ **Query efficiently** → indexes make queries fast
6. ✅ **Prevent duplicates** → idempotency_key prevents duplicate executions
7. ✅ **Backfill history** → script converts old results to experiences

## Next Steps After Phase 1

Once Phase 1 is complete and tested:
- Phase 2: Add task lifecycle events to identity ledger
- Phase 3: Implement unified trace context that flows through all systems
- Phase 4: Build goal model and link tasks to goals
- Phase 5: Create post-task hooks for automatic self-claim extraction
- Phase 6: Build task quality analytics

## Questions to Consider During Implementation

- How to compute idempotency_key? (sha256 of task_id + scheduled_at + attempt)
- How to handle tasks that don't retrieve memories? (empty parents, memory_count=0)
- Should embeddings be created for task execution text? (No, not in Phase 1)
- How to capture files_written? (track in task execution, compare filesystem before/after)
- Where to run indexes? (on startup in app.py or in raw_store.__init__)

---

**Create branch**: `feature/task-execution-tracking`

**Start by reading** `.claude/tasks/task-007-end-to-end-tracking.md` for full context

**When complete, commit**: "Implement Phase 1: Task execution tracking with full correlation and idempotency"

**Run the test suite** and verify all invariants pass before committing.

Good luck! This is surgical work - follow the spec exactly. 🎯
