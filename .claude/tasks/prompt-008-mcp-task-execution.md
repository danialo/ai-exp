# Task 008: MCP Server for Task Execution Tracking

**Status**: Ready for implementation
**Branch**: `feature/mcp-task-execution`
**Depends on**: Phase 1 (feature/task-execution-tracking) - COMPLETE
**Priority**: High
**Estimated effort**: 4-6 hours

## Executive Summary

Expose task execution tracking data via Model Context Protocol (MCP) server to enable:
- LLM-assisted debugging of task failures
- Interactive exploration of task execution history
- Pattern detection across task executions
- Integration with Claude Desktop and other MCP clients

## Current State (Phase 1 Complete)

✅ **Infrastructure in place:**
- `TASK_EXECUTION` experience type with full metadata
- SQLite indexes: type+timestamp, task_id, task_slug, trace_id, idempotency_key
- `RawStore` query methods: `list_task_executions()`, `get_by_trace_id()`
- Backfill script for historical data
- SQL probe queries for monitoring
- 6/6 tests passing

✅ **Data available:**
- Task execution status (success/failed)
- Duration metrics (started_at, ended_at, duration_ms)
- Error details (type, message, stack_hash)
- Correlation IDs (trace_id, span_id)
- Retrieval metadata (memory counts, sources)
- I/O tracking (files_written)
- Idempotency keys for deduplication

## Implementation Scope

### Phase 2A: MCP Server Setup (This Task)

Implement MCP server with three core tools:

1. **`tasks_list`** - Query task executions with filters
2. **`tasks_by_trace`** - Retrieve execution by correlation ID
3. **`tasks_last_failed`** - Get recent failures for debugging

### Out of Scope (Future)

- MCP tools for beliefs, memories, identity (separate tasks)
- Real-time streaming of task executions
- Task retry/rerun capabilities via MCP
- Integration with awareness loop (Task 009)

## MCP Tool Specifications

### Tool 1: `tasks_list`

**Purpose**: Query task execution history with filtering

**Input Schema**:
```json
{
  "task_id": "string (optional)",
  "status": "success | failed (optional)",
  "limit": "integer (default: 20, max: 100)",
  "since": "ISO 8601 timestamp (optional)",
  "backfilled": "boolean (optional) - filter by backfilled flag"
}
```

**Output Schema**:
```json
{
  "executions": [
    {
      "id": "string",
      "task_id": "string",
      "task_slug": "string",
      "task_name": "string",
      "status": "success | failed",
      "started_at": "ISO 8601",
      "ended_at": "ISO 8601",
      "duration_ms": "integer",
      "trace_id": "string",
      "span_id": "string",
      "attempt": "integer",
      "error": {
        "type": "string",
        "message": "string",
        "stack_hash": "string"
      } | null,
      "retrieval": {
        "memory_count": "integer",
        "source": ["string"]
      },
      "io": {
        "files_written": ["string"]
      },
      "backfilled": "boolean"
    }
  ],
  "total": "integer",
  "has_more": "boolean"
}
```

**Implementation Notes**:
- Use `raw_store.list_task_executions(task_id, limit)`
- Add filtering in SQL for status, since, backfilled
- Parse structured content from experiences
- Sort by created_at DESC (most recent first)

**Example Usage**:
```python
# Get all failed tasks in last 24 hours
tasks_list(status="failed", since="2025-11-04T00:00:00Z", limit=50)

# Get executions for specific task
tasks_list(task_id="daily_reflection", limit=10)

# Get only live (non-backfilled) executions
tasks_list(backfilled=false, limit=20)
```

### Tool 2: `tasks_by_trace`

**Purpose**: Retrieve task execution by correlation ID (useful for debugging retries)

**Input Schema**:
```json
{
  "trace_id": "string (required)"
}
```

**Output Schema**:
```json
{
  "executions": [
    {
      // Same schema as tasks_list executions
      // Multiple entries if task was retried (same trace_id, different span_id)
    }
  ],
  "retry_count": "integer",
  "final_status": "success | failed"
}
```

**Implementation Notes**:
- Use `raw_store.get_by_trace_id(trace_id)`
- May return multiple experiences if task was retried
- Sort by attempt number
- Identify final status from highest attempt

**Example Usage**:
```python
# Debug a specific task execution trace
tasks_by_trace(trace_id="550e8400-e29b-41d4-a716-446655440000")
```

### Tool 3: `tasks_last_failed`

**Purpose**: Get recent failures for debugging (sorted by recency)

**Input Schema**:
```json
{
  "limit": "integer (default: 10, max: 50)",
  "task_id": "string (optional) - filter by task",
  "unique_errors": "boolean (default: false) - dedupe by stack_hash"
}
```

**Output Schema**:
```json
{
  "failures": [
    {
      // Same schema as tasks_list executions
      // Only failed tasks
    }
  ],
  "error_patterns": [
    {
      "stack_hash": "string",
      "count": "integer",
      "first_seen": "ISO 8601",
      "last_seen": "ISO 8601",
      "example_task_id": "string",
      "example_trace_id": "string"
    }
  ]
}
```

**Implementation Notes**:
- Query WHERE status = 'failed' ORDER BY created_at DESC
- If unique_errors=true, dedupe by stack_hash (keep most recent)
- Include error_patterns summary (group by stack_hash)
- This is the primary debugging tool

**Example Usage**:
```python
# Get last 10 failures across all tasks
tasks_last_failed(limit=10)

# Get unique error patterns (one per stack_hash)
tasks_last_failed(limit=20, unique_errors=true)

# Debug specific task failures
tasks_last_failed(task_id="belief_gardener", limit=5)
```

## File Structure

```
src/
  mcp/
    __init__.py
    server.py          # MCP server initialization
    tools/
      __init__.py
      task_tools.py    # Implementation of 3 tools above
    schemas.py         # Pydantic models for MCP I/O

scripts/
  start_mcp_server.py  # CLI entry point

tests/
  test_mcp_task_tools.py  # Test suite for MCP tools

docs/
  MCP_INTEGRATION.md      # Usage documentation
```

## Implementation Approach

### Architecture: MCP Sidecar Pattern

Use a **read-only adapter** pattern to keep MCP server separate from main Astra API:

```
MCP Client (Claude Desktop)
    ↓ stdio
MCP Server (mcp_sidecar/server.py)
    ↓ HTTP adapter
Astra API (app.py) - new endpoints
    ↓ SQL
RawStore (raw_store.db)
```

### Implementation Steps

## Step 1: Wire Adapter to Astra API

Add 3 new API endpoints to `app.py`:

```python
# app.py - Task Execution Tracking API

@app.get("/api/tasks/{task_id}/executions")
async def get_task_executions(task_id: str, limit: int = 50):
    """List task executions with filtering.

    MCP Tool: astra.tasks.list
    """
    # Query raw_store using existing methods
    # Return JSON matching schema

@app.get("/api/tasks/by-trace/{trace_id}")
async def get_task_by_trace(trace_id: str):
    """Retrieve task execution by correlation ID.

    MCP Tool: astra.tasks.by_trace
    """
    # Use raw_store.get_by_trace_id()
    # Calculate retry_count and final_status
    # Return JSON matching schema

@app.get("/api/tasks/{task_id}/last-failed")
async def get_last_failed_tasks(
    task_id: str,
    unique_errors: bool = False,
    limit: int = 10
):
    """Get recent task failures for debugging.

    MCP Tool: astra.tasks.last_failed
    """
    # Query WHERE status='failed'
    # Dedupe by stack_hash if unique_errors=true
    # Generate error_patterns summary
    # Return JSON matching schema
```

**API Mapping**:
- `tasks_list(task_id, limit)` → `GET /api/tasks/{task_id}/executions?limit=N`
- `tasks_by_trace(trace_id)` → `GET /api/tasks/by-trace/{trace_id}`
- `tasks_last_failed(task_id, unique_errors)` → `GET /api/tasks/{task_id}/last-failed?unique_errors=1`

## Step 2: Implement MCP Tools

Create `mcp_sidecar/` directory with structure:

```
mcp_sidecar/
  __init__.py
  server.py              # MCP server with 3 tools + metrics
  adapters/
    __init__.py
    astra_ro.py          # HTTP adapter with guards
  contracts/
    task_execution.schema.json
    task_by_trace.schema.json
    task_last_failed.schema.json
  tests/
    __init__.py
    test_mcp_tools.py
  README.md
  smoke_test.sh
```

**adapter (adapters/astra_ro.py)**:
```python
import httpx

class AstraRO:
    """Read-only HTTP client for Astra API."""

    def __init__(self, base="http://localhost:8000", timeout=2.0):
        self.client = httpx.Client(base_url=base, timeout=timeout)

    def tasks_list(self, task_id: str, limit: int = 50):
        r = self.client.get(f"/api/tasks/{task_id}/executions",
                           params={"limit": limit})
        r.raise_for_status()
        return r.json()

    def tasks_by_trace(self, trace_id: str):
        r = self.client.get(f"/api/tasks/by-trace/{trace_id}")
        r.raise_for_status()
        return r.json()

    def tasks_last_failed(self, task_id: str, unique_errors: bool):
        r = self.client.get(f"/api/tasks/{task_id}/last-failed",
                           params={"unique_errors": int(unique_errors)})
        r.raise_for_status()
        return r.json()
```

**server (server.py)**:
```python
import asyncio
import os
from mcp.server import Server
from adapters.astra_ro import AstraRO

srv = Server(name="astra-mcp")
ro = AstraRO(os.getenv("ASTRA_API", "http://localhost:8000"))

@srv.tool()
async def astra_tasks_list(task_id: str, limit: int = 50):
    """List task executions with filtering."""
    return ro.tasks_list(task_id, limit)

@srv.tool()
async def astra_tasks_by_trace(trace_id: str):
    """Retrieve execution by correlation ID."""
    return ro.tasks_by_trace(trace_id)

@srv.tool()
async def astra_tasks_last_failed(
    task_id: str,
    unique_errors: bool = True
):
    """Get recent failures for debugging."""
    return ro.tasks_last_failed(task_id, unique_errors)

async def main():
    await srv.run_stdio()

if __name__ == "__main__":
    asyncio.run(main())
```

**Return exactly the JSON from your schemas. Do not reshape.**

## Step 3: Guards

Implement in `AstraRO` class:

### Rate Limiting
- **Limit**: 10 qps per tool
- **Burst**: 20
- **Implementation**: Token bucket algorithm
- **Behavior**: Raise exception if exceeded

### Timeouts
- **Upstream**: 2s (Astra API)
- **MCP Call**: 30s cap (total including retries)
- **Implementation**: httpx.Timeout

### Response Size
- **Max**: 1 MB
- **Behavior**: Truncate arrays, add `truncated=true` flag
- **Implementation**: Check serialized JSON size, slice arrays if needed

```python
class AstraRO:
    MAX_RESPONSE_SIZE = 1024 * 1024  # 1 MB
    UPSTREAM_TIMEOUT = 2.0

    def _truncate_if_needed(self, data: dict) -> dict:
        import json
        if len(json.dumps(data)) > self.MAX_RESPONSE_SIZE:
            # Truncate executions/failures arrays
            data["truncated"] = True
            data["original_count"] = data.get("total", len(data.get("executions", [])))
        return data
```

## Step 4: Observability

Add structured logging and metrics to `server.py`:

**Counters**:
- `mcp_calls_total{tool=}`
- `mcp_errors_total{tool=}`

**Histograms**:
- `mcp_latency_ms_bucket{tool=}`

**Log Fields**:
- `tool`: Tool name
- `args_hash`: SHA-256 of arguments (privacy)
- `http_status`: Upstream status
- `rows`: Result count
- `latency_ms`: Execution time

```python
import logging
import time
import hashlib

logger = logging.getLogger(__name__)

@srv.tool()
async def astra_tasks_list(task_id: str, limit: int = 50):
    start = time.time()
    args_hash = hashlib.sha256(f"{task_id}:{limit}".encode()).hexdigest()[:8]

    try:
        logger.info(f"[astra.tasks.list] args_hash={args_hash} task_id={task_id} limit={limit}")
        result = ro.tasks_list(task_id, limit)
        latency_ms = (time.time() - start) * 1000
        logger.info(f"[astra.tasks.list] SUCCESS rows={result['total']} latency_ms={latency_ms:.2f}")
        return result
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.error(f"[astra.tasks.list] ERROR error={str(e)} latency_ms={latency_ms:.2f}")
        raise
```

## Step 5: Tests

Create `tests/test_mcp_tools.py` with:

### Schema Validation
```python
def test_tasks_list_schema(mcp_client, schema_loader):
    res = mcp_client.call("astra.tasks.list", {"task_id": "demo", "limit": 5})
    schema_loader("task_execution.schema.json").validate(res)
```

### Golden Tests
- Verify exact output format for each tool
- Use mock HTTP responses with realistic data

### Error Paths
- `404 task_id` → return empty list
- `404 trace_id` → return empty result
- `upstream 500` → propagate error with logging

### Performance
- List 1k executions under 200ms with pagination
- Use `pytest-benchmark` for latency tests

## Step 6: Smoke Script

Create `smoke_test.sh`:

```bash
#!/bin/bash
# Start Astra API
# Start MCP server in background
# Test 3 tools via mcp-cli or stdio
# Verify responses match schemas
# Check error handling (404s)
# Stop servers

ASTRA_API=http://localhost:8000 python -m server &
mcp-cli call astra.tasks.list '{"task_id":"daily_reflection","limit":5}'
mcp-cli call astra.tasks.last_failed '{"task_id":"idea_generation","unique_errors":true}'
```

## Roll-In Checklist for This Branch

- [ ] **Create branch**: `git checkout -b feature/mcp-task-execution`
- [ ] **Add API endpoints** to `app.py` (Step 1)
- [ ] **Create mcp_sidecar/** structure (Step 2)
- [ ] **Implement adapter** with rate limits and guards (Step 3)
- [ ] **Implement MCP server** with 3 tools + observability (Step 4)
- [ ] **Drop contracts/*.json** schemas into `mcp_sidecar/contracts/`
- [ ] **Add tests** with schema validation (Step 5)
- [ ] **Create smoke_test.sh** and README (Step 6)
- [ ] **Update STATUS-PHASE-1-COMPLETE.md** with MCP sidecar note
- [ ] **Run tests**: `pytest mcp_sidecar/tests/ -v`
- [ ] **Run smoke test**: `bash mcp_sidecar/smoke_test.sh`
- [ ] **Commit changes** with descriptive message
- [ ] **Update PR description** with MCP integration details

## Configuration

No settings.py changes required. Use environment variable:

```bash
# .env or shell
export ASTRA_API=http://localhost:8000  # Astra API base URL
```

For Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "astra": {
      "command": "python",
      "args": ["-m", "server"],
      "cwd": "/path/to/ai-exp/mcp_sidecar",
      "env": {
        "ASTRA_API": "http://localhost:8000"
      }
    }
  }
}
```

## Testing Requirements

### Unit Tests (tests/test_mcp_task_tools.py)
- [ ] Test each tool with valid inputs
- [ ] Test each tool with invalid inputs (error handling)
- [ ] Test filtering logic (status, since, backfilled)
- [ ] Test pagination and limits
- [ ] Test deduplication (unique_errors)
- [ ] Test empty result sets

### Integration Tests
- [ ] Test with backfilled data from Phase 1
- [ ] Test with live task execution data
- [ ] Test concurrent tool calls
- [ ] Test large result sets (1000+ executions)

### Manual Testing Checklist
- [ ] Connect from Claude Desktop
- [ ] Execute each tool via MCP client
- [ ] Verify output matches SQL probe queries
- [ ] Test debugging workflow:
  1. `tasks_last_failed()` → identify pattern
  2. `tasks_by_trace(trace_id)` → inspect specific execution
  3. `tasks_list(task_id, status="failed")` → see all failures for task

## Success Criteria

✅ **Functional Requirements**:
- MCP server starts successfully
- All 3 tools callable from MCP client
- Results match raw_store queries
- Error handling for invalid inputs
- Proper pagination for large result sets

✅ **Quality Requirements**:
- 100% test coverage for tool handlers
- All tests passing (unit + integration)
- Response time < 500ms for typical queries
- Documentation complete and accurate

✅ **Integration Requirements**:
- Works with Claude Desktop MCP client
- No breaking changes to Phase 1 code
- Can run alongside FastAPI server (different port)

## Migration Notes

**Phase 1 → Phase 2A**:
- No schema changes required
- No data migration required
- RawStore methods unchanged
- Purely additive feature

**Deployment**:
1. Merge `feature/task-execution-tracking` to main
2. Create `feature/mcp-task-execution` branch from main
3. Implement MCP server (this task)
4. Test with Claude Desktop
5. Merge to main when complete

## Questions for Implementer

1. **MCP Library**: Use `anthropic-mcp-server` or implement custom SSE server?
2. **Authentication**: Add API key authentication for MCP tools?
3. **Rate Limiting**: Limit queries per minute to prevent abuse?
4. **Caching**: Cache frequently accessed data (e.g., error patterns)?

## Example Debugging Workflow

**Scenario**: Belief gardener keeps failing with same error

1. **Identify recent failures**:
   ```
   tasks_last_failed(task_id="belief_gardener", limit=10)
   → Shows 5 failures in last hour, all same stack_hash
   ```

2. **Get unique error pattern**:
   ```
   tasks_last_failed(unique_errors=true)
   → Returns error_patterns showing stack_hash appears 5 times
   ```

3. **Inspect specific execution**:
   ```
   tasks_by_trace(trace_id="abc123...")
   → Shows execution details, error message, retrieval metadata
   ```

4. **Query all executions for this task**:
   ```
   tasks_list(task_id="belief_gardener", limit=50)
   → Compare successful vs failed executions
   → Notice failures only occur when memory_count > 100
   ```

5. **Root cause identified**: Memory retrieval overflow causing JSON parse error

## Dependencies

**New Python packages**:
```bash
pip install mcp>=0.1.0
# Or implement using SSE (Server-Sent Events) directly
```

**Existing dependencies** (no changes):
- src.memory.raw_store (RawStore)
- src.memory.models (ExperienceModel, ExperienceType)
- config.settings (Settings)

## Metrics to Track

Once deployed, monitor:
- MCP tool call frequency (which tools used most)
- Average response times per tool
- Error rates (invalid inputs, server errors)
- Result set sizes (detect queries needing optimization)

---

## Appendix: MCP Tool Registration

Example server setup:
```python
from mcp import Server, Tool
from src.mcp.tools.task_tools import (
    handle_tasks_list,
    handle_tasks_by_trace,
    handle_tasks_last_failed
)

server = Server("astra-task-execution")

server.add_tool(
    Tool(
        name="tasks_list",
        description="Query task execution history with filtering",
        input_schema=TasksListInput,
        handler=handle_tasks_list
    )
)

server.add_tool(
    Tool(
        name="tasks_by_trace",
        description="Retrieve task execution by correlation ID",
        input_schema=TasksByTraceInput,
        handler=handle_tasks_by_trace
    )
)

server.add_tool(
    Tool(
        name="tasks_last_failed",
        description="Get recent task failures for debugging",
        input_schema=TasksLastFailedInput,
        handler=handle_tasks_last_failed
    )
)

if __name__ == "__main__":
    server.run(host="localhost", port=8001)
```

---

**Task created**: 2025-11-05 14:15 UTC
**Assignee**: Codex
**Reviewer**: Quantum Tsar of Arrays (for API design)
**Target completion**: Phase 2A complete within 1 week
