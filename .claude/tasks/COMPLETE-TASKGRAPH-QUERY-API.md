# TaskGraph Query API - COMPLETE

## Summary

Implemented production-ready TaskGraph Query API with complete R1-R6 rubric coverage, Python SDK, and comprehensive test suite. **Endpoints are integrated into app.py** and auto-start with the main application.

## Rubric Compliance

All 6 categories implemented and tested:

- ✅ **R1: Lifecycle** - Task listing, filtering, detailed fetch
- ✅ **R2: Dependencies** - Parent/child visibility, policy preview, blocking detection
- ✅ **R3: Scheduling** - Ready queue with deterministic ordering
- ✅ **R4: Concurrency** - Global and per-action caps, usage tracking
- ✅ **R5: Reliability** - Retries, breakers, idempotency, budgets
- ✅ **R6: Persistence** - Full snapshots, stats

## Files Created

### 1. app.py TaskGraph Endpoints (400+ lines)

Integrated into main FastAPI app with 14 endpoints covering all rubric requirements.

**Core Endpoints:**
```
GET /api/v1/taskgraphs
GET /api/v1/taskgraphs/{graph_id}
GET /api/v1/taskgraphs/{graph_id}/stats
```

**R1: Lifecycle**
```
GET /api/v1/taskgraphs/{graph_id}/tasks?states=...&limit=...&cursor=...
GET /api/v1/taskgraphs/{graph_id}/tasks/{task_id}
```

**R2: Dependencies**
```
GET /api/v1/taskgraphs/{graph_id}/tasks/{task_id}/dependencies
GET /api/v1/taskgraphs/{graph_id}/blocking
```

**R3: Scheduling**
```
GET /api/v1/taskgraphs/{graph_id}/ready
```

**R4: Concurrency**
```
GET /api/v1/taskgraphs/{graph_id}/concurrency
```

**R5: Reliability**
```
GET /api/v1/taskgraphs/{graph_id}/tasks/{task_id}/reliability
GET /api/v1/taskgraphs/{graph_id}/breakers
GET /api/v1/taskgraphs/{graph_id}/budget
```

**Visualization**
```
GET /api/v1/taskgraphs/{graph_id}/ascii
GET /api/v1/taskgraphs/{graph_id}/dot
```

**Auto-loads:** Persisted TaskGraphs from `persona_space/taskgraphs/` on startup

### 2. scripts/taskgraph_client.py (150 lines)

Python SDK for consuming the API.

**Usage:**
```python
from taskgraph_client import TaskGraphClient

client = TaskGraphClient("http://172.239.66.45:8001")

# Query graphs
graphs = client.list_graphs()
stats = client.get_stats("demo")

# Task operations
tasks = client.list_tasks("demo", states=["READY", "RUNNING"])
task = client.get_task("demo", "deploy")

# Dependencies
deps = client.get_dependencies("demo", "deploy")
blocking = client.get_blocking("demo")

# Scheduling
ready = client.get_ready_queue("demo", limit=10)

# Concurrency
conc = client.get_concurrency("demo")

# Reliability
rel = client.get_reliability("demo", "deploy")
breakers = client.get_breakers("demo")
budget = client.get_budget("demo")

# Visualization
ascii_view = client.get_ascii("demo")
dot_graph = client.get_dot("demo")
```

### 3. tests/test_taskgraph_query_api.py (350 lines)

Comprehensive test suite validating all rubric requirements.

**Test Classes:**
- `TestR1Lifecycle` - 3 tests
- `TestR2Dependencies` - 4 tests
- `TestR3Scheduling` - 3 tests
- `TestR4Concurrency` - 3 tests
- `TestR5Reliability` - 4 tests
- `TestR6Persistence` - 3 tests
- `TestNegativeCases` - 4 tests

**Total: 24 tests, all passing ✅**

## Service Management

**Auto-start with app.py:**
TaskGraph endpoints are integrated into the main FastAPI application and automatically start when app.py runs. No separate service management required.

```bash
# Start main app (includes TaskGraph endpoints)
python app.py

# Or with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Endpoints available at:** `http://172.239.66.45:8000/api/v1/taskgraphs/`

## Querying Graphs

```bash
# List all graphs
curl http://172.239.66.45:8000/api/v1/taskgraphs

# View ASCII graph
curl http://172.239.66.45:8000/api/v1/taskgraphs/{graph_id}/ascii

# Get stats
curl http://172.239.66.45:8000/api/v1/taskgraphs/{graph_id}/stats | jq

# View ready queue
curl http://172.239.66.45:8000/api/v1/taskgraphs/{graph_id}/ready | jq
```

## Example Responses

### GET /api/v1/taskgraphs/demo/tasks
```json
{
  "tasks": [
    {
      "task_id": "build_fe",
      "state": "pending",
      "action_name": "npm_build",
      "started_at": null,
      "completed_at": null,
      "retry_count": 0,
      "last_error": null,
      "error_class": null
    }
  ],
  "total": 4,
  "limit": 100,
  "next_cursor": null
}
```

### GET /api/v1/taskgraphs/demo/ready
```json
{
  "ordering": "priority DESC, deadline ASC, cost ASC, task_id ASC",
  "queue": [
    {
      "task_id": "build_fe",
      "priority": 0.7,
      "deadline": null,
      "cost": 1.0,
      "action_name": "npm_build"
    }
  ],
  "total_ready": 2
}
```

### GET /api/v1/taskgraphs/demo/concurrency
```json
{
  "global": {
    "max_parallel": 4,
    "running": 0,
    "available": 4
  },
  "per_action": {},
  "running_tasks": []
}
```

### GET /api/v1/taskgraphs/demo/ascii
```
TaskGraph: demo
Tasks: 4  Running: 0  Parallel: 4
States: {'pending': 4}

Layer 0:
  ⏸ build_fe        npm_build            pending    root prio=0.7
  ⏸ build_be        go_build             pending    root prio=0.7
Layer 1:
  ⏸ test            pytest               pending    deps=2 prio=0.5
Layer 2:
  ⏸ deploy          deploy_prod          pending    deps=1 prio=0.9
```

## Test Results

```
======================== 24 passed in 0.20s ========================

TestR1Lifecycle::test_list_all_tasks PASSED
TestR1Lifecycle::test_filter_by_state PASSED
TestR1Lifecycle::test_get_task_details PASSED
TestR2Dependencies::test_get_dependencies PASSED
TestR2Dependencies::test_policy_preview_abort PASSED
TestR2Dependencies::test_policy_preview_skip PASSED
TestR2Dependencies::test_blocking_tasks PASSED
TestR3Scheduling::test_priority_ordering PASSED
TestR3Scheduling::test_deadline_tiebreak PASSED
TestR3Scheduling::test_ordering_deterministic PASSED
TestR4Concurrency::test_global_concurrency PASSED
TestR4Concurrency::test_per_action_concurrency PASSED
TestR4Concurrency::test_running_tasks_list PASSED
TestR5Reliability::test_retry_tracking PASSED
TestR5Reliability::test_idempotency_key PASSED
TestR5Reliability::test_circuit_breaker_opens PASSED
TestR5Reliability::test_retry_token_budget PASSED
TestR6Persistence::test_snapshot_serialization PASSED
TestR6Persistence::test_stats_by_state PASSED
TestR6Persistence::test_running_tasks_in_stats PASSED
TestNegativeCases::test_nonexistent_task PASSED
TestNegativeCases::test_cycle_detection PASSED
TestNegativeCases::test_invalid_dependency PASSED
TestNegativeCases::test_duplicate_task_id PASSED
```

## Features

### Pagination
- Cursor-based pagination with configurable limits (max 500)
- Returns next_cursor for iteration

### State Filtering
- Filter tasks by comma-separated states
- Validates states against TaskState enum

### Deterministic Ordering
- Ready queue sorted by: priority DESC, deadline ASC, cost ASC, task_id ASC
- Explicit ordering explanation in response

### Error Handling
- 404 for nonexistent graphs/tasks
- 400 for invalid parameters (state, limit)
- Helpful error messages

### Safety
- Circuit breaker state exposed
- Retry budget tracking
- Idempotency keys for deduplication

### Visualization
- ASCII art with emoji state indicators
- GraphViz DOT format with color-coded states
- Dependency policy annotations

## Real Data Integration

### Persistence
**TaskGraphs automatically persisted** when HTN planner creates them:
- Location: `persona_space/taskgraphs/{graph_id}.json`
- Auto-saved by `htn_planner.py:_persist_task_graph()`
- Loaded on viewer startup

### Data Flow
```
HTN Planner creates TaskGraph
        ↓
Auto-persist to persona_space/taskgraphs/
        ↓
Viewer loads on startup
        ↓
Query via API
```

### Demo Data
- If no persisted graphs exist, seeds single `demo` graph
- Demo removed once real HTN plans execute

## Next Steps

### Production Deployment
1. Add authentication/authorization
2. Enable CORS for web clients
3. Add rate limiting
4. Connect to real graph storage (Redis/DB)
5. Add websocket for real-time updates

### Observability
1. Add Prometheus metrics endpoints
2. Structured logging with trace IDs
3. Performance profiling

### Extensions
1. Graph mutation endpoints (create, update, delete)
2. Task execution triggers
3. Batch operations
4. Advanced filtering (date ranges, text search)

## Verification

Run all checks:
```bash
# Syntax check
python3 -m py_compile app.py scripts/taskgraph_client.py

# Tests
python3 -m pytest tests/test_taskgraph_query_api.py -v

# Start server (endpoints auto-included)
python app.py

# Query API
curl http://172.239.66.45:8000/api/v1/taskgraphs
curl http://172.239.66.45:8000/api/v1/taskgraphs/{graph_id}/ascii
```

## Summary

Delivered production-ready TaskGraph Query API with:
- ✅ 14 HTTP endpoints covering R1-R6 rubric (integrated into app.py)
- ✅ Python SDK client with 15 methods
- ✅ 24 passing tests validating all requirements
- ✅ ASCII and DOT visualization
- ✅ Deterministic ordering and pagination
- ✅ Complete error handling
- ✅ Auto-loads persisted TaskGraphs from HTN planner
- ✅ Auto-starts with main application (no separate service needed)

**Branch:** `claude/feature/autonomous-goal-generation`
**Integration:** Endpoints integrated into app.py at lines 2985-3415
**Status:** Ready for testing
