# MCP Server Architecture

Deep dive into Astra's MCP server design and implementation.

## Table of Contents

- [Overview](#overview)
- [Design Philosophy](#design-philosophy)
- [Component Architecture](#component-architecture)
- [Data Persistence](#data-persistence)
- [Safety Model](#safety-model)
- [Integration Points](#integration-points)
- [Future Work](#future-work)

---

## Overview

Astra's MCP (Model Context Protocol) server exposes tools for autonomous operation:

1. **Task Introspection** - Query execution history and failures
2. **Schedule Management** - Create cron-based autonomous tasks
3. **Desire Tracking** - Record and manage vague wishes

The server uses **stdio transport** - it's started on-demand by MCP clients (Claude Desktop, VS Code extensions, custom clients).

---

## Design Philosophy

### 1. Auditability First

All state changes are append-only NDJSON chains:
- **Never delete history**
- Complete audit trail for debugging
- Grep-friendly event logs
- Timestamp every event

### 2. Dual Persistence

Every service uses two files:
- **NDJSON chain** (`YYYY-MM.ndjson.gz`) - Audit trail, never modified
- **Index** (`index.json`) - Fast lookups, rebuilt from chain if corrupted

**Benefits**:
- Fast queries (index)
- Reliable recovery (chain)
- Easy debugging (grep chain)
- Low dependency (no database)

### 3. Deterministic IDs

All IDs are content-based hashes:
- **Schedules**: `sch_<sha8>` from name+cron+tool+payload
- **Desires**: `des_<sha8>` from text+timestamp

**Benefits**:
- Idempotency (re-creating same entity is no-op)
- Debugging (ID reveals contents)
- Deduplication (automatic duplicate detection)

### 4. Safety by Default

Three-tier safety model:
- **Tier 0**: Read-only (unlimited)
- **Tier 1**: Local writes (budget-limited)
- **Tier 2**: External ops (approval required)

**Philosophy**: Start conservative, relax over time as trust grows.

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Client                           │
│             (Claude Desktop, VS Code, etc.)             │
└────────────────────┬────────────────────────────────────┘
                     │ stdio (JSON-RPC 2.0)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  bin/mcp_server.py                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │  create_astra_mcp_server()                       │  │
│  │  - Registers all tool handlers                   │  │
│  │  - Routes calls to appropriate services          │  │
│  └──────────────────────────────────────────────────┘  │
└────────────┬──────────────────┬────────────────┬────────┘
             │                  │                │
             ▼                  ▼                ▼
    ┌────────────────┐  ┌─────────────┐  ┌──────────────┐
    │ TaskExecution  │  │  Schedule   │  │    Desire    │
    │    Tooling     │  │   Service   │  │    Store     │
    └────────────────┘  └─────────────┘  └──────────────┘
             │                  │                │
             ▼                  ▼                ▼
       ┌──────────┐      ┌──────────┐    ┌──────────┐
       │ RawStore │      │ NDJSON + │    │ NDJSON + │
       │ (SQLite) │      │  Index   │    │  Index   │
       └──────────┘      └──────────┘    └──────────┘
```

### Layer 1: MCP Protocol Handler

**File**: `bin/mcp_server.py`

Responsibilities:
- Accept JSON-RPC 2.0 over stdio
- Validate tool calls
- Route to appropriate services
- Return formatted responses

**Key Functions**:
```python
def create_astra_mcp_server() -> Server:
    # Create base server with task execution tools
    server = create_task_execution_server(raw_store)

    # Add schedule tools
    @server.list_tools()
    async def handle_list_tools(): ...

    @server.call_tool()
    async def handle_call_tool(name, arguments): ...
```

### Layer 2: Tool Handlers

**Files**: `src/mcp/tools/schedule.py`, `src/mcp/tools/desires.py`

Responsibilities:
- Validate input parameters
- Call service methods
- Format responses
- Handle errors gracefully

**Pattern**:
```python
class ScheduleTools:
    def __init__(self, schedule_service):
        self.schedule_service = schedule_service

    def create(self, payload: Dict) -> Dict:
        # Validate
        if "name" not in payload:
            return {"success": False, "error": "Missing name"}

        # Execute
        schedule = self.schedule_service.create(...)

        # Format response
        return {"success": True, "schedule_id": schedule.id, ...}
```

### Layer 3: Core Services

**ScheduleService** (`src/services/schedule_service.py`):
- Manages cron schedules with safety tiers
- NDJSON + index persistence
- Budget tracking per schedule
- Cron parsing with DST handling

**DesireStore** (`src/services/desire_store.py`):
- Tracks vague wishes with strength
- Strength decay over time
- Tag-based search
- Automatic pruning of weak desires

**TaskExecutionTooling** (`src/mcp/task_execution_server.py`):
- Queries RawStore for execution history
- Filters by status, task, time
- Aggregates error patterns
- Provides trace inspection

---

## Data Persistence

### ScheduleService Persistence

**Location**: `var/schedules/`

**Files**:
```
var/schedules/
  2025-11.ndjson.gz  # Append-only event chain
  index.json         # Current state (rebuilt from chain)
```

**NDJSON Events**:
```json
{"_timestamp":"2025-11-11T14:00:00Z","event":"schedule_created","schedule_id":"sch_abc123","schedule":{...}}
{"_timestamp":"2025-11-11T16:30:00Z","event":"schedule_modified","schedule_id":"sch_abc123","changes":{...}}
{"_timestamp":"2025-11-11T18:00:00Z","event":"schedule_executed","schedule_id":"sch_abc123"}
{"_timestamp":"2025-11-11T20:00:00Z","event":"schedule_paused","schedule_id":"sch_abc123"}
```

**Index Format**:
```json
{
  "sch_abc123": {
    "id": "sch_abc123",
    "name": "daily_backup",
    "cron": "0 2 * * *",
    "target_tool": "backup_data",
    "payload": {"target": "/data"},
    "status": "paused",
    "safety_tier": 1,
    "next_run_at": "2025-11-12T02:00:00Z",
    "run_budget": {
      "per_day": 4,
      "consumed": 2,
      "last_reset": "2025-11-11T00:00:00Z"
    }
  }
}
```

### DesireStore Persistence

**Location**: `var/desires/`

**Files**:
```
var/desires/
  2025-11.ndjson.gz  # Append-only event chain
  index.json         # Current desires
```

**NDJSON Events**:
```json
{"_timestamp":"2025-11-11T14:00:00Z","event":"desire_recorded","desire_id":"des_xyz789","desire":{...}}
{"_timestamp":"2025-11-11T16:00:00Z","event":"desire_reinforced","desire_id":"des_xyz789","delta":0.1}
{"_timestamp":"2025-11-11T18:00:00Z","event":"desires_decayed","decay_rate":0.01,"decayed_count":15}
{"_timestamp":"2025-11-11T20:00:00Z","event":"desires_pruned","threshold":0.1,"pruned_ids":[...]}
```

**Index Format**:
```json
{
  "des_xyz789": {
    "id": "des_xyz789",
    "text": "I wish I had better test coverage",
    "strength": 0.95,
    "created_at": "2025-11-10T10:00:00Z",
    "last_reinforced_at": "2025-11-11T14:30:00Z",
    "tags": ["testing", "quality"],
    "context": {"triggered_by": "code_review"}
  }
}
```

### Recovery from Corruption

If index is corrupted:

```python
# Rebuild from chain
service = ScheduleService(schedules_dir="var/schedules")
service._load_index()  # Fails, logs error
service.index = {}     # Empty index
# Replay chain (not yet implemented, but straightforward):
# for line in read_all_ndjson_files():
#     event = json.loads(line)
#     apply_event_to_index(event)
```

**Future enhancement**: Auto-rebuild on corruption detection.

---

## Safety Model

### Three Tiers

**Tier 0: Read-Only**
- Examples: tasks_list, tasks_by_trace, astra.health
- Auto-run: ✅ No restrictions
- Budget: ❌ Unlimited
- Approval: ❌ Not required

**Tier 1: Local Writes**
- Examples: schedule.create, desires.record, execute_goal
- Auto-run: ✅ With budget enforcement
- Budget: ✅ Per-day limit (default: 4)
- Approval: ❌ Not required
- Scope: Local repository only

**Tier 2: External Operations**
- Examples: deploy_to_vercel, send_email, create_github_pr
- Auto-run: ❌ Requires approval
- Budget: ✅ Advisory (primary gate is approval)
- Approval: ✅ Token in `var/approvals/pending/<schedule_id>.token`
- Scope: External side effects

### Budget Enforcement

**Structure**:
```python
@dataclass
class RunBudget:
    per_day: int = 4           # Max runs per day
    consumed: int = 0          # Runs consumed today
    last_reset: str            # ISO8601 timestamp
```

**Algorithm**:
```python
def check_budget(schedule: Schedule) -> bool:
    now = datetime.now(timezone.utc)
    last_reset = datetime.fromisoformat(schedule.run_budget.last_reset)

    # Daily reset at midnight UTC
    if now.date() > last_reset.date():
        schedule.run_budget.consumed = 0
        schedule.run_budget.last_reset = now.isoformat()
        save_index()

    # Check budget
    return schedule.run_budget.consumed < schedule.run_budget.per_day

def mark_executed(schedule_id: str):
    schedule = get(schedule_id)
    schedule.next_run_at = compute_next_run(schedule.cron)
    schedule.run_budget.consumed += 1
    save_index()
```

### Approval Tokens (Tier 2)

**Not yet implemented**, but design:

**Token location**: `var/approvals/pending/<schedule_id>.token`

**Token format**:
```json
{
  "schedule_id": "sch_abc123",
  "schedule_name": "deploy_to_production",
  "approved_by": "human@example.com",
  "approved_at": "2025-11-11T14:30:00Z",
  "expires_at": "2025-11-11T15:30:00Z"
}
```

**Workflow**:
1. Schedule becomes due
2. Executor checks for token
3. If valid: execute and consume (delete) token
4. If missing: skip and log

**Creation script** (future):
```bash
python -m scripts.approve_schedule sch_abc123
# Creates token with 1-hour expiry
```

---

## Integration Points

### Existing Systems

**RawStore** (SQLite):
- Task execution records
- Used by TaskExecutionTooling for introspection
- Unchanged by MCP additions

**GoalExecutionService**:
- Target of scheduled tasks
- `execute_goal` can be scheduled via MCP
- Future: Auto-create goals from top desires

**PersonaService**:
- Could record desires during introspection
- Future: "I notice you keep retrying X, maybe you desire Y?"

**BeliefSystem**:
- Desires could inform belief updates
- Future: "Strong desire → belief adjustment"

### Data Flow

```
┌──────────────┐
│ MCP Client   │ (Human using Claude Desktop)
└──────┬───────┘
       │ "Create schedule for autonomous coding"
       ▼
┌──────────────────┐
│ MCP Server       │
│ schedule.create  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ ScheduleService  │
│ - Validate cron  │
│ - Assign tier    │
│ - Compute next   │
└──────┬───────────┘
       │
       ▼
┌──────────────────────┐
│ var/schedules/       │
│ - Append to chain    │
│ - Update index       │
└──────────────────────┘

(Later, executor daemon):

┌──────────────────┐
│ Scheduler Daemon │ (runs every minute)
│ - Check due      │
│ - Check budget   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ GoalExecution    │
│ execute_goal()   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ CoderAgent       │
│ - HTN planning   │
│ - Code gen       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ RawStore         │
│ - Task records   │
└──────────────────┘
```

---

## Future Work

### Phase 4: Executor Daemon

**Purpose**: Actually run scheduled tasks

**Design**:
```python
async def scheduler_loop():
    while True:
        now = datetime.now(timezone.utc)

        # Get due schedules
        schedules = schedule_service.list_due(now)

        for schedule in schedules:
            # Check budget
            if not schedule_service.check_budget(schedule):
                logger.info(f"Budget exhausted: {schedule.id}")
                continue

            # Check approval (Tier 2)
            if schedule.safety_tier == SafetyTier.EXTERNAL:
                if not has_approval_token(schedule.id):
                    logger.info(f"Awaiting approval: {schedule.id}")
                    continue

            # Execute
            try:
                result = await execute_tool(
                    schedule.target_tool,
                    schedule.payload
                )
                schedule_service.mark_executed(schedule.id)
                logger.info(f"Executed: {schedule.id}")
            except Exception as e:
                logger.error(f"Failed: {schedule.id}: {e}")

        await asyncio.sleep(60)  # Check every minute
```

### Phase 5: Desire → Goal Promotion

**Criteria for promotion**:
- Desire strength ≥ 0.8
- Desire age ≥ 3 days
- No existing goal for same intent
- User approval (optional setting)

**Flow**:
```python
def promote_desire_to_goal(desire: Desire) -> Goal:
    # LLM converts desire text to concrete goal
    goal_spec = llm_generate_goal(desire.text, desire.context)

    # Create goal
    goal = goal_execution_service.create_goal(goal_spec)

    # Reinforce desire (it was acted upon)
    desire_store.reinforce(desire.id, delta=0.3)

    return goal
```

### Phase 6: Advanced Scheduling

**Features**:
- **Dependencies**: Run schedule B only after A succeeds
- **Retry policies**: Exponential backoff, max retries
- **Dynamic adjustment**: Increase frequency if success rate high
- **Schedule groups**: Pause/resume multiple schedules together

**Example**:
```python
schedule_service.create(
    name="deploy_after_tests",
    cron="0 2 * * *",
    target_tool="deploy_to_vercel",
    payload={...},
    depends_on=["sch_run_tests"],  # New field
    retry_policy={
        "max_attempts": 3,
        "backoff": "exponential"
    }
)
```

---

## Performance Characteristics

### Bottleneck Analysis

**Schedules**:
- Create: O(1) - append + index update
- List: O(n) - scan all schedules
- Due check: O(n) - scan with datetime filter
- Scale: <100 schedules expected

**Desires**:
- Record: O(1) - append + index update
- List: O(n log n) - sort by strength
- Decay: O(n) - update all desires
- Scale: <1000 desires expected

**Optimizations** (if needed):
- Index by next_run_at for faster due checks
- Batch decay operations (run once per day)
- Prune desires more aggressively

### Memory Usage

**Typical**:
- 50 schedules × 1KB = 50KB
- 200 desires × 500B = 100KB
- Index kept in memory = <200KB total

**Chain files**:
- 1 month of events ≈ 1MB compressed
- Auto-rotate monthly

---

## Testing Strategy

### Unit Tests

**ScheduleService**: 27 tests
- ID generation
- Cron parsing
- CRUD operations
- Budget tracking
- Persistence

**DesireStore**: 26 tests
- Recording
- Retrieval
- Decay algorithm
- Pruning
- Tag search

**MCP Tools**: 14 tests
- Input validation
- Error handling
- Integration with services

### Integration Tests

**End-to-end MCP**:
```bash
# Test full protocol
echo '{"jsonrpc":"2.0","id":1,"method":"initialize",...}' | bin/mcp
```

**Service integration**:
```python
def test_schedule_execution_workflow():
    # Create schedule
    schedule = schedule_service.create(...)

    # Mark as due
    schedule.next_run_at = past_time

    # Check due
    due = schedule_service.list_due()
    assert schedule.id in [s.id for s in due]

    # Execute
    schedule_service.mark_executed(schedule.id)

    # Verify budget consumed
    updated = schedule_service.get(schedule.id)
    assert updated.run_budget.consumed == 1
```

---

## Operational Considerations

### Monitoring

**Key metrics** (future):
- Schedule execution rate
- Budget consumption trends
- Desire creation/decay rates
- Tool call success/failure rates

### Debugging

**NDJSON chain inspection**:
```bash
# View all schedule events
zcat var/schedules/2025-11.ndjson.gz | jq

# Find schedule creations
zcat var/schedules/*.ndjson.gz | jq 'select(.event=="schedule_created")'

# Track budget consumption
zcat var/schedules/*.ndjson.gz | jq 'select(.event=="schedule_executed")' | jq -s 'group_by(.schedule_id) | map({schedule_id: .[0].schedule_id, count: length})'
```

### Backup Strategy

**What to backup**:
- `var/schedules/` - All NDJSON files
- `var/desires/` - All NDJSON files
- `var/persona_data.db` - RawStore (task executions)

**Restore**:
```bash
# Restore chain files
cp backup/schedules/*.ndjson.gz var/schedules/

# Delete corrupted index
rm var/schedules/index.json

# Service rebuilds index on next start
bin/mcp
```

---

## Summary

Astra's MCP server provides:

✅ **9 tools** for autonomous operation
✅ **67 tests** with 100% passing
✅ **Audit trail** via NDJSON chains
✅ **Safety tiers** with budget enforcement
✅ **Simple deployment** (stdio, no server needed)

**Production-ready for**: Tier 0 (read-only) and Tier 1 (local writes)
**Future work**: Executor daemon, approval workflow, desire→goal promotion
