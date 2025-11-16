# MCP Tools Reference

Complete reference for all tools exposed by Astra's MCP server.

## Tool Categories

- [Task Introspection](#task-introspection-tier-0) (4 tools)
- [Schedule Management](#schedule-management-tier-1) (5 tools)
- [Desire Management](#desire-management-tier-1) (3 tools)

---

## Task Introspection (Tier 0)

**Safety**: Read-only, auto-run, unlimited

### `tasks_list`

Query task execution history with filters.

**Input Schema**:
```json
{
  "task_id": "string (optional)",
  "status": "success | failed (optional)",
  "limit": "integer (1-100, default: 20)",
  "since": "ISO8601 timestamp (optional)",
  "backfilled": "boolean (optional)"
}
```

**Output**:
```json
{
  "executions": [
    {
      "id": "exp_abc123",
      "task_id": "task_xyz",
      "task_slug": "write_tests",
      "status": "success",
      "started_at": "2025-11-11T14:30:00Z",
      "ended_at": "2025-11-11T14:32:15Z",
      "duration_ms": 135000,
      "trace_id": "trace_123",
      "attempt": 1,
      "error": null
    }
  ],
  "total": 42,
  "has_more": true
}
```

**Example**:
```json
{"name": "tasks_list", "arguments": {"status": "failed", "limit": 10}}
```

---

### `tasks_by_trace`

Inspect all attempts for a specific execution trace.

**Input Schema**:
```json
{
  "trace_id": "string (required)"
}
```

**Output**:
```json
{
  "executions": [/* array of execution records */],
  "retry_count": 2,
  "final_status": "success"
}
```

**Example**:
```json
{"name": "tasks_by_trace", "arguments": {"trace_id": "trace_abc123"}}
```

---

### `tasks_last_failed`

List recent failed executions with error pattern analysis.

**Input Schema**:
```json
{
  "limit": "integer (1-50, default: 10)",
  "task_id": "string (optional)",
  "unique_errors": "boolean (default: false)"
}
```

**Output**:
```json
{
  "failures": [
    {
      "id": "exp_fail1",
      "task_id": "task_xyz",
      "error": {
        "type": "ValueError",
        "message": "Invalid input",
        "stack_hash": "abc123"
      }
    }
  ],
  "error_patterns": [
    {
      "stack_hash": "abc123",
      "count": 5,
      "first_seen": "2025-11-11T10:00:00Z",
      "last_seen": "2025-11-11T14:00:00Z",
      "example_task_id": "task_xyz"
    }
  ]
}
```

**Example**:
```json
{"name": "tasks_last_failed", "arguments": {"unique_errors": true, "limit": 5}}
```

---

### `astra.health`

Server health check and tool listing.

**Input Schema**:
```json
{}
```

**Output**:
```json
{
  "ok": true,
  "server": "astra-mcp",
  "tools": ["tasks_list", "astra.schedule.create", ...],
  "tool_count": 9
}
```

**Example**:
```json
{"name": "astra.health", "arguments": {}}
```

---

## Schedule Management (Tier 1)

**Safety**: Local writes only, per-day budget limits (default: 4/day)

### `astra.schedule.create`

Create a new cron schedule with safety tier and budget.

**Input Schema**:
```json
{
  "name": "string (required)",
  "cron_expression": "string (required, valid cron syntax)",
  "target_tool": "string (required)",
  "payload": "object (required)",
  "safety_tier": "integer (0, 1, or 2, default: 1)",
  "per_day_budget": "integer (default: 4)"
}
```

**Output**:
```json
{
  "success": true,
  "schedule_id": "sch_abc12345",
  "schedule": {
    "id": "sch_abc12345",
    "name": "daily_backup",
    "cron": "0 2 * * *",
    "target_tool": "backup_data",
    "payload": {"target": "/data"},
    "status": "active",
    "safety_tier": 1,
    "next_run_at": "2025-11-12T02:00:00Z",
    "run_budget": {
      "per_day": 4,
      "consumed": 0,
      "last_reset": "2025-11-11T00:00:00Z"
    }
  }
}
```

**Cron Syntax**:
- `*/15 * * * *` - Every 15 minutes
- `0 */6 * * *` - Every 6 hours
- `0 9 * * 1-5` - Weekdays at 9 AM
- `0 0 1 * *` - First day of month

**Example**:
```json
{
  "name": "astra.schedule.create",
  "arguments": {
    "name": "autonomous_coding",
    "cron_expression": "0 */4 * * *",
    "target_tool": "execute_goal",
    "payload": {"goal_source": "top_desire"},
    "safety_tier": 1,
    "per_day_budget": 6
  }
}
```

---

### `astra.schedule.modify`

Modify an existing schedule's cron, payload, or budget.

**Input Schema**:
```json
{
  "schedule_id": "string (required)",
  "cron_expression": "string (optional)",
  "target_tool": "string (optional)",
  "payload": "object (optional)",
  "per_day_budget": "integer (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "schedule_id": "sch_abc12345",
  "schedule": {/* updated schedule */}
}
```

**Example**:
```json
{
  "name": "astra.schedule.modify",
  "arguments": {
    "schedule_id": "sch_abc12345",
    "per_day_budget": 8
  }
}
```

---

### `astra.schedule.pause`

Pause a schedule to stop execution.

**Input Schema**:
```json
{
  "schedule_id": "string (required)"
}
```

**Output**:
```json
{
  "success": true,
  "schedule_id": "sch_abc12345",
  "status": "paused"
}
```

---

### `astra.schedule.resume`

Resume a paused schedule.

**Input Schema**:
```json
{
  "schedule_id": "string (required)"
}
```

**Output**:
```json
{
  "success": true,
  "schedule_id": "sch_abc12345",
  "status": "active",
  "next_run_at": "2025-11-12T06:00:00Z"
}
```

---

### `astra.schedule.list`

List all schedules with optional status filter.

**Input Schema**:
```json
{
  "status": "active | paused (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "schedules": [
    {
      "id": "sch_abc12345",
      "name": "daily_backup",
      "cron": "0 2 * * *",
      "status": "active",
      "safety_tier": 1,
      "next_run_at": "2025-11-12T02:00:00Z",
      "budget_remaining": 4
    }
  ],
  "count": 1
}
```

**Example**:
```json
{"name": "astra.schedule.list", "arguments": {"status": "active"}}
```

---

## Desire Management (Tier 1)

**Safety**: Local writes only, no budget limits (desires are metadata)

### `astra.desires.record`

Record a new vague wish or desire.

**Input Schema**:
```json
{
  "text": "string (required)",
  "strength": "number (0.0-1.0, default: 1.0)",
  "tags": "array of strings (optional)",
  "context": "object (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "desire_id": "des_xyz789",
  "desire": {
    "id": "des_xyz789",
    "text": "I wish I had better test coverage",
    "strength": 1.0,
    "created_at": "2025-11-11T14:30:00Z",
    "last_reinforced_at": "2025-11-11T14:30:00Z",
    "tags": ["testing", "quality"],
    "context": {"triggered_by": "code_review"}
  }
}
```

**Example**:
```json
{
  "name": "astra.desires.record",
  "arguments": {
    "text": "I want to optimize database queries",
    "strength": 0.8,
    "tags": ["performance", "database"],
    "context": {"slow_query_count": 15}
  }
}
```

---

### `astra.desires.list`

List top desires sorted by strength.

**Input Schema**:
```json
{
  "limit": "integer (1-50, default: 10)",
  "min_strength": "number (0.0-1.0, default: 0.0)",
  "tag": "string (optional, filter by tag)"
}
```

**Output**:
```json
{
  "success": true,
  "desires": [
    {
      "id": "des_xyz789",
      "text": "I wish I had better test coverage",
      "strength": 0.95,
      "created_at": "2025-11-10T10:00:00Z",
      "tags": ["testing", "quality"]
    }
  ],
  "count": 1
}
```

**Example**:
```json
{
  "name": "astra.desires.list",
  "arguments": {"limit": 5, "min_strength": 0.5}
}
```

---

### `astra.desires.reinforce`

Manually boost a desire's strength (prevents decay).

**Input Schema**:
```json
{
  "desire_id": "string (required)",
  "delta": "number (default: 0.1)"
}
```

**Output**:
```json
{
  "success": true,
  "desire_id": "des_xyz789",
  "new_strength": 1.0
}
```

**Example**:
```json
{
  "name": "astra.desires.reinforce",
  "arguments": {"desire_id": "des_xyz789", "delta": 0.2}
}
```

---

## Error Handling

All tools return error responses in this format:

```json
{
  "success": false,
  "error": "Error message description"
}
```

Common errors:
- **Missing required field**: `"Missing required field: {field_name}"`
- **Invalid cron**: `"Invalid cron expression"`
- **Not found**: `"{Resource} not found: {id}"`
- **Invalid range**: `"{Field} must be between {min} and {max}"`

---

## Usage Patterns

### Autonomous Coding Loop

```json
// 1. Record desire
{"name": "astra.desires.record", "arguments": {
  "text": "I want to improve code quality",
  "tags": ["quality"]
}}

// 2. Create schedule to act on desires
{"name": "astra.schedule.create", "arguments": {
  "name": "quality_improvement",
  "cron_expression": "0 */6 * * *",
  "target_tool": "execute_goal",
  "payload": {"goal_source": "top_desire"},
  "safety_tier": 1,
  "per_day_budget": 4
}}

// 3. Monitor executions
{"name": "tasks_list", "arguments": {"limit": 10}}

// 4. Check for failures
{"name": "tasks_last_failed", "arguments": {"unique_errors": true}}
```

### Daily Introspection

```json
{"name": "astra.schedule.create", "arguments": {
  "name": "morning_review",
  "cron_expression": "0 9 * * *",
  "target_tool": "tasks_last_failed",
  "payload": {"limit": 10},
  "safety_tier": 0,
  "per_day_budget": 0
}}
```

### Desire Tracking

```json
// Record desires as you work
{"name": "astra.desires.record", "arguments": {
  "text": "Need better error handling in API layer",
  "strength": 0.7,
  "tags": ["reliability", "api"]
}}

// Review top desires weekly
{"name": "astra.desires.list", "arguments": {"limit": 10}}

// Reinforce important desires
{"name": "astra.desires.reinforce", "arguments": {
  "desire_id": "des_abc123",
  "delta": 0.2
}}
```

---

## Testing Tools

Use `bin/mcp` to test tools manually:

```bash
cat << 'EOF' | bin/mcp 2>/dev/null | tail -1 | python3 -m json.tool
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","id":2,"method":"initialized"}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"astra.health","arguments":{}}}
EOF
```

---

## See Also

- [Quick Start Guide](MCP_QUICKSTART.md)
- [Implementation Details](MCP_AUTONOMOUS_SCHEDULING_IMPLEMENTATION.md)
- [Safety Tiers](SCHEDULE_SAFETY_TIERS.md)
