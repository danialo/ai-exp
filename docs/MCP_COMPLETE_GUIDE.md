# Complete Guide to Astra's MCP Server

Everything you need to know about using and understanding the MCP server.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [How It Works](#how-it-works)
4. [All 9 Tools](#all-9-tools)
5. [Usage Patterns](#usage-patterns)
6. [Architecture](#architecture)
7. [Safety & Budgets](#safety--budgets)
8. [Data Persistence](#data-persistence)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Astra's MCP server exposes tools for autonomous operation through the Model Context Protocol.

**What you get**:
- 9 tools for introspection, scheduling, and desire tracking
- Production-ready (67/67 tests passing)
- Stdio transport (works with Claude Desktop, VS Code, etc.)
- Complete audit trail via NDJSON chains
- Safety tiers with budget enforcement

**Where to go next**:
- Quick usage → [MCP Quick Start](MCP_QUICKSTART.md)
- All tool details → [MCP Tools Reference](MCP_TOOLS_REFERENCE.md)
- Deep dive → [MCP Architecture](MCP_ARCHITECTURE.md)

---

## Quick Start

### 1. Test the server

```bash
cd /home/d/git/ai-exp
bin/mcp
```

You should see log messages confirming startup.

### 2. Configure Claude Desktop

Edit `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "astra": {
      "command": "/home/d/git/ai-exp/bin/mcp"
    }
  }
}
```

### 3. Restart Claude Desktop

Completely quit and reopen Claude Desktop.

### 4. Test a tool

Ask Claude:
> "Use astra.health to check the server status"

You should see a response with tool count and status.

---

## How It Works

### Data Flow

```
1. You ask Claude Desktop a question
2. Claude decides to use an Astra tool
3. Claude Desktop starts bin/mcp (if not running)
4. Tool call sent via stdio (JSON-RPC 2.0)
5. MCP server routes to appropriate handler
6. Handler calls core service (ScheduleService, DesireStore, etc.)
7. Service updates state and persistence
8. Response returned to Claude
9. Claude shows you the result
```

### Stdio Transport

The MCP server uses **stdio** (standard input/output):
- Not a daemon - started by clients on-demand
- Communicates via JSON-RPC 2.0
- Stateless between invocations
- Standard MCP pattern

This is why you don't need to "keep it running" - clients start it when needed.

---

## All 9 Tools

### Introspection (Tier 0 - Read-only)

**1. tasks_list**
Query task execution history with filters.

Example: "Show me the last 20 successful tasks"

**2. tasks_by_trace**
Inspect a specific execution trace and all retry attempts.

Example: "Show me all attempts for trace_abc123"

**3. tasks_last_failed**
Recent failures with error pattern analysis.

Example: "What are the most common errors in the last 24 hours?"

**4. astra.health**
Server health check and tool listing.

Example: "Is the MCP server working?"

### Scheduling (Tier 1 - Local writes)

**5. astra.schedule.create**
Create a cron schedule with safety tier and budget.

Example: "Schedule autonomous coding every 6 hours, max 4 times per day"

**6. astra.schedule.modify**
Modify an existing schedule's parameters.

Example: "Change the budget for schedule sch_abc123 to 8 per day"

**7. astra.schedule.pause**
Pause a schedule to stop execution.

Example: "Pause the autonomous coding schedule"

**8. astra.schedule.resume**
Resume a paused schedule.

Example: "Resume the daily backup schedule"

**9. astra.schedule.list**
List all schedules with optional status filter.

Example: "Show me all active schedules"

### Desires (Tier 1 - Local writes)

**10. astra.desires.record**
Record a vague wish or desire.

Example: "I wish I had better test coverage"

**11. astra.desires.list**
List top desires sorted by strength.

Example: "What are my top 5 desires?"

**12. astra.desires.reinforce**
Manually boost a desire's strength.

Example: "Reinforce the test coverage desire"

---

## Usage Patterns

### Pattern 1: Daily Introspection

Ask Claude:
> "Create a schedule to check for failed tasks every morning at 9 AM"

Claude will call:
```json
{
  "name": "astra.schedule.create",
  "arguments": {
    "name": "morning_review",
    "cron_expression": "0 9 * * *",
    "target_tool": "tasks_last_failed",
    "payload": {"limit": 10, "unique_errors": true},
    "safety_tier": 0
  }
}
```

### Pattern 2: Autonomous Coding

Ask Claude:
> "Set up autonomous coding to run every 4 hours, capped at 6 times per day"

Claude will call:
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

### Pattern 3: Desire Tracking

Throughout your session, tell Claude:
> "I wish the API error handling was more robust"

Claude will call:
```json
{
  "name": "astra.desires.record",
  "arguments": {
    "text": "I wish the API error handling was more robust",
    "strength": 0.8,
    "tags": ["reliability", "api"]
  }
}
```

Later:
> "What are my top desires?"

Claude will call:
```json
{
  "name": "astra.desires.list",
  "arguments": {"limit": 10, "min_strength": 0.5}
}
```

---

## Architecture

### Component Stack

```
┌─────────────────────────────────────┐
│      MCP Client (Claude Desktop)    │
└──────────────┬──────────────────────┘
               │ stdio (JSON-RPC 2.0)
               ▼
┌─────────────────────────────────────┐
│        bin/mcp_server.py            │
│  - Protocol handling                │
│  - Tool routing                     │
└──────────────┬──────────────────────┘
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
   ┌───────┐ ┌───────┐ ┌───────┐
   │ Task  │ │Sched- │ │Desire │
   │ Tools │ │ Tools │ │ Tools │
   └───┬───┘ └───┬───┘ └───┬───┘
       │         │         │
       ▼         ▼         ▼
   ┌───────┐ ┌───────┐ ┌───────┐
   │ Raw   │ │Sched- │ │Desire │
   │ Store │ │Service│ │ Store │
   └───┬───┘ └───┬───┘ └───┬───┘
       │         │         │
       ▼         ▼         ▼
   ┌───────┐ ┌──────────────┐
   │SQLite │ │NDJSON + Index│
   └───────┘ └──────────────┘
```

### Key Files

**Entry Point**:
- `bin/mcp` - Wrapper script (117 bytes)
- `bin/mcp_server.py` - Main server (12KB, 300 lines)

**Core Services**:
- `src/services/schedule_service.py` - Scheduling (565 lines)
- `src/services/desire_store.py` - Desires (349 lines)
- `src/mcp/task_execution_server.py` - Task introspection

**Tool Handlers**:
- `src/mcp/tools/schedule.py` - Schedule tools (312 lines)
- `src/mcp/tools/desires.py` - Desire tools (195 lines)

**Tests**:
- `tests/test_schedule_service.py` - 27 tests
- `tests/test_desire_store.py` - 26 tests
- `tests/test_mcp_schedule_tools.py` - 14 tests

---

## Safety & Budgets

### Three-Tier System

**Tier 0: Read-Only**
- Tools: tasks_list, tasks_by_trace, tasks_last_failed, astra.health
- Auto-run: ✅ No restrictions
- Budget: ❌ Unlimited
- Scope: Read-only queries

**Tier 1: Local Writes**
- Tools: schedule.*, desires.*, execute_goal
- Auto-run: ✅ With budget enforcement
- Budget: ✅ Default 4/day per schedule
- Scope: Local repository only

**Tier 2: External** *(future)*
- Tools: deploy_to_vercel, send_email, create_github_pr
- Auto-run: ❌ Requires approval
- Budget: ✅ Advisory (approval is primary gate)
- Scope: External side effects

### Budget Enforcement

Each Tier 1 schedule has a daily budget:

```python
class RunBudget:
    per_day: int = 4           # Max runs per day
    consumed: int = 0          # Runs today
    last_reset: str            # Last reset timestamp
```

**How it works**:
1. Schedule becomes due
2. Check: `consumed < per_day`?
3. If yes: execute and increment consumed
4. If no: skip and log "budget exhausted"
5. At midnight UTC: reset consumed = 0

**Example**:
```
Schedule: autonomous_coding
Budget: 4/day
Cron: Every 4 hours (0 */4 * * *)

Timeline:
00:00 - consumed=0, reset occurs
04:00 - Run #1, consumed=1
08:00 - Run #2, consumed=2
12:00 - Run #3, consumed=3
16:00 - Run #4, consumed=4
20:00 - SKIPPED (budget exhausted)
24:00 - consumed=0, reset occurs
```

### Approval Workflow (Tier 2)

**Not yet implemented**, but design:

1. Schedule with tier=2 becomes due
2. Executor checks for approval token at:
   `var/approvals/pending/sch_abc123.token`
3. If token exists and valid:
   - Execute tool
   - Delete token (one-time use)
4. If no token:
   - Skip execution
   - Log "awaiting approval"

**Token format**:
```json
{
  "schedule_id": "sch_abc123",
  "approved_by": "human@example.com",
  "approved_at": "2025-11-12T10:00:00Z",
  "expires_at": "2025-11-12T11:00:00Z"
}
```

---

## Data Persistence

### NDJSON + Index Pattern

All state uses **dual persistence**:

**NDJSON Chain** (append-only):
- File: `var/{schedules|desires}/YYYY-MM.ndjson.gz`
- Purpose: Audit trail, never modified
- Format: One JSON object per line
- Grep-friendly for debugging

**Index** (compact):
- File: `var/{schedules|desires}/index.json`
- Purpose: Fast lookups, current state
- Format: JSON object mapping ID → entity
- Rebuilt from chain if corrupted

### Example Files

**Schedule Chain** (`var/schedules/2025-11.ndjson.gz`):
```json
{"_timestamp":"2025-11-11T14:00:00Z","event":"schedule_created","schedule_id":"sch_abc123",...}
{"_timestamp":"2025-11-11T16:00:00Z","event":"schedule_executed","schedule_id":"sch_abc123"}
{"_timestamp":"2025-11-11T20:00:00Z","event":"schedule_paused","schedule_id":"sch_abc123"}
```

**Schedule Index** (`var/schedules/index.json`):
```json
{
  "sch_abc123": {
    "id": "sch_abc123",
    "name": "daily_backup",
    "cron": "0 2 * * *",
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

### Why This Pattern?

**Benefits**:
- ✅ Complete audit trail (chain)
- ✅ Fast queries (index)
- ✅ Easy debugging (grep chain)
- ✅ Reliable recovery (rebuild index)
- ✅ No database dependency
- ✅ Git-friendly (plain text)

**Trade-offs**:
- ❌ Not optimized for >10,000 entities
- ❌ Manual index rebuild required (future enhancement)

---

## Testing

### Run All MCP Tests

```bash
pytest tests/test_schedule_service.py \
       tests/test_desire_store.py \
       tests/test_mcp_schedule_tools.py \
       -v
```

Expected: **67/67 tests passing**

### Test Server Manually

```bash
# Test initialization
cat << 'EOF' | bin/mcp 2>/dev/null | tail -1 | python3 -m json.tool
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","id":2,"method":"initialized"}
{"jsonrpc":"2.0","id":3,"method":"tools/list"}
EOF
```

You should see JSON response with all 9 tools listed.

### Test Individual Tool

```bash
cat << 'EOF' | bin/mcp 2>/dev/null | tail -1 | python3 -m json.tool
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","id":2,"method":"initialized"}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"astra.health","arguments":{}}}
EOF
```

Expected response includes:
```json
{
  "ok": true,
  "server": "astra-mcp",
  "tool_count": 9
}
```

---

## Troubleshooting

### Problem: "MCP server not responding"

**Check 1**: Server starts successfully
```bash
bin/mcp
# Look for: "MCP server created with..."
```

**Check 2**: Config path is correct
```bash
cat ~/.config/Claude/claude_desktop_config.json | jq .mcpServers.astra
# Should show absolute path to bin/mcp
```

**Check 3**: Restart Claude Desktop completely
- Quit entirely (not just close window)
- Reopen

### Problem: "Tool calls fail"

**Check service logs**:
```bash
tail -f logs/mcp_server.log
# (if you've configured logging)
```

**Check data directories exist**:
```bash
ls -la var/schedules var/desires
```

**Check file permissions**:
```bash
ls -l bin/mcp
# Should be executable (rwxr-xr-x)
```

### Problem: "Tests failing"

**Check Python environment**:
```bash
which python3
python3 --version
# Should be Python 3.8+
```

**Check dependencies**:
```bash
pip list | grep -E "mcp|croniter"
```

**Run with verbose output**:
```bash
pytest tests/test_schedule_service.py -v --tb=long
```

---

## Next: Read the Docs

Now that you understand the basics, explore:

1. **[MCP Quick Start](MCP_QUICKSTART.md)** - Hands-on tutorial
2. **[MCP Tools Reference](MCP_TOOLS_REFERENCE.md)** - Complete tool specs
3. **[MCP Architecture](MCP_ARCHITECTURE.md)** - Deep dive into design
4. **[Safety Tiers](SCHEDULE_SAFETY_TIERS.md)** - Budget and approval details
5. **[Implementation](MCP_AUTONOMOUS_SCHEDULING_IMPLEMENTATION.md)** - Full technical details

---

## Summary

You now know:
- ✅ What the MCP server does (9 tools for autonomy)
- ✅ How to set it up (bin/mcp + Claude Desktop config)
- ✅ How it works (stdio transport, NDJSON persistence)
- ✅ The safety model (3 tiers, budgets, approvals)
- ✅ How to test it (67 tests, manual testing)
- ✅ How to troubleshoot (common issues + fixes)

**Ready to use Astra's MCP server!**
