# MCP Server Summary

Quick reference for Astra's Model Context Protocol server implementation.

## TL;DR

**What**: 9 MCP tools for autonomous scheduling, task introspection, and desire tracking

**Status**: ✅ Production-ready (67/67 tests passing)

**Start**: `bin/mcp` (stdio transport, on-demand)

**Docs**: See `docs/MCP_QUICKSTART.md`

---

## Quick Start (30 seconds)

```bash
# 1. Start server (for testing)
bin/mcp

# 2. Configure Claude Desktop
# Add to ~/.config/Claude/claude_desktop_config.json:
{
  "mcpServers": {
    "astra": {
      "command": "/home/d/git/ai-exp/bin/mcp"
    }
  }
}

# 3. Restart Claude Desktop

# 4. Ask Claude: "Use astra.schedule.list to show me all schedules"
```

---

## What It Does

### 1. Task Introspection (Read-only)
Query execution history, inspect traces, analyze failures.

**Tools**: `tasks_list`, `tasks_by_trace`, `tasks_last_failed`, `astra.health`

**Example**:
```
Human: "Show me the last 5 failed tasks"
Claude: [calls tasks_last_failed with limit=5]
```

### 2. Autonomous Scheduling (Local writes, budgeted)
Create cron schedules with safety tiers and per-day budgets.

**Tools**: `astra.schedule.create/modify/pause/resume/list`

**Example**:
```
Human: "Schedule autonomous coding every 6 hours, max 4 times per day"
Claude: [calls astra.schedule.create with appropriate parameters]
```

### 3. Desire Tracking (Local writes)
Record vague wishes that decay over time unless reinforced.

**Tools**: `astra.desires.record/list/reinforce`

**Example**:
```
Human: "I wish I had better test coverage"
Claude: [calls astra.desires.record to save the wish]
```

---

## Safety Model

**Tier 0: Read-only**
- No restrictions, auto-run, unlimited
- Examples: tasks_list, astra.health

**Tier 1: Local writes**
- Auto-run with per-day budget (default: 4/day)
- Examples: schedule.create, desires.record

**Tier 2: External** *(future)*
- Requires approval token
- Examples: deploy_to_vercel, send_email

---

## Data Storage

All persistent state uses **NDJSON + index** pattern:

```
var/
  schedules/
    2025-11.ndjson.gz  # Append-only audit trail
    index.json         # Current state (fast lookups)
  desires/
    2025-11.ndjson.gz  # Append-only audit trail
    index.json         # Current desires
  approvals/           # Future: approval tokens
```

**Benefits**:
- Complete audit trail (NDJSON)
- Fast queries (index)
- Easy debugging (grep NDJSON)
- Reliable recovery (rebuild index from chain)

---

## Architecture

```
MCP Client (Claude Desktop)
    ↓ stdio (JSON-RPC 2.0)
bin/mcp_server.py
    ↓
Tool Handlers (schedule.py, desires.py)
    ↓
Core Services (ScheduleService, DesireStore)
    ↓
Persistence (NDJSON + Index)
```

---

## Implementation Details

**Files Created**:
- `bin/mcp` - Wrapper script for clients
- `bin/mcp_server.py` - Main server (300 lines)
- `src/services/schedule_service.py` - Scheduling (565 lines)
- `src/services/desire_store.py` - Desires (349 lines)
- `src/mcp/tools/schedule.py` - Schedule MCP tools (312 lines)
- `src/mcp/tools/desires.py` - Desire MCP tools (195 lines)

**Tests**:
- ScheduleService: 27 tests
- DesireStore: 26 tests
- MCP Tools: 14 tests
- **Total**: 67/67 passing

**Documentation**:
- Quick Start Guide (MCP_QUICKSTART.md)
- Tools Reference (MCP_TOOLS_REFERENCE.md)
- Architecture Deep Dive (MCP_ARCHITECTURE.md)
- Safety Model (SCHEDULE_SAFETY_TIERS.md)
- Implementation Details (MCP_AUTONOMOUS_SCHEDULING_IMPLEMENTATION.md)

**Total**: ~2,000 lines of code, 67 tests, 5 docs

---

## Usage Examples

### Create Autonomous Coding Schedule

```json
{
  "name": "astra.schedule.create",
  "arguments": {
    "name": "autonomous_coding",
    "cron_expression": "0 */6 * * *",
    "target_tool": "execute_goal",
    "payload": {"goal_source": "top_desire"},
    "safety_tier": 1,
    "per_day_budget": 4
  }
}
```

### Record a Desire

```json
{
  "name": "astra.desires.record",
  "arguments": {
    "text": "I wish I had better test coverage",
    "strength": 0.9,
    "tags": ["testing", "quality"]
  }
}
```

### Query Failed Tasks

```json
{
  "name": "tasks_last_failed",
  "arguments": {
    "unique_errors": true,
    "limit": 10
  }
}
```

---

## Next Steps

### Phase 4: Executor Daemon *(future)*
Actually run scheduled tasks:
- Poll for due schedules
- Check budgets and approvals
- Execute tools
- Handle retries

### Phase 5: Desire → Goal Promotion *(future)*
Automatically create goals from strong desires:
- Promotion criteria (strength ≥ 0.8, age ≥ 3 days)
- LLM converts desire to concrete goal
- User approval workflow

### Phase 6: Advanced Scheduling *(future)*
- Dependencies (run B after A)
- Retry policies
- Dynamic adjustment
- Schedule groups

---

## Troubleshooting

**Server won't start**:
```bash
# Check Python path
which python3
python3 -m bin.mcp_server

# Check imports
python3 -c "from src.mcp.task_execution_server import *"
```

**No tools in Claude Desktop**:
- Check config path: `~/.config/Claude/claude_desktop_config.json`
- Verify absolute path to `bin/mcp`
- Restart Claude Desktop completely

**Tests failing**:
```bash
# Run all MCP tests
pytest tests/test_schedule_service.py tests/test_mcp_schedule_tools.py tests/test_desire_store.py -v
```

---

## Documentation Map

| Doc | Purpose | Audience |
|-----|---------|----------|
| **MCP_QUICKSTART.md** | Get started in 5 min | Users |
| **MCP_TOOLS_REFERENCE.md** | Complete tool docs | Developers |
| **MCP_ARCHITECTURE.md** | Design & implementation | Developers |
| **SCHEDULE_SAFETY_TIERS.md** | Safety model | Operators |
| **MCP_AUTONOMOUS_SCHEDULING_IMPLEMENTATION.md** | Full details | Developers |
| **This file** | Quick reference | Everyone |

---

## Key Design Decisions

**Why stdio transport?**
- Standard MCP pattern
- Clients start server on-demand
- No daemon management needed
- Works with Claude Desktop, VS Code, etc.

**Why NDJSON + index?**
- Complete audit trail
- Fast queries
- Easy debugging
- No database dependency

**Why deterministic IDs?**
- Idempotency (re-create is no-op)
- Automatic deduplication
- ID reveals content

**Why 3 safety tiers?**
- Gradual trust model
- Start conservative
- Human oversight for critical ops
- Budget prevents runaway execution

---

## Production Checklist

- [x] 67 tests passing
- [x] Stdio server working
- [x] All tools registered
- [x] Safety tiers documented
- [x] Data persistence verified
- [x] Client wrapper (`bin/mcp`)
- [x] Comprehensive docs
- [ ] Executor daemon (Phase 4)
- [ ] Approval workflow (Tier 2)
- [ ] Metrics/telemetry (Phase 3.8)

---

## Contact & Support

**Documentation**: `docs/INDEX.md`

**Issues**: GitHub Issues (if applicable)

**Questions**: See documentation first, then ask
