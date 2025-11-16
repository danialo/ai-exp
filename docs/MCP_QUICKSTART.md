# MCP Server Quick Start

## What is it?

The Astra MCP server exposes tools for:
- **Task introspection** - Query execution history
- **Scheduling** - Create cron-based autonomous tasks
- **Desires** - Record and track vague wishes

## Start the server

```bash
cd /home/d/git/ai-exp
bin/mcp
```

The server runs on stdio - it's started by MCP clients (Claude Desktop, VS Code, etc.).

## Configure Claude Desktop

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

Restart Claude Desktop and you'll see Astra's tools available.

## Available Tools

**9 tools total:**

### Introspection (Read-only)
- `tasks_list` - Query task execution history
- `tasks_by_trace` - Inspect trace details
- `tasks_last_failed` - Recent failures
- `astra.health` - Server status

### Scheduling (Local writes, budgeted)
- `astra.schedule.create` - New cron schedule
- `astra.schedule.list` - View schedules
- (+ modify, pause, resume)

### Desires (Local writes, budgeted)
- `astra.desires.record` - Record wish
- `astra.desires.list` - View top desires

## Example: Schedule autonomous coding

Ask Claude Desktop with Astra MCP enabled:

> "Use astra.schedule.create to run execute_goal every 6 hours, max 4 times per day, with safety tier 1"

Claude will create the schedule using the MCP tool.

## Data Storage

- Schedules: `var/schedules/` (NDJSON + index)
- Desires: `var/desires/` (NDJSON + index)
- All changes are auditable via NDJSON chains

## Safety Model

- **Tier 0**: Read-only (unlimited)
- **Tier 1**: Local writes (budget-limited)
- **Tier 2**: External ops (requires approval - not yet implemented)

## Test manually

```bash
cat << 'EOF' | bin/mcp 2>/dev/null | tail -1 | python3 -m json.tool
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","id":2,"method":"initialized"}
{"jsonrpc":"2.0","id":3,"method":"tools/list"}
EOF
```

You should see all 9 tools listed in the response.

## Next Steps

1. Configure Claude Desktop with the MCP server
2. Test by asking Claude to list schedules: "Use astra.schedule.list"
3. Create your first autonomous schedule
4. Record desires as you work

See `docs/MCP_AUTONOMOUS_SCHEDULING_IMPLEMENTATION.md` for full details.
