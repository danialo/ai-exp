# MCP Server

MCP (Model Context Protocol) stdio server for Astra self-scheduling and introspection.

## Setup

```bash
# Create runtime directories
mkdir -p var/schedules var/approvals/pending var/desires logs

# Install MCP library (if not already installed)
pip install mcp>=1.0.0
```

## Running

The MCP server uses **stdio transport** - it's started on-demand by MCP clients.

### Via wrapper script (recommended):
```bash
bin/mcp
```

### Via Python module:
```bash
python -m bin.mcp_server
```

### Configure in MCP clients:

For Claude Desktop (`~/.config/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "astra": {
      "command": "/home/d/git/ai-exp/bin/mcp",
      "args": []
    }
  }
}
```

### Via systemd user unit:
```ini
[Unit]
Description=Astra MCP Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/ai-exp
ExecStart=/path/to/ai-exp/venv/bin/python -m bin.mcp_server
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

Save to `~/.config/systemd/user/astra-mcp.service` then:
```bash
systemctl --user daemon-reload
systemctl --user enable astra-mcp
systemctl --user start astra-mcp
systemctl --user status astra-mcp
```

## Tools

### Introspection (Tier 0 - Read-only)
- `astra.health` - Server health check and tool listing
- `tasks_list` - Query task execution history with filters
- `tasks_by_trace` - Inspect specific task execution trace
- `tasks_last_failed` - Recent failures and error patterns

### Scheduling (Tier 1 - Local writes)
- `astra.schedule.create` - Create cron schedule with safety tier and budget
- `astra.schedule.modify` - Modify existing schedule (cron, payload, budget)
- `astra.schedule.pause` - Pause a schedule to stop execution
- `astra.schedule.resume` - Resume a paused schedule
- `astra.schedule.list` - List all schedules (optionally filtered by status)

### Desires (Tier 1 - Local writes)
- `astra.desires.record` - Record a vague wish or desire
- `astra.desires.list` - List top desires sorted by strength
- `astra.desires.reinforce` - Manually boost a desire's strength

## Safety Tiers

- **Tier 0**: Read-only introspection (auto-run)
- **Tier 1**: Local writes with per-day budgets (auto-run, capped)
- **Tier 2**: External side effects (requires approval token in `var/approvals/pending/`)

## Testing

```bash
# Quick health check via stdio
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"astra.health"},"id":1}' | python -m bin.mcp_server

# With MCP inspector (if available)
mcp-inspector bin.mcp_server
```

## Logs

Logs go to stderr. Redirect as needed:
```bash
python -m bin.mcp_server 2>> logs/mcp_server.log
```
