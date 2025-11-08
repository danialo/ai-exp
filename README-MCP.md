# MCP Task Execution Server

## Overview

The MCP (Model Context Protocol) server exposes Astra's task execution tracking data to LLM clients like Claude Desktop for debugging and introspection.

## Current Status

⚠️ **The MCP server code needs to be updated for the new `mcp` library API (v1.21.0).**

The current implementation was written for an older `modelcontextprotocol` package that no longer exists. The new `mcp` package has a different API structure.

## What Needs to be Done

The `src/mcp/task_execution_server.py` file needs to be rewritten to use the new MCP API:

### Old API (not working):
```python
from modelcontextprotocol.server import Server
from modelcontextprotocol.types import TextContent, ToolResult

@server.tool(...)
async def tasks_list(...) -> ToolResult:
    return ToolResult(content=[TextContent(...)])
```

### New API (needs implementation):
```python
from mcp.server import Server
from mcp.types import Tool, CallToolResult, TextContent

# New decorator-based registration
# See: https://github.com/anthropics/python-sdk/tree/main/mcp examples
```

## Architecture

The MCP server is designed as a **read-only stdio adapter**:

```
Claude Desktop (MCP Client)
    ↓ stdio (JSON-RPC)
MCP Server (task_execution_server.py)
    ↓ SQL queries
RawStore (data/raw_store.db)
```

## Available Tools (Once Fixed)

1. **`tasks_list`** - Query task executions with filters
2. **`tasks_by_trace`** - Retrieve execution by correlation ID
3. **`tasks_last_failed`** - Get recent failures for debugging

## How to Fix

1. Study the new `mcp` library API documentation
2. Update `src/mcp/task_execution_server.py` to use new API
3. Test with a simple MCP client
4. Update `pyproject.toml` to use `mcp>=1.0.0` (already done)
5. Create integration tests

## Alternative Approach

Until the MCP server is fixed, you can query task executions directly via SQL:

```python
from src.memory.raw_store import create_raw_store

raw_store = create_raw_store()
executions = raw_store.list_task_executions(
    task_id="belief_gardener",
    status="failed",
    limit=10
)

for exp in executions:
    print(exp.content)
```

Or via the FastAPI endpoints (if they exist in app.py).

## Claude Desktop Configuration (When Fixed)

Once the server is updated, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "astra-task-execution": {
      "command": "python",
      "args": ["scripts/run_task_execution_mcp.py"],
      "cwd": "/home/d/git/ai-exp",
      "env": {
        "PYTHONPATH": "/home/d/git/ai-exp"
      }
    }
  }
}
```

## References

- MCP Specification: https://spec.modelcontextprotocol.io/
- Python MCP SDK: https://github.com/modelcontextprotocol/python-sdk
- Task Execution Tracking Implementation: `.claude/tasks/prompt-008-mcp-task-execution.md`
