#!/usr/bin/env python3
"""MCP stdio server for Astra self-scheduling and introspection.

This server exposes:
- Task execution introspection (tasks_list, tasks_by_trace, tasks_last_failed)
- Self-scheduling tools (schedule.create, schedule.modify, schedule.pause, schedule.resume)
- Desire recording (desires.record)
- Goal creation (goals.create)

Usage:
    python -m bin.mcp_server

For systemd:
    [Service]
    ExecStart=/path/venv/bin/python -m bin.mcp_server
    Restart=always
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.memory.raw_store import create_raw_store
from src.mcp.task_execution_server import create_task_execution_server
from src.mcp.tools.schedule import create_schedule_tools
from src.mcp.tools.desires import create_desire_tools

# Optional MCP imports (graceful degradation)
try:
    from mcp.server import Server
    from mcp.types import TextContent, Tool
    from mcp.server.stdio import stdio_server

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    Server = None
    TextContent = None
    Tool = None
    stdio_server = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def create_astra_mcp_server() -> Server:
    """Create MCP server with all Astra tools registered.

    Returns:
        Configured MCP server

    Raises:
        RuntimeError: If MCP library not available
    """
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "MCP library not available. Install with: pip install mcp>=1.0.0"
        )

    # Initialize RawStore for task execution tools
    raw_store = create_raw_store(settings.RAW_STORE_DB_PATH)

    # Create server with task execution tools
    server = create_task_execution_server(raw_store=raw_store)

    # Initialize schedule and desire tools
    schedule_tools = create_schedule_tools()
    desire_tools = create_desire_tools()

    # Get existing tools from base server
    existing_list_tools_handler = server.request_handlers.get(
        Tool.__class__.__bases__[0]  # Get ListToolsRequest type
    )

    # Register combined list_tools handler
    @server.list_tools()
    async def handle_list_tools():
        """Return all available tools (task execution + schedule + health)."""
        # Get task execution tools from base server
        if existing_list_tools_handler:
            base_result = await existing_list_tools_handler(None)
            base_tools = base_result.result.tools if hasattr(base_result, "result") else []
        else:
            base_tools = []

        # Add schedule tools
        schedule_tool_defs = [
            Tool(
                name="astra.schedule.create",
                description="Create a scheduled task with cron expression and safety tier",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "cron_expression": {"type": "string"},
                        "target_tool": {"type": "string"},
                        "payload": {"type": "object"},
                        "safety_tier": {"type": "integer", "enum": [0, 1, 2], "default": 1},
                        "per_day_budget": {"type": "integer", "default": 4},
                    },
                    "required": ["name", "cron_expression", "target_tool", "payload"],
                },
            ),
            Tool(
                name="astra.schedule.modify",
                description="Modify an existing schedule (cron, payload, or budget)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "schedule_id": {"type": "string"},
                        "cron_expression": {"type": "string"},
                        "target_tool": {"type": "string"},
                        "payload": {"type": "object"},
                        "per_day_budget": {"type": "integer"},
                    },
                    "required": ["schedule_id"],
                },
            ),
            Tool(
                name="astra.schedule.pause",
                description="Pause a schedule to stop execution",
                inputSchema={
                    "type": "object",
                    "properties": {"schedule_id": {"type": "string"}},
                    "required": ["schedule_id"],
                },
            ),
            Tool(
                name="astra.schedule.resume",
                description="Resume a paused schedule",
                inputSchema={
                    "type": "object",
                    "properties": {"schedule_id": {"type": "string"}},
                    "required": ["schedule_id"],
                },
            ),
            Tool(
                name="astra.schedule.list",
                description="List all schedules (optionally filtered by status)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["active", "paused"]},
                    },
                },
            ),
            Tool(
                name="astra.health",
                description="Health check for MCP server",
                inputSchema={"type": "object"},
            ),
        ]

        # Add desire tools
        desire_tool_defs = [
            Tool(
                name="astra.desires.record",
                description="Record a new vague wish or desire",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "context": {"type": "object"},
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="astra.desires.list",
                description="List top desires sorted by strength",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                        "min_strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
                        "tag": {"type": "string"},
                    },
                },
            ),
            Tool(
                name="astra.desires.reinforce",
                description="Manually reinforce a desire to boost its strength",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "desire_id": {"type": "string"},
                        "delta": {"type": "number", "default": 0.1},
                    },
                    "required": ["desire_id"],
                },
            ),
        ]

        return base_tools + schedule_tool_defs + desire_tool_defs

    # Get existing call_tool handler
    existing_call_tool_handler = None
    from mcp import types as mcp_types
    if mcp_types.CallToolRequest in server.request_handlers:
        # Save reference to original handler for task execution tools
        original_handler = server.request_handlers[mcp_types.CallToolRequest]

        async def existing_call_tool_handler(name: str, arguments: dict):
            # Reconstruct request to call original handler
            req = mcp_types.CallToolRequest(
                method="tools/call",
                params=mcp_types.CallToolRequestParams(name=name, arguments=arguments),
            )
            result = await original_handler(req)
            return result.result.content

    # Register combined call_tool handler
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict):
        """Route tool calls to appropriate handlers."""
        arguments = arguments or {}

        # Route schedule tools
        if name == "astra.schedule.create":
            result = schedule_tools.create(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "astra.schedule.modify":
            result = schedule_tools.modify(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "astra.schedule.pause":
            result = schedule_tools.pause(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "astra.schedule.resume":
            result = schedule_tools.resume(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "astra.schedule.list":
            result = schedule_tools.list_schedules(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "astra.health":
            # Get tool count from list_tools
            all_tools = await handle_list_tools()
            health_info = {
                "ok": True,
                "server": "astra-mcp",
                "tools": [t.name for t in all_tools],
                "tool_count": len(all_tools),
            }
            return [TextContent(type="text", text=json.dumps(health_info, indent=2))]

        # Route desire tools
        elif name == "astra.desires.record":
            result = desire_tools.record(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "astra.desires.list":
            result = desire_tools.list_desires(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "astra.desires.reinforce":
            result = desire_tools.reinforce(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # Fall back to task execution tools
        elif existing_call_tool_handler:
            return await existing_call_tool_handler(name, arguments)

        else:
            raise ValueError(f"Unknown tool: {name}")

    logger.info("MCP server created with task execution, schedule, and desire tools")

    # TODO: Register goal tools from src/mcp/tools/goals.py (future work)

    return server


async def main():
    """Run MCP server via stdio transport."""
    if not _MCP_AVAILABLE:
        logger.error("MCP library not installed. Install with: pip install mcp>=1.0.0")
        sys.exit(1)

    logger.info("Starting Astra MCP server (stdio transport)")

    try:
        server = create_astra_mcp_server()

        # Run server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    except KeyboardInterrupt:
        logger.info("MCP server stopped by user")
    except Exception as e:
        logger.error(f"MCP server failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
