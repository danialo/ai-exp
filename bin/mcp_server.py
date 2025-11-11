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

    # Add health check tool
    @server.tool(name="astra.health", description="Health check for MCP server")
    async def health_check(payload=None):
        """Return server health status and registered tools."""
        # Get list of registered tools
        tool_names = [tool.name for tool in server.list_tools()]

        health_info = {
            "ok": True,
            "server": "astra-mcp",
            "tools": tool_names,
            "tool_count": len(tool_names),
        }

        return {
            "content": [
                TextContent(type="text", text=json.dumps(health_info, indent=2))
            ]
        }

    logger.info(f"MCP server created with {len(server.list_tools())} tools")

    # TODO: Register schedule tools from src/mcp/tools/schedule.py
    # TODO: Register desire tools from src/mcp/tools/desires.py
    # TODO: Register goal tools from src/mcp/tools/goals.py

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
