#!/usr/bin/env python3
"""MCP HTTP/SSE server for Astra - runs as persistent daemon.

This server exposes the same tools as the stdio version but over HTTP/SSE,
allowing it to run as a persistent service that multiple clients can connect to.

Usage:
    python -m bin.mcp_server_http --host 0.0.0.0 --port 8765

For systemd:
    [Service]
    ExecStart=/path/venv/bin/python -m bin.mcp_server_http
    Restart=always
"""

import argparse
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

# Optional MCP imports
try:
    from mcp.server import Server
    from mcp.types import TextContent, Tool
    from mcp.server.sse import SseServerTransport
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.responses import JSONResponse

    _MCP_AVAILABLE = True
except ImportError as e:
    _MCP_AVAILABLE = False
    print(f"MCP or HTTP dependencies not available: {e}", file=sys.stderr)
    print("Install with: pip install mcp uvicorn starlette", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_astra_mcp_server() -> Server:
    """Create MCP server with all Astra tools (same as stdio version)."""
    raw_store = create_raw_store(settings.RAW_STORE_DB_PATH)
    server = create_task_execution_server(raw_store=raw_store)

    schedule_tools = create_schedule_tools()
    desire_tools = create_desire_tools()

    # Get existing tools from base server
    existing_list_tools_handler = server.request_handlers.get(
        Tool.__class__.__bases__[0]
    )

    # Register combined list_tools handler
    @server.list_tools()
    async def handle_list_tools():
        """Return all available tools."""
        if existing_list_tools_handler:
            base_result = await existing_list_tools_handler(None)
            base_tools = base_result.result.tools if hasattr(base_result, "result") else []
        else:
            base_tools = []

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
                name="astra.schedule.list",
                description="List all schedules",
                inputSchema={"type": "object", "properties": {"status": {"type": "string"}}},
            ),
            Tool(
                name="astra.health",
                description="Health check",
                inputSchema={"type": "object"},
            ),
        ]

        desire_tool_defs = [
            Tool(
                name="astra.desires.record",
                description="Record a new desire",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "strength": {"type": "number", "default": 1.0},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="astra.desires.list",
                description="List top desires",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10},
                        "min_strength": {"type": "number", "default": 0.0},
                    },
                },
            ),
        ]

        return base_tools + schedule_tool_defs + desire_tool_defs

    # Get existing call_tool handler
    existing_call_tool_handler = None
    from mcp import types as mcp_types
    if mcp_types.CallToolRequest in server.request_handlers:
        original_handler = server.request_handlers[mcp_types.CallToolRequest]

        async def existing_call_tool_handler(name: str, arguments: dict):
            req = mcp_types.CallToolRequest(
                method="tools/call",
                params=mcp_types.CallToolRequestParams(name=name, arguments=arguments),
            )
            result = await original_handler(req)
            return result.result.content

    # Register combined call_tool handler
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict):
        """Route tool calls."""
        arguments = arguments or {}

        if name == "astra.schedule.create":
            result = schedule_tools.create(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        elif name == "astra.schedule.list":
            result = schedule_tools.list_schedules(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        elif name == "astra.desires.record":
            result = desire_tools.record(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        elif name == "astra.desires.list":
            result = desire_tools.list_desires(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        elif name == "astra.health":
            all_tools = await handle_list_tools()
            health_info = {
                "ok": True,
                "server": "astra-mcp-http",
                "tools": [t.name for t in all_tools],
                "tool_count": len(all_tools),
            }
            return [TextContent(type="text", text=json.dumps(health_info, indent=2))]
        elif existing_call_tool_handler:
            return await existing_call_tool_handler(name, arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    return server


async def main():
    """Run MCP server over HTTP/SSE."""
    parser = argparse.ArgumentParser(description="Astra MCP HTTP/SSE Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    args = parser.parse_args()

    logger.info(f"Starting Astra MCP HTTP server on {args.host}:{args.port}")

    # Create MCP server
    mcp_server = create_astra_mcp_server()

    # Create SSE transport
    sse = SseServerTransport("/mcp/messages")

    # Simple health check endpoint
    async def health(request):
        return JSONResponse({"status": "ok", "server": "astra-mcp"})

    # SSE endpoint for receiving messages
    async def sse_handler(scope, receive, send):
        await sse.connect_sse(scope, receive, send)

    # POST endpoint for sending messages
    async def post_handler(scope, receive, send):
        await sse.handle_post_message(scope, receive, send)

    # Initialize transport with server
    async def init_sse():
        async with sse.connect_server(mcp_server):
            # Keep server running
            await asyncio.Event().wait()

    # Create Starlette app
    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/mcp/sse", sse_handler),
            Route("/mcp/messages", post_handler, methods=["POST"]),
        ]
    )

    logger.info("MCP server created with task execution, schedule, and desire tools")
    logger.info(f"MCP endpoint: http://{args.host}:{args.port}/mcp/sse")
    logger.info(f"Health check: http://{args.host}:{args.port}/health")

    # Run with uvicorn
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
