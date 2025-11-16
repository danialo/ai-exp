#!/usr/bin/env python3
"""Launch the task execution Model Context Protocol server.

This script is a thin runtime wrapper around
``src.mcp.task_execution_server.create_task_execution_server`` so the MCP
server can be started without touching the main Astra runtime.  It keeps the
dependency optional—if ``modelcontextprotocol`` is not installed the script will
exit with a clear message instead of crashing the shared environment.

Usage examples:

    # Use default raw-store configured in config/settings.py
    python scripts/run_task_execution_mcp.py

    # Point at an explicit SQLite database
    python scripts/run_task_execution_mcp.py --db data/raw_store.db

Press Ctrl+C to stop the server.  The underlying ``Server.run()`` call blocks
and communicates over stdio, ready for MCP-compatible clients (Claude Desktop,
CLI harnesses, etc.).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.memory.raw_store import RawStore, create_raw_store
from src.mcp.task_execution_server import create_task_execution_server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the task execution MCP server (stdio transport)."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to the raw store SQLite database. Defaults to project settings.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level for the launcher script (default: INFO).",
    )
    return parser


def _initialise_raw_store(db_path: Path | None) -> RawStore:
    if db_path is not None:
        return RawStore(db_path)
    return create_raw_store()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("mcp.launcher")

    try:
        raw_store = _initialise_raw_store(args.db)
    except Exception:  # pragma: no cover - surface unexpected init errors
        logger.exception("Failed to initialise raw store")
        return 2

    try:
        server = create_task_execution_server(raw_store)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:  # pragma: no cover - guard against unforeseen issues
        logger.exception("Failed to construct MCP server")
        return 2

    logger.info("Starting task execution MCP server (stdio transport)…")
    try:
        server.run()
    except KeyboardInterrupt:  # pragma: no cover - runtime control flow
        logger.info("Received interrupt, shutting down MCP server")
    finally:
        raw_store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
