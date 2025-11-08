#!/bin/bash
# Check MCP server status

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$REPO_ROOT/logs/mcp-server.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "MCP server: NOT RUNNING (no PID file)"
    exit 1
fi

PID=$(cat "$PIDFILE")
if ps -p "$PID" > /dev/null 2>&1; then
    echo "MCP server: RUNNING (PID $PID)"
    echo "Logs: $REPO_ROOT/logs/mcp-server.log"
    ps -p "$PID" -o pid,ppid,cmd,etime,stat
    exit 0
else
    echo "MCP server: NOT RUNNING (stale PID)"
    rm "$PIDFILE"
    exit 1
fi
