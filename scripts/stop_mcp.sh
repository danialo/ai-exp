#!/bin/bash
# Stop MCP server

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$REPO_ROOT/logs/mcp-server.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "MCP server not running (no PID file)"
    exit 0
fi

PID=$(cat "$PIDFILE")
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "MCP server not running (stale PID)"
    rm "$PIDFILE"
    exit 0
fi

echo "Stopping MCP server (PID $PID)..."
kill "$PID"

# Wait for graceful shutdown
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "MCP server stopped"
        rm "$PIDFILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
if ps -p "$PID" > /dev/null 2>&1; then
    echo "Force killing MCP server..."
    kill -9 "$PID"
    rm "$PIDFILE"
fi

echo "MCP server stopped"
