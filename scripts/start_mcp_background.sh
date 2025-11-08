#!/bin/bash
# Start MCP server in background with logging

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

PIDFILE="$LOG_DIR/mcp-server.pid"
LOGFILE="$LOG_DIR/mcp-server.log"
ERRFILE="$LOG_DIR/mcp-server-error.log"

# Check if already running
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "MCP server already running with PID $PID"
        exit 0
    else
        echo "Removing stale PID file"
        rm "$PIDFILE"
    fi
fi

# Start server
source venv/bin/activate
nohup python scripts/run_task_execution_mcp.py \
    --db data/raw_store.db \
    --log-level INFO \
    >> "$LOGFILE" 2>> "$ERRFILE" &

PID=$!
echo $PID > "$PIDFILE"
echo "MCP server started with PID $PID"
echo "Logs: $LOGFILE"
echo "Errors: $ERRFILE"
