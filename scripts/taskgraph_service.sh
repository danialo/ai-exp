#!/bin/bash
# TaskGraph Query API Service Manager
# Usage: ./taskgraph_service.sh {start|stop|restart|status}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_ROOT/venv"
PID_FILE="/tmp/taskgraph_viewer.pid"
LOG_FILE="/tmp/taskgraph_viewer.log"
VIEWER_SCRIPT="$SCRIPT_DIR/taskgraph_viewer.py"
PYTHON="$VENV/bin/python3"

start() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "TaskGraph viewer already running (PID: $PID)"
            return 1
        else
            echo "Stale PID file found, removing..."
            rm "$PID_FILE"
        fi
    fi

    echo "Starting TaskGraph viewer..."
    cd "$PROJECT_ROOT"
    nohup "$PYTHON" "$VIEWER_SCRIPT" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"

    sleep 2
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✓ Started (PID: $PID)"
        echo "  Logs: $LOG_FILE"
        echo "  API: http://172.239.66.45:8001"
        return 0
    else
        echo "✗ Failed to start. Check logs: $LOG_FILE"
        rm "$PID_FILE" 2>/dev/null
        return 1
    fi
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "TaskGraph viewer not running (no PID file)"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "TaskGraph viewer not running (stale PID)"
        rm "$PID_FILE"
        return 1
    fi

    echo "Stopping TaskGraph viewer (PID: $PID)..."
    kill "$PID"

    # Wait up to 5 seconds for graceful shutdown
    for i in {1..5}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            rm "$PID_FILE"
            echo "✓ Stopped"
            return 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo "Force killing..."
    kill -9 "$PID" 2>/dev/null
    rm "$PID_FILE"
    echo "✓ Stopped (forced)"
    return 0
}

status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Status: STOPPED (no PID file)"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Status: RUNNING (PID: $PID)"
        echo "  API: http://172.239.66.45:8001"
        echo "  Logs: $LOG_FILE"

        # Show recent log lines
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Recent logs:"
            tail -5 "$LOG_FILE"
        fi
        return 0
    else
        echo "Status: STOPPED (stale PID)"
        rm "$PID_FILE"
        return 1
    fi
}

restart() {
    echo "Restarting TaskGraph viewer..."
    stop
    sleep 1
    start
}

logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "No log file found: $LOG_FILE"
        return 1
    fi
}

case "${1:-}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
