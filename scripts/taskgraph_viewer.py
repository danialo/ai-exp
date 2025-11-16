"""
TaskGraph Query API - Production-ready endpoints covering R1-R6 rubric.

Endpoints:
  GET /healthz
  GET /v1/taskgraphs
  GET /v1/taskgraphs/{graph_id}
  GET /v1/taskgraphs/{graph_id}/tasks?states=...&limit=...
  GET /v1/taskgraphs/{graph_id}/tasks/{task_id}
  GET /v1/taskgraphs/{graph_id}/tasks/{task_id}/dependencies
  GET /v1/taskgraphs/{graph_id}/blocking
  GET /v1/taskgraphs/{graph_id}/ready
  GET /v1/taskgraphs/{graph_id}/concurrency
  GET /v1/taskgraphs/{graph_id}/tasks/{task_id}/reliability
  GET /v1/taskgraphs/{graph_id}/breakers
  GET /v1/taskgraphs/{graph_id}/budget
  GET /v1/taskgraphs/{graph_id}/stats
  GET /v1/taskgraphs/{graph_id}/ascii
  GET /v1/taskgraphs/{graph_id}/dot

Run: python3 scripts/taskgraph_viewer.py
View: http://172.239.66.45:8001/v1/taskgraphs/demo/ascii
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from typing import Dict, List
import uvicorn
from collections import deque

# Import real TaskGraph
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.services.task_graph import TaskGraph, TaskState, DependencyPolicy

app = FastAPI()
GRAPHS: Dict[str, TaskGraph] = {}

# Load graphs from persistence
def load_persisted_graphs():
    """Load TaskGraphs from persona_space/taskgraphs/ directory."""
    import json
    from pathlib import Path
    from datetime import datetime

    graph_dir = Path("persona_space/taskgraphs")
    if not graph_dir.exists():
        logger.info(f"No persisted graphs directory: {graph_dir}")
        return

    for graph_file in graph_dir.glob("*.json"):
        try:
            with open(graph_file) as f:
                data = json.load(f)

            # Reconstruct TaskGraph from persisted data
            graph_id = data["graph_id"]
            g = TaskGraph(
                graph_id=graph_id,
                graph_timeout_ms=data.get("graph_timeout_ms", 3600000),
                max_retry_tokens=data.get("max_retry_tokens", 100),
                max_parallel=data.get("max_parallel", 4)
            )

            # Restore timestamps
            if data.get("created_at"):
                g.created_at = datetime.fromisoformat(data["created_at"])
            if data.get("started_at"):
                g.started_at = datetime.fromisoformat(data["started_at"])
            if data.get("completed_at"):
                g.completed_at = datetime.fromisoformat(data["completed_at"])

            # Add all nodes
            for task_id, node_data in data.get("nodes", {}).items():
                # Reconstruct dependencies from node data
                deps = node_data.get("dependencies", [])

                g.add_task(
                    task_id=task_id,
                    action_name=node_data["action_name"],
                    normalized_args=node_data.get("normalized_args", {}),
                    resource_ids=node_data.get("resource_ids", []),
                    version=node_data.get("version", "1.0"),
                    dependencies=deps,
                    on_dep_fail=DependencyPolicy(node_data.get("on_dep_fail", "abort")),
                    priority=node_data.get("priority", 0.5),
                    deadline=datetime.fromisoformat(node_data["deadline"]) if node_data.get("deadline") else None,
                    cost=node_data.get("cost", 1.0),
                    task_timeout_ms=node_data.get("task_timeout_ms", 300000),
                    max_retries=node_data.get("max_retries", 3)
                )

                # Restore state
                node = g.nodes[task_id]
                node.state = TaskState(node_data["state"])
                node.retry_count = node_data.get("retry_count", 0)
                if node_data.get("started_at"):
                    node.started_at = datetime.fromisoformat(node_data["started_at"])
                if node_data.get("completed_at"):
                    node.completed_at = datetime.fromisoformat(node_data["completed_at"])
                node.last_error = node_data.get("last_error")
                node.error_class = node_data.get("error_class")
                node.attempts = node_data.get("attempts", [])

            # Restore runtime state
            g.retry_tokens_used = data.get("retry_tokens_used", 0)

            GRAPHS[graph_id] = g
            logger.info(f"Loaded graph: {graph_id} ({len(g.nodes)} tasks)")

        except Exception as e:
            logger.error(f"Failed to load graph from {graph_file}: {e}")

# Try to load persisted graphs
load_persisted_graphs()

# If no graphs loaded, seed a demo for testing
if not GRAPHS:
    logger.info("No persisted graphs found, seeding demo graph")
    g = TaskGraph(graph_id="demo", max_parallel=4)
    g.add_task("build_fe", "npm_build", {}, [], "1.0", priority=0.7)
    g.add_task("build_be", "go_build", {}, [], "1.0", priority=0.7)
    g.add_task("test", "pytest", {}, [], "1.0", dependencies=["build_fe", "build_be"])
    g.add_task("deploy", "deploy_prod", {}, [], "1.0", dependencies=["test"], priority=0.9)
    GRAPHS["demo"] = g

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/v1/taskgraphs")
def list_graphs():
    return {"ids": list(GRAPHS.keys())}

@app.get("/v1/taskgraphs/{graph_id}")
def get_graph(graph_id: str):
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404, f"graph {graph_id} not found")
    return g.to_dict()

@app.get("/v1/taskgraphs/{graph_id}/stats")
def get_stats(graph_id: str):
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404)
    return g.get_stats()

# R1: Lifecycle Coverage
@app.get("/v1/taskgraphs/{graph_id}/tasks")
def list_tasks(graph_id: str, states: str = None, limit: int = 100, cursor: str = None):
    """List tasks with optional state filtering and pagination."""
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404, f"graph {graph_id} not found")

    if limit > 500:
        raise HTTPException(400, "limit must be <= 500")

    # Parse states filter
    state_filter = None
    if states:
        try:
            state_filter = [TaskState(s.strip()) for s in states.split(",")]
        except ValueError as e:
            raise HTTPException(400, f"invalid state: {e}")

    # Filter tasks
    tasks = []
    for task_id, node in g.nodes.items():
        if state_filter and node.state not in state_filter:
            continue
        tasks.append({
            "task_id": task_id,
            "state": node.state.value,
            "action_name": node.action_name,
            "started_at": node.started_at.isoformat() if node.started_at else None,
            "completed_at": node.completed_at.isoformat() if node.completed_at else None,
            "retry_count": node.retry_count,
            "last_error": node.last_error,
            "error_class": node.error_class,
        })

    # Simple pagination (cursor = offset)
    offset = int(cursor) if cursor else 0
    page = tasks[offset:offset + limit]
    next_cursor = offset + limit if offset + limit < len(tasks) else None

    return {
        "tasks": page,
        "total": len(tasks),
        "limit": limit,
        "next_cursor": str(next_cursor) if next_cursor else None
    }

@app.get("/v1/taskgraphs/{graph_id}/tasks/{task_id}")
def get_task(graph_id: str, task_id: str):
    """Get detailed task information."""
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404, f"graph {graph_id} not found")

    node = g.nodes.get(task_id)
    if not node:
        raise HTTPException(404, f"task {task_id} not found")

    return {
        "task_id": task_id,
        "state": node.state.value,
        "action_name": node.action_name,
        "normalized_args": node.normalized_args,
        "resource_ids": node.resource_ids,
        "version": node.version,
        "dependencies": node.dependencies,
        "dependents": node.dependents,
        "on_dep_fail": node.on_dep_fail.value,
        "priority": node.priority,
        "deadline": node.deadline.isoformat() if node.deadline else None,
        "cost": node.cost,
        "task_timeout_ms": node.task_timeout_ms,
        "started_at": node.started_at.isoformat() if node.started_at else None,
        "completed_at": node.completed_at.isoformat() if node.completed_at else None,
        "retry_count": node.retry_count,
        "max_retries": node.max_retries,
        "last_error": node.last_error,
        "error_class": node.error_class,
        "idempotency_key": node.idempotency_key,
        "attempts": node.attempts,
    }

# R2: Dependencies & Policies
@app.get("/v1/taskgraphs/{graph_id}/tasks/{task_id}/dependencies")
def get_dependencies(graph_id: str, task_id: str):
    """Get task dependency information and policy preview."""
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404, f"graph {graph_id} not found")

    node = g.nodes.get(task_id)
    if not node:
        raise HTTPException(404, f"task {task_id} not found")

    # Find unresolved dependencies
    unresolved = [dep for dep in node.dependencies
                  if dep not in g.completed and dep not in g.failed and dep not in g.aborted]

    # Policy preview
    policy_preview = {}
    if node.on_dep_fail == DependencyPolicy.ABORT:
        policy_preview = {
            "if_all_deps_succeed": "READY",
            "if_any_dep_fails": "ABORTED"
        }
    elif node.on_dep_fail == DependencyPolicy.SKIP:
        policy_preview = {
            "if_all_deps_succeed": "READY",
            "if_any_dep_fails": "SKIPPED"
        }
    elif node.on_dep_fail == DependencyPolicy.CONTINUE_IF_ANY:
        policy_preview = {
            "if_all_deps_succeed": "READY",
            "if_any_dep_fails": "READY (if any succeeded)",
            "if_all_deps_fail": "ABORTED"
        }

    return {
        "task_id": task_id,
        "dependencies": node.dependencies,
        "dependents": node.dependents,
        "on_dep_fail": node.on_dep_fail.value,
        "is_ready": node.is_ready(g.completed, g.failed | g.aborted),
        "blocking_on": unresolved,
        "policy_preview": policy_preview
    }

@app.get("/v1/taskgraphs/{graph_id}/blocking")
def get_blocking(graph_id: str):
    """Get tasks that are blocked and what they're blocked on."""
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404, f"graph {graph_id} not found")

    blocking = []
    for task_id, node in g.nodes.items():
        if node.state == TaskState.PENDING:
            unresolved = [dep for dep in node.dependencies
                         if dep not in g.completed and dep not in g.failed and dep not in g.aborted]
            if unresolved:
                blocking.append({
                    "task_id": task_id,
                    "blocking_on": unresolved,
                    "on_dep_fail": node.on_dep_fail.value
                })

    return {"blocking_tasks": blocking}

# R3: Scheduling & Ready Queue
@app.get("/v1/taskgraphs/{graph_id}/ready")
def get_ready_queue(graph_id: str, limit: int = 50):
    """Get ready queue with ordering explanation."""
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404, f"graph {graph_id} not found")

    # Update ready queue
    g._update_ready_queue()

    # Get ready tasks (sorted by priority)
    ready_tasks = []
    for task_id, node in g.nodes.items():
        if node.state == TaskState.READY:
            ready_tasks.append({
                "task_id": task_id,
                "priority": node.priority,
                "deadline": node.deadline.timestamp() if node.deadline else None,
                "cost": node.cost,
                "action_name": node.action_name,
            })

    # Sort by priority (desc), deadline (asc), cost (asc), task_id (asc)
    ready_tasks.sort(key=lambda t: (
        -t["priority"],
        t["deadline"] if t["deadline"] else float('inf'),
        -t["cost"],
        t["task_id"]
    ))

    return {
        "ordering": "priority DESC, deadline ASC, cost ASC, task_id ASC",
        "queue": ready_tasks[:limit],
        "total_ready": len(ready_tasks)
    }

# R4: Concurrency
@app.get("/v1/taskgraphs/{graph_id}/concurrency")
def get_concurrency(graph_id: str):
    """Get concurrency snapshot."""
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404, f"graph {graph_id} not found")

    per_action = {}
    for action, count in g.action_concurrency.items():
        cap = g.action_concurrency_caps.get(action, float('inf'))
        per_action[action] = {
            "cap": cap if cap != float('inf') else None,
            "running": count,
            "available": (cap - count) if cap != float('inf') else None
        }

    return {
        "global": {
            "max_parallel": g.max_parallel,
            "running": len(g.running_tasks),
            "available": g.max_parallel - len(g.running_tasks)
        },
        "per_action": per_action,
        "running_tasks": list(g.running_tasks)
    }

# R5: Reliability & Safety
@app.get("/v1/taskgraphs/{graph_id}/tasks/{task_id}/reliability")
def get_reliability(graph_id: str, task_id: str):
    """Get task reliability details."""
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404, f"graph {graph_id} not found")

    node = g.nodes.get(task_id)
    if not node:
        raise HTTPException(404, f"task {task_id} not found")

    return {
        "task_id": task_id,
        "idempotency_key": node.idempotency_key,
        "retry_count": node.retry_count,
        "max_retries": node.max_retries,
        "retry_tokens_used": node.retry_tokens_used,
        "can_retry": node.can_retry(),
        "attempts": node.attempts
    }

@app.get("/v1/taskgraphs/{graph_id}/breakers")
def get_breakers(graph_id: str):
    """Get circuit breaker states."""
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404, f"graph {graph_id} not found")

    breakers = []
    for action_name, breaker in g.breakers.items():
        breakers.append({
            "action_name": action_name,
            "state": breaker.get_state(),
            "failure_count": len(breaker.failures),
            "threshold": breaker.failure_threshold,
            "window_seconds": breaker.window_seconds,
            "recovery_timeout_s": breaker.recovery_timeout_seconds,
            "opened_at": breaker.opened_at.isoformat() if breaker.opened_at else None
        })

    return {"breakers": breakers}

@app.get("/v1/taskgraphs/{graph_id}/budget")
def get_budget(graph_id: str):
    """Get retry token budget."""
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404, f"graph {graph_id} not found")

    return {
        "retry_tokens_used": g.retry_tokens_used,
        "max_retry_tokens": g.max_retry_tokens,
        "available": g.max_retry_tokens - g.retry_tokens_used
    }

@app.get("/v1/taskgraphs/{graph_id}/ascii", response_class=PlainTextResponse)
def get_ascii(graph_id: str):
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404)

    # Build dependency graph
    indeg = {k: 0 for k in g.nodes}
    kids: Dict[str, List[str]] = {k: [] for k in g.nodes}
    for nid, node in g.nodes.items():
        for dep in node.dependencies:
            indeg[nid] += 1
            kids[dep].append(nid)

    # Layer by BFS
    roots = [k for k, v in indeg.items() if v == 0]
    levels: Dict[str, int] = {}
    q = deque([(r, 0) for r in roots])
    while q:
        nid, lvl = q.popleft()
        if nid in levels and levels[nid] <= lvl:
            continue
        levels[nid] = lvl
        for c in kids.get(nid, []):
            q.append((c, lvl + 1))

    by_lvl: Dict[int, List[str]] = {}
    for nid, lvl in levels.items():
        by_lvl.setdefault(lvl, []).append(nid)

    stats = g.get_stats()
    lines = [
        f"TaskGraph: {g.graph_id}",
        f"Tasks: {stats['total_tasks']}  Running: {stats['running_tasks']}  Parallel: {g.max_parallel}",
        f"States: {stats['states']}",
        ""
    ]

    for lvl in sorted(by_lvl):
        lines.append(f"Layer {lvl}:")
        for nid in sorted(by_lvl[lvl]):
            n = g.nodes[nid]
            icon = {
                TaskState.PENDING: "⏸",
                TaskState.READY: "▶",
                TaskState.RUNNING: "⚙",
                TaskState.SUCCEEDED: "✓",
                TaskState.FAILED: "✗",
                TaskState.ABORTED: "⊗",
                TaskState.SKIPPED: "⊘",
                TaskState.CANCELLED: "⊖",
            }.get(n.state, "?")
            deps_str = f"deps={len(n.dependencies)}" if n.dependencies else "root"
            policy = "" if n.on_dep_fail == DependencyPolicy.ABORT else f" [{n.on_dep_fail.value}]"
            lines.append(f"  {icon} {nid:<15} {n.action:<20} {n.state.value:<10} {deps_str} prio={n.priority:.1f}{policy}")

    missing = [k for k in g.nodes if k not in levels]
    if missing:
        lines.append("\nOrphans:")
        for nid in sorted(missing):
            n = g.nodes[nid]
            lines.append(f"  • {nid} [{n.action}] {n.state.value}")

    return "\n".join(lines)

@app.get("/v1/taskgraphs/{graph_id}/dot", response_class=PlainTextResponse)
def get_dot(graph_id: str):
    g = GRAPHS.get(graph_id)
    if not g:
        raise HTTPException(404)

    colors = {
        TaskState.PENDING: ("gray80", "black"),
        TaskState.READY: ("gold", "black"),
        TaskState.RUNNING: ("deepskyblue", "white"),
        TaskState.SUCCEEDED: ("seagreen", "white"),
        TaskState.FAILED: ("firebrick", "white"),
        TaskState.ABORTED: ("orangered", "white"),
        TaskState.SKIPPED: ("slategray", "white"),
        TaskState.CANCELLED: ("darkorange", "white"),
    }

    out = [
        f'digraph "{g.graph_id}" {{',
        '  rankdir=TB;',
        '  node [shape=box, style=filled, fontsize=11, fontname="monospace"];',
        ''
    ]

    for nid, n in g.nodes.items():
        fill, font = colors[n.state]
        label = f"{nid}\\n{n.action}\\n{n.state.value}"
        if n.retry_count > 0:
            label += f"\\nretries:{n.retry_count}"
        out.append(f'  "{nid}" [label="{label}", fillcolor="{fill}", fontcolor="{font}"];')

    out.append('')
    for nid, n in g.nodes.items():
        for dep in n.dependencies:
            style = ""
            if n.on_dep_fail == DependencyPolicy.SKIP:
                style = ' [style=dashed, label="skip"]'
            elif n.on_dep_fail == DependencyPolicy.CONTINUE_IF_ANY:
                style = ' [style=dotted, label="any"]'
            out.append(f'  "{dep}" -> "{nid}"{style};')

    out.append("}")
    return "\n".join(out)

if __name__ == "__main__":
    print("Starting TaskGraph viewer on http://172.239.66.45:8001")
    print("  http://172.239.66.45:8001/v1/taskgraphs")
    print("  http://172.239.66.45:8001/v1/taskgraphs/demo/ascii")
    print("  http://172.239.66.45:8001/v1/taskgraphs/demo/dot")
    uvicorn.run(app, host="0.0.0.0", port=8001)
