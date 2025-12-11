"""
TaskGraph Query SDK - Python client for TaskGraph API.

Usage:
    from taskgraph_client import TaskGraphClient

    client = TaskGraphClient("http://localhost:8001")

    # List graphs
    graphs = client.list_graphs()

    # Get stats
    stats = client.get_stats("demo")

    # Query tasks
    tasks = client.list_tasks("demo", states=["READY", "RUNNING"])
    task = client.get_task("demo", "deploy")

    # Dependencies
    deps = client.get_dependencies("demo", "deploy")
    blocking = client.get_blocking("demo")

    # Scheduling
    ready = client.get_ready_queue("demo", limit=10)

    # Concurrency
    conc = client.get_concurrency("demo")

    # Reliability
    rel = client.get_reliability("demo", "deploy")
    breakers = client.get_breakers("demo")
    budget = client.get_budget("demo")

    # Visualization
    ascii_view = client.get_ascii("demo")
    dot_graph = client.get_dot("demo")
"""

import requests
from typing import List, Optional, Dict, Any


class TaskGraphClient:
    """Python SDK for TaskGraph Query API."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        """Make GET request and return JSON."""
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _get_text(self, path: str, params: Optional[Dict] = None) -> str:
        """Make GET request and return text."""
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.text

    # Core
    def healthz(self) -> Dict:
        """Check API health."""
        return self._get("/healthz")

    def list_graphs(self) -> Dict:
        """List all graphs."""
        return self._get("/v1/taskgraphs")

    def get_graph(self, graph_id: str) -> Dict:
        """Get full graph snapshot."""
        return self._get(f"/v1/taskgraphs/{graph_id}")

    def get_stats(self, graph_id: str) -> Dict:
        """Get graph statistics."""
        return self._get(f"/v1/taskgraphs/{graph_id}/stats")

    # R1: Lifecycle
    def list_tasks(
        self,
        graph_id: str,
        states: Optional[List[str]] = None,
        limit: int = 100,
        cursor: Optional[str] = None
    ) -> Dict:
        """List tasks with optional filtering."""
        params = {"limit": limit}
        if states:
            params["states"] = ",".join(states)
        if cursor:
            params["cursor"] = cursor
        return self._get(f"/v1/taskgraphs/{graph_id}/tasks", params)

    def get_task(self, graph_id: str, task_id: str) -> Dict:
        """Get detailed task information."""
        return self._get(f"/v1/taskgraphs/{graph_id}/tasks/{task_id}")

    # R2: Dependencies
    def get_dependencies(self, graph_id: str, task_id: str) -> Dict:
        """Get task dependencies and policy."""
        return self._get(f"/v1/taskgraphs/{graph_id}/tasks/{task_id}/dependencies")

    def get_blocking(self, graph_id: str) -> Dict:
        """Get tasks that are blocked."""
        return self._get(f"/v1/taskgraphs/{graph_id}/blocking")

    # R3: Scheduling
    def get_ready_queue(self, graph_id: str, limit: int = 50) -> Dict:
        """Get ready queue with ordering."""
        return self._get(f"/v1/taskgraphs/{graph_id}/ready", {"limit": limit})

    # R4: Concurrency
    def get_concurrency(self, graph_id: str) -> Dict:
        """Get concurrency snapshot."""
        return self._get(f"/v1/taskgraphs/{graph_id}/concurrency")

    # R5: Reliability
    def get_reliability(self, graph_id: str, task_id: str) -> Dict:
        """Get task reliability details."""
        return self._get(f"/v1/taskgraphs/{graph_id}/tasks/{task_id}/reliability")

    def get_breakers(self, graph_id: str) -> Dict:
        """Get circuit breaker states."""
        return self._get(f"/v1/taskgraphs/{graph_id}/breakers")

    def get_budget(self, graph_id: str) -> Dict:
        """Get retry token budget."""
        return self._get(f"/v1/taskgraphs/{graph_id}/budget")

    # Visualization
    def get_ascii(self, graph_id: str) -> str:
        """Get ASCII visualization."""
        return self._get_text(f"/v1/taskgraphs/{graph_id}/ascii")

    def get_dot(self, graph_id: str) -> str:
        """Get DOT graph for GraphViz."""
        return self._get_text(f"/v1/taskgraphs/{graph_id}/dot")


if __name__ == "__main__":
    # Example usage
    client = TaskGraphClient("http://localhost:8001")

    print("=== Health Check ===")
    print(client.healthz())

    print("\n=== List Graphs ===")
    graphs = client.list_graphs()
    print(graphs)

    if graphs["ids"]:
        graph_id = graphs["ids"][0]
        print(f"\n=== Graph: {graph_id} ===")

        print("\nStats:")
        stats = client.get_stats(graph_id)
        print(stats)

        print("\nReady Queue:")
        ready = client.get_ready_queue(graph_id)
        print(ready)

        print("\nConcurrency:")
        conc = client.get_concurrency(graph_id)
        print(conc)

        print("\nASCII View:")
        print(client.get_ascii(graph_id))
