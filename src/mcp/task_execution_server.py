"""Model Context Protocol server for task execution introspection.

This module wires task execution data stored in the raw store into three MCP
tools described in `.claude/tasks/prompt-008-mcp-task-execution.md`:

- ``tasks_list`` – filtered history queries
- ``tasks_by_trace`` – drill into a specific trace/attempt chain
- ``tasks_last_failed`` – surface recent failures and aggregate error patterns

The heavy lifting (filtering, shaping, summarising) is implemented in
``TaskExecutionTooling`` so it can be exercised directly in unit tests without
requiring an MCP runtime.  The small adaptor in ``create_task_execution_server``
registers those helpers with the official ``modelcontextprotocol`` server when
the library is available.  Environments that do not have the dependency
installed can still import this module—the optional import is handled
gracefully—but attempting to start the MCP server will raise a clear runtime
error so operators know how to resolve it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.memory.raw_store import ExperienceModel, RawStore

try:  # modelcontextprotocol is optional for testing environments
    from modelcontextprotocol.server import Server
    from modelcontextprotocol.types import TextContent, ToolResult

    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dependency missing
    Server = None  # type: ignore[assignment]
    TextContent = None  # type: ignore[assignment]
    ToolResult = None  # type: ignore[assignment]
    _MCP_AVAILABLE = False


def _parse_iso8601(timestamp: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 timestamps from stored structured content."""

    if not timestamp:
        return None

    # Normalise trailing Z to +00:00 for datetime.fromisoformat compatibility
    normalised = timestamp.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalised)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _as_utc_iso(dt: Optional[datetime]) -> Optional[str]:
    """Serialise datetime objects as ISO-8601 strings with UTC offset."""

    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _clean_error(error: Any) -> Optional[dict[str, Any]]:
    """Standardise error payloads to the schema expected by clients."""

    if not isinstance(error, dict):
        return None

    cleaned = {
        "type": error.get("type"),
        "message": error.get("message"),
        "stack_hash": error.get("stack_hash"),
    }

    if not any(cleaned.values()):
        return None
    return cleaned


def _boolish(value: Any) -> bool:
    """Interpret values that may be stored as bool/int/string in SQLite JSON."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return False


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


@dataclass(frozen=True)
class TaskExecutionRecord:
    """Lightweight serialisable representation of a task execution."""

    id: str
    task_id: Optional[str]
    task_slug: Optional[str]
    task_name: Optional[str]
    status: Optional[str]
    started_at: Optional[str]
    ended_at: Optional[str]
    duration_ms: Optional[int]
    trace_id: Optional[str]
    span_id: Optional[str]
    attempt: Optional[int]
    error: Optional[dict[str, Any]]
    retrieval: dict[str, Any]
    io: dict[str, Any]
    backfilled: bool
    task_type: Optional[str] = None
    scheduled_vs_manual: Optional[str] = None

    @classmethod
    def from_experience(cls, experience: ExperienceModel) -> "TaskExecutionRecord":
        structured = dict(getattr(experience.content, "structured", {}) or {})
        retrieval_raw = structured.get("retrieval") or {}
        io_raw = structured.get("io") or {}

        started_iso = structured.get("started_at_iso")
        ended_iso = structured.get("ended_at_iso")

        return cls(
            id=experience.id,
            task_id=structured.get("task_id"),
            task_slug=structured.get("task_slug"),
            task_name=structured.get("task_name"),
            status=structured.get("status"),
            started_at=started_iso or _as_utc_iso(experience.created_at),
            ended_at=ended_iso,
            duration_ms=structured.get("duration_ms"),
            trace_id=structured.get("trace_id"),
            span_id=structured.get("span_id"),
            attempt=structured.get("attempt"),
            error=_clean_error(structured.get("error")),
            retrieval={
                "memory_count": retrieval_raw.get("memory_count", 0),
                "source": _ensure_list(retrieval_raw.get("source")),
            },
            io={
                "files_written": _ensure_list(io_raw.get("files_written")),
            },
            backfilled=_boolish(structured.get("backfilled")),
            task_type=structured.get("task_type"),
            scheduled_vs_manual=structured.get("scheduled_vs_manual"),
        )


class TaskExecutionTooling:
    """Pure-Python helpers backing the MCP tools."""

    def __init__(self, raw_store: RawStore):
        self.raw_store = raw_store

    # ------------------------------------------------------------------
    # Public API consumed by MCP tool handlers
    # ------------------------------------------------------------------
    def tasks_list(
        self,
        *,
        task_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        since: Optional[str] = None,
        backfilled: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Return a filtered list of task executions."""

        limit = self._clamp(limit, 1, 100)
        status_normalised = self._normalise_status(status)
        since_dt = _parse_iso8601(since) if since else None

        fetch_limit = min(limit + 1, 101)
        experiences = self.raw_store.list_task_executions(
            task_id=task_id,
            status=status_normalised,
            since=since_dt,
            backfilled=backfilled,
            limit=fetch_limit,
        )

        has_more = len(experiences) > limit
        if has_more:
            experiences = experiences[:limit]

        executions = [TaskExecutionRecord.from_experience(exp) for exp in experiences]

        return {
            "executions": [asdict(record) for record in executions],
            "total": len(executions),
            "has_more": has_more,
        }

    def tasks_by_trace(self, *, trace_id: str) -> dict[str, Any]:
        """Return all attempts that share a correlation trace."""

        if not trace_id:
            raise ValueError("trace_id is required")

        experiences = self.raw_store.get_by_trace_id(trace_id)
        executions = [TaskExecutionRecord.from_experience(exp) for exp in experiences]

        retry_count = 0
        final_status = None
        if executions:
            attempts = [rec.attempt or 0 for rec in executions]
            if attempts:
                retry_count = max(attempts) - 1
            final_status = executions[-1].status

        return {
            "executions": [asdict(record) for record in executions],
            "retry_count": max(retry_count, 0),
            "final_status": final_status,
        }

    def tasks_last_failed(
        self,
        *,
        limit: int = 10,
        task_id: Optional[str] = None,
        unique_errors: bool = False,
    ) -> dict[str, Any]:
        """Return recent failed executions with optional error deduplication."""

        limit = self._clamp(limit, 1, 50)

        fetch_limit = min(101, max(limit * 3, limit + 5))
        raw_failures = self.raw_store.list_task_executions(
            task_id=task_id,
            status="failed",
            backfilled=None,
            limit=fetch_limit,
        )

        summaries = self._summarise_failures(raw_failures)
        failures = summaries["failures"]

        if unique_errors:
            deduped: list[TaskExecutionRecord] = []
            seen_hashes: set[str] = set()
            for record in failures:
                stack_hash = (record.error or {}).get("stack_hash") or "<unknown>"
                if stack_hash in seen_hashes:
                    continue
                seen_hashes.add(stack_hash)
                deduped.append(record)
                if len(deduped) >= limit:
                    break
            failures = deduped
        else:
            failures = failures[:limit]

        patterns = self._build_error_patterns(summaries["all_failures"])

        return {
            "failures": [asdict(record) for record in failures],
            "error_patterns": patterns,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(value: int, lower: int, upper: int) -> int:
        if value < lower:
            return lower
        if value > upper:
            return upper
        return value

    @staticmethod
    def _normalise_status(status: Optional[str]) -> Optional[str]:
        if status is None:
            return None
        normalised = status.lower()
        if normalised not in {"success", "failed"}:
            raise ValueError("status must be 'success' or 'failed'")
        return normalised

    @staticmethod
    def _summarise_failures(
        experiences: Iterable[ExperienceModel],
    ) -> dict[str, list[TaskExecutionRecord]]:
        failures = [TaskExecutionRecord.from_experience(exp) for exp in experiences]
        return {
            "failures": failures,
            "all_failures": failures,
        }

    @staticmethod
    def _build_error_patterns(
        failures: Iterable[TaskExecutionRecord],
    ) -> list[dict[str, Any]]:
        patterns: dict[str, dict[str, Any]] = {}

        for record in failures:
            error = record.error or {}
            stack_hash = error.get("stack_hash") or "<unknown>"
            started_dt = _parse_iso8601(record.started_at)

            entry = patterns.setdefault(
                stack_hash,
                {
                    "stack_hash": stack_hash,
                    "count": 0,
                    "first_seen": None,
                    "last_seen": None,
                    "example_task_id": record.task_id,
                    "example_trace_id": record.trace_id,
                },
            )

            entry["count"] += 1

            if started_dt is not None:
                first_seen = _parse_iso8601(entry["first_seen"])
                last_seen = _parse_iso8601(entry["last_seen"])

                if first_seen is None or started_dt < first_seen:
                    entry["first_seen"] = _as_utc_iso(started_dt)
                if last_seen is None or started_dt > last_seen:
                    entry["last_seen"] = _as_utc_iso(started_dt)

            if entry["example_task_id"] is None:
                entry["example_task_id"] = record.task_id
            if entry["example_trace_id"] is None:
                entry["example_trace_id"] = record.trace_id

        # Sort predictable for clients: most recent last_seen descending
        sorted_patterns = sorted(
            patterns.values(),
            key=lambda item: (
                item["last_seen"] or "",
                item["count"],
            ),
            reverse=True,
        )

        return sorted_patterns


def create_task_execution_server(raw_store: RawStore) -> "Server":
    """Create and configure the MCP server for task execution tooling."""

    if not _MCP_AVAILABLE:  # pragma: no cover - depends on external package
        raise RuntimeError(
            "modelcontextprotocol is not installed. Install it to run the MCP server."
        )

    tooling = TaskExecutionTooling(raw_store)
    server = Server("astra-task-executions")

    @server.tool(name="tasks_list", description="List task executions with filters")
    async def tasks_list(payload: Optional[dict[str, Any]] = None) -> ToolResult:
        payload = payload or {}
        result = tooling.tasks_list(
            task_id=payload.get("task_id"),
            status=payload.get("status"),
            limit=payload.get("limit", 20),
            since=payload.get("since"),
            backfilled=payload.get("backfilled"),
        )
        return ToolResult(content=[TextContent(type="text", text=json.dumps(result))])

    @server.tool(name="tasks_by_trace", description="Inspect a task execution trace")
    async def tasks_by_trace(payload: Optional[dict[str, Any]] = None) -> ToolResult:
        payload = payload or {}
        result = tooling.tasks_by_trace(trace_id=payload.get("trace_id"))
        return ToolResult(content=[TextContent(type="text", text=json.dumps(result))])

    @server.tool(
        name="tasks_last_failed",
        description="List recent task failures and summarise error patterns",
    )
    async def tasks_last_failed(payload: Optional[dict[str, Any]] = None) -> ToolResult:
        payload = payload or {}
        result = tooling.tasks_last_failed(
            limit=payload.get("limit", 10),
            task_id=payload.get("task_id"),
            unique_errors=payload.get("unique_errors", False),
        )
        return ToolResult(content=[TextContent(type="text", text=json.dumps(result))])

    return server


__all__ = [
    "TaskExecutionTooling",
    "TaskExecutionRecord",
    "create_task_execution_server",
]
