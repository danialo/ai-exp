"""Helper module for creating TASK_EXECUTION experiences with full correlation and idempotency.

This module provides utilities to convert task execution metadata into properly
structured TASK_EXECUTION experiences that can be stored in the raw store.
"""

import hashlib
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.memory.models import (
    Actor,
    AffectModel,
    CaptureMethod,
    ContentModel,
    ExperienceModel,
    ExperienceType,
    ProvenanceModel,
)


def _generate_idempotency_key(task_id: str, scheduled_at_iso: str, attempt: int = 1) -> str:
    """Generate idempotency key for task execution.

    Args:
        task_id: Task identifier
        scheduled_at_iso: ISO timestamp of when task was scheduled
        attempt: Attempt number (1-n for retries)

    Returns:
        SHA256 hash of task_id + scheduled_at + attempt
    """
    composite = f"{task_id}:{scheduled_at_iso}:{attempt}"
    return hashlib.sha256(composite.encode()).hexdigest()


def _generate_task_config_digest(task_config: Dict[str, Any]) -> str:
    """Generate digest of task configuration for provenance.

    Args:
        task_config: Task configuration dictionary

    Returns:
        SHA256 hash of task config
    """
    # Sort keys for deterministic hashing
    config_str = str(sorted(task_config.items()))
    digest = hashlib.sha256(config_str.encode()).hexdigest()
    return f"sha256:{digest[:16]}"  # Truncate for readability


def _scrub_pii(text: str) -> str:
    """Basic PII scrubbing for task responses.

    Uses same patterns as identity ledger.

    Args:
        text: Text to scrub

    Returns:
        Scrubbed text
    """
    import re

    # Email addresses
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", text)

    # IP addresses
    text = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "[ip]", text)

    # Long IDs (likely UUIDs or session tokens)
    text = re.sub(r"\b[a-f0-9]{32,}\b", "[id]", text, flags=re.IGNORECASE)

    return text


def create_task_execution_experience(
    task_id: str,
    task_slug: str,
    task_name: str,
    task_type: str,
    scheduled_vs_manual: str,
    started_at: datetime,
    ended_at: datetime,
    status: str,
    response_text: Optional[str] = None,
    error: Optional[Dict[str, str]] = None,
    parent_experience_ids: Optional[List[str]] = None,
    retrieval_metadata: Optional[Dict[str, Any]] = None,
    files_written: Optional[List[str]] = None,
    tool_calls: Optional[List[str]] = None,
    script_runs: Optional[List[str]] = None,
    task_config: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    attempt: int = 1,
    retry_of: Optional[str] = None,
) -> ExperienceModel:
    """Create a TASK_EXECUTION experience from task execution metadata.

    This follows the concrete field contract from the Phase 1 specification.

    Args:
        task_id: Unique task identifier
        task_slug: Human-readable task slug for queries
        task_name: Display name of task
        task_type: Type of task (reflection, assessment, ingest, custom)
        scheduled_vs_manual: "scheduled" or "manual"
        started_at: Task execution start time (UTC)
        ended_at: Task execution end time (UTC)
        status: Execution status ("success", "failed", "partial")
        response_text: Task response text (will be PII scrubbed)
        error: Error details if status="failed" (type, message, stack_hash)
        parent_experience_ids: Retrieved memory IDs used as context
        retrieval_metadata: Retrieval provenance (query, filters, latency_ms, etc.)
        files_written: List of file paths written during execution
        tool_calls: List of tool call IDs (references to tool_call table)
        script_runs: List of script run IDs (references to script_run table)
        task_config: Task configuration for digest generation
        trace_id: Correlation ID (stable across retries), auto-generated if None
        span_id: Span ID (unique per attempt), auto-generated if None
        attempt: Attempt number (1-n for retries)
        retry_of: Span ID of prior attempt (if this is a retry)

    Returns:
        ExperienceModel ready to be inserted into raw store
    """
    # Ensure UTC-aware timestamps
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    if ended_at.tzinfo is None:
        ended_at = ended_at.replace(tzinfo=timezone.utc)

    # Generate correlation IDs if not provided
    if trace_id is None:
        trace_id = str(uuid4())
    if span_id is None:
        span_id = str(uuid4())

    # Calculate duration (ensure non-negative)
    duration_ms = int(max((ended_at - started_at).total_seconds() * 1000, 0))

    # ISO timestamps
    started_at_iso = started_at.isoformat()
    ended_at_iso = ended_at.isoformat()

    # Timestamps as float seconds for SQL
    started_at_ts = started_at.timestamp()
    ended_at_ts = ended_at.timestamp()

    # Generate idempotency key
    idempotency_key = _generate_idempotency_key(task_id, started_at_iso, attempt)

    # Generate task config digest
    task_config_digest = _generate_task_config_digest(task_config or {})

    # Generate short UUID for experience ID
    short_uuid = str(uuid4())[-4:]
    timestamp_str = started_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    experience_id = f"task_exec_{task_slug}_{timestamp_str}_{short_uuid}"

    # Session ID format: task:{task_id}:{exec_ts_iso}:{short_uuid}
    session_id = f"task:{task_id}:{started_at_iso}:{short_uuid}"

    # Scrub PII from response text
    text_content = response_text or ""
    if text_content:
        text_content = _scrub_pii(text_content)

    # Build retrieval metadata
    retrieval = retrieval_metadata or {}
    if not retrieval:
        # Default empty retrieval
        retrieval = {
            "memory_count": 0,
            "source": []
        }

    # Ensure parent_experience_ids matches retrieval.memory_count
    parents = parent_experience_ids or []
    if retrieval.get("memory_count", 0) != len(parents):
        # Log warning but don't fail - adjust memory_count to match parents
        retrieval["memory_count"] = len(parents)

    # Build io metadata
    io_metadata = {
        "files_written": files_written or [],
        "artifact_ids": [],  # Future use
        "tool_calls": tool_calls or [],
        "script_runs": script_runs or [],
    }

    # Build provenance_runner metadata
    provenance_runner = {
        "name": "scheduler",
        "version": "1.0.0",
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }

    # Build grants metadata (stub for now)
    grants = {
        "scopes": [],
        "impersonation": False,
    }

    # Build structured content
    structured = {
        # Schema versioning
        "schema_version": 1,

        # Core identity
        "task_id": task_id,
        "task_slug": task_slug,
        "task_name": task_name,
        "task_type": task_type,
        "scheduled_vs_manual": scheduled_vs_manual,

        # Status with dual timestamps
        "status": status,
        "started_at_iso": started_at_iso,
        "ended_at_iso": ended_at_iso,
        "started_at_ts": started_at_ts,
        "ended_at_ts": ended_at_ts,
        "duration_ms": duration_ms,

        # Correlation
        "trace_id": trace_id,
        "span_id": span_id,
        "attempt": attempt,
        "retry_of": retry_of,
        "idempotency_key": idempotency_key,

        # Config provenance
        "task_config_digest": task_config_digest,

        # Retrieval
        "retrieval": retrieval,

        # Side effects
        "io": io_metadata,

        # Embeddings flag
        "embedding_skipped": True,  # Phase 1: always skip embeddings

        # Scheduler identity
        "provenance_runner": provenance_runner,

        # Permissions
        "grants": grants,

        # Error (non-null on failure)
        "error": error,

        # PII scrubbing flag
        "scrubbed": True,
    }

    # Build content model
    content = ContentModel(
        text=text_content,
        structured=structured,
    )

    # Build provenance
    capture_method = CaptureMethod.SCHEDULED_TASK if scheduled_vs_manual == "scheduled" else CaptureMethod.MANUAL_TASK
    provenance = ProvenanceModel(
        actor=Actor.AGENT,
        method=capture_method,
        sources=[],
    )

    # Build experience model
    experience = ExperienceModel(
        id=experience_id,
        type=ExperienceType.TASK_EXECUTION,
        created_at=started_at,
        content=content,
        provenance=provenance,
        parents=parents,
        causes=[],  # Empty for Phase 1, used in Phase 4
        session_id=session_id,
    )

    return experience
