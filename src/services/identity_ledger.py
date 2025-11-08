"""Identity ledger - append-only audit trail for belief changes and anchor updates.

Provides forensic visibility into identity evolution with:
- PII-redacted event logs
- Daily NDJSON.gz rotation
- Schema versioning
- Thread-safe append operations
- Rolling SHA-256 chain for integrity verification
"""

import gzip
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_LEDGER_DIR = Path(os.getenv("ASTRA_LEDGER_DIR", "data/identity"))
_LEDGER_DIR.mkdir(parents=True, exist_ok=True)
_SCHEMA_VERSION = 2  # Incremented for SHA chain support
_LOCK = threading.RLock()
_CHAIN_STATE_FILE = _LEDGER_DIR / "chain_state.json"

# In-memory cache of last SHA for current day
_last_sha_cache: Dict[str, str] = {}


def _day_stamp(ts: Optional[float] = None) -> str:
    """Get YYYYMMDD stamp for a timestamp."""
    import datetime as dt
    t = dt.datetime.utcfromtimestamp(ts or time.time())
    return t.strftime("%Y%m%d")


def _file_for_day(day: str) -> Path:
    """Get ledger file path for a given day."""
    return _LEDGER_DIR / f"ledger-{day}.ndjson.gz"


def _open_gz_append(path: Path):
    """Open gzip file for appending."""
    return gzip.open(path, "ab")


def _scrub(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Basic PII redaction in free-text fields."""
    def red(s: str) -> str:
        s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", s)
        s = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "[ip]", s)
        s = re.sub(r"\b[A-Z0-9]{6,}\b", "[id]", s)  # crude
        return s

    out = {}
    for k, v in obj.items():
        if isinstance(v, str):
            out[k] = red(v)
        elif isinstance(v, list):
            out[k] = [red(x) if isinstance(x, str) else x for x in v]
        elif isinstance(v, dict):
            out[k] = _scrub(v)
        else:
            out[k] = v
    return out


@dataclass
class LedgerEvent:
    """Single event in the identity ledger."""
    ts: float
    schema: int
    event: str  # "belief_versioned", "anchor_updated", "dissonance_resolved", "kept_fracture_visible", "decision_made", "decision_aborted", "parameter_adapted"
    strategy: Optional[str] = None  # "Commit" | "Reframe" | "Nuance" | "Defer"
    beliefs_touched: Optional[List[str]] = None
    cost_named: Optional[str] = None
    sims_before_after: Optional[Dict[str, float]] = None  # {"sim_live_before":0.71,"sim_live_after":0.73,"sim_origin":0.66}
    belief_ver_from: Optional[int] = None  # For belief_versioned events
    belief_ver_to: Optional[int] = None
    evidence_refs: Optional[List[str]] = None
    coherence_drop: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None  # freeform, scrubbed
    prev_sha: Optional[str] = None  # SHA of previous entry in chain
    sha: Optional[str] = None  # SHA of this entry (computed after serialization)

    # Decision framework fields
    decision_id: Optional[str] = None  # Type of decision (belief_formation, belief_promotion, etc.)
    decision_record_id: Optional[str] = None  # Unique record ID for this decision
    parameters_used: Optional[Dict[str, float]] = None  # Parameters at time of decision
    success_score: Optional[float] = None  # Success score from outcome evaluation
    abort_reason: Optional[str] = None  # Reason for abort if applicable


def _compute_sha(rec: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of ledger record (excluding sha field)."""
    data = {k: v for k, v in rec.items() if k != "sha"}
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _get_chain_tail(day: str) -> Optional[str]:
    """Get SHA of last entry in day's ledger."""
    # Check cache first
    if day in _last_sha_cache:
        return _last_sha_cache[day]

    # Read from file
    path = _file_for_day(day)
    if not path.exists():
        return None

    try:
        with gzip.open(path, "rt") as f:
            last_line = None
            for line in f:
                last_line = line

            if last_line:
                rec = json.loads(last_line)
                return rec.get("sha")
    except Exception:
        return None

    return None


def _get_prev_day_tail(day: str) -> Optional[str]:
    """Get SHA of last entry from previous day."""
    import datetime as dt
    d = dt.datetime.strptime(day, "%Y%m%d")
    prev_d = d - dt.timedelta(days=1)
    prev_day = prev_d.strftime("%Y%m%d")
    return _get_chain_tail(prev_day)


def append_event(ev: LedgerEvent) -> None:
    """Append event to daily ledger with PII scrubbing and SHA chain."""
    day = _day_stamp(ev.ts)

    with _LOCK:
        # Get previous SHA (from cache or last entry in current day, or previous day tail)
        prev_sha = _get_chain_tail(day)
        if prev_sha is None:
            # First entry of the day - link to previous day
            prev_sha = _get_prev_day_tail(day) or "genesis"

        rec = asdict(ev)
        rec["schema"] = _SCHEMA_VERSION
        rec["prev_sha"] = prev_sha
        rec = _scrub(rec)

        # Compute SHA
        sha = _compute_sha(rec)
        rec["sha"] = sha

        # Write to file
        path = _file_for_day(day)
        line = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")
        with _open_gz_append(path) as f:
            f.write(line)

        # Update cache
        _last_sha_cache[day] = sha


def anchor_update_event(
    sim_live_before: float,
    sim_live_after: float,
    sim_origin: float,
    strategy: str,
    beliefs_touched: List[str],
    cost_named: str = "",
    coherence_drop: float = 0.0,
    evidence_refs: Optional[List[str]] = None,
) -> None:
    """Log anchor update event with belief linking."""
    ev = LedgerEvent(
        ts=time.time(),
        schema=_SCHEMA_VERSION,
        event="anchor_updated",
        strategy=strategy,
        beliefs_touched=beliefs_touched,
        cost_named=cost_named or None,
        sims_before_after={
            "sim_live_before": round(sim_live_before, 3),
            "sim_live_after": round(sim_live_after, 3),
            "sim_origin": round(sim_origin, 3)
        },
        coherence_drop=round(coherence_drop, 3) if coherence_drop else None,
        evidence_refs=evidence_refs,
        meta=None,
        prev_sha=None,  # Will be filled by append_event
        sha=None,
    )
    append_event(ev)


def belief_versioned_event(
    belief_id: str,
    from_ver: int,
    to_ver: int,
    reason_changed: str,
    confidence: float,
    cause: str,
    evidence_refs: Optional[List[str]] = None,
    sim_live: Optional[float] = None,
    sim_origin: Optional[float] = None,
    coherence_drop: Optional[float] = None,
) -> None:
    """Log belief version change with full context."""
    sims = None
    if sim_live is not None and sim_origin is not None:
        sims = {
            "sim_live": round(sim_live, 3),
            "sim_origin": round(sim_origin, 3),
        }

    ev = LedgerEvent(
        ts=time.time(),
        schema=_SCHEMA_VERSION,
        event="belief_versioned",
        strategy=None,
        beliefs_touched=[belief_id],
        cost_named=None,
        sims_before_after=sims,
        belief_ver_from=from_ver,
        belief_ver_to=to_ver,
        evidence_refs=evidence_refs,
        coherence_drop=round(coherence_drop, 3) if coherence_drop else None,
        meta={
            "reason_changed": reason_changed,
            "confidence": round(confidence, 3),
            "cause": cause,
        },
        prev_sha=None,  # Will be filled by append_event
        sha=None,
    )
    append_event(ev)


def kept_fracture_visible_event(beliefs_touched: List[str], note: str = "") -> None:
    """Log when fracture was kept visible instead of forced resolution."""
    ev = LedgerEvent(
        ts=time.time(),
        schema=_SCHEMA_VERSION,
        event="kept_fracture_visible",
        strategy=None,
        beliefs_touched=beliefs_touched or None,
        cost_named=None,
        sims_before_after=None,
        meta={"note": note[:256]} if note else None
    )
    append_event(ev)

# Decision Framework Audit Logging

def decision_made_event(
    decision_id: str,
    decision_record_id: str,
    parameters_used: Dict[str, float],
    beliefs_touched: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a decision made by the adaptive framework.

    Args:
        decision_id: Type of decision (belief_formation, belief_promotion, etc.)
        decision_record_id: Unique record ID for this decision
        parameters_used: Parameters used to make the decision
        beliefs_touched: Belief IDs affected by this decision
        meta: Additional context (e.g., evidence_count, feedback_score)
    """
    ev = LedgerEvent(
        ts=time.time(),
        schema=_SCHEMA_VERSION,
        event="decision_made",
        decision_id=decision_id,
        decision_record_id=decision_record_id,
        parameters_used=parameters_used,
        beliefs_touched=beliefs_touched,
        meta=meta,
        prev_sha=None,  # Will be filled by append_event
        sha=None,
    )
    append_event(ev)


def decision_aborted_event(
    abort_reason: str,
    decision_id: Optional[str] = None,
    coherence_drop: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log when abort conditions trigger and halt autonomous decisions.

    Args:
        abort_reason: Reason for abort (e.g., "dissonance_spike", "coherence_drop")
        decision_id: Type of decision that was aborted (if applicable)
        coherence_drop: Amount of coherence drop if relevant
        meta: Additional context (e.g., threshold values, current metrics)
    """
    ev = LedgerEvent(
        ts=time.time(),
        schema=_SCHEMA_VERSION,
        event="decision_aborted",
        abort_reason=abort_reason,
        decision_id=decision_id,
        coherence_drop=round(coherence_drop, 3) if coherence_drop else None,
        meta=meta,
        prev_sha=None,
        sha=None,
    )
    append_event(ev)


def parameter_adapted_event(
    decision_id: str,
    parameters_updated: Dict[str, Dict[str, float]],  # {param_name: {"old": 0.5, "new": 0.6}}
    success_score: float,
    sample_count: int,
    meta: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log when parameters are adapted based on outcome evaluation.

    Args:
        decision_id: Type of decision whose parameters were adapted
        parameters_updated: Map of parameter names to old/new values
        success_score: Average success score that triggered adaptation
        sample_count: Number of decisions evaluated
        meta: Additional context (e.g., reason, adaptation_method)
    """
    ev = LedgerEvent(
        ts=time.time(),
        schema=_SCHEMA_VERSION,
        event="parameter_adapted",
        decision_id=decision_id,
        success_score=round(success_score, 3),
        meta={
            "parameters_updated": parameters_updated,
            "sample_count": sample_count,
            **(meta or {})
        },
        prev_sha=None,
        sha=None,
    )
    append_event(ev)
