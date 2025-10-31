"""Identity ledger - append-only audit trail for belief changes and anchor updates.

Provides forensic visibility into identity evolution with:
- PII-redacted event logs
- Daily NDJSON.gz rotation
- Schema versioning
- Thread-safe append operations
"""

import gzip
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
_SCHEMA_VERSION = 1
_LOCK = threading.RLock()


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
    event: str  # "belief_versioned", "anchor_updated", "dissonance_resolved", "kept_fracture_visible"
    strategy: Optional[str] = None  # "Commit" | "Reframe" | "Nuance" | "Defer"
    beliefs_touched: Optional[List[str]] = None
    cost_named: Optional[str] = None
    sims_before_after: Optional[Dict[str, float]] = None  # {"sim_live_before":0.71,"sim_live_after":0.73,"sim_origin":0.66}
    meta: Optional[Dict[str, Any]] = None  # freeform, scrubbed


def append_event(ev: LedgerEvent) -> None:
    """Append event to daily ledger with PII scrubbing."""
    rec = asdict(ev)
    rec["schema"] = _SCHEMA_VERSION
    rec = _scrub(rec)
    day = _day_stamp(ev.ts)
    path = _file_for_day(day)
    line = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")
    with _LOCK:
        with _open_gz_append(path) as f:
            f.write(line)


def anchor_update_event(
    sim_live_before: float,
    sim_live_after: float,
    sim_origin: float,
    strategy: str,
    beliefs_touched: List[str],
    cost_named: str = ""
) -> None:
    """Log anchor update event."""
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
        meta=None,
    )
    append_event(ev)


def belief_versioned_event(
    belief_id: str,
    prev_version: int,
    new_version: int,
    reason_changed: str,
    confidence: float,
    cause: str
) -> None:
    """Log belief version change."""
    ev = LedgerEvent(
        ts=time.time(),
        schema=_SCHEMA_VERSION,
        event="belief_versioned",
        strategy=None,
        beliefs_touched=[belief_id],
        cost_named=None,
        sims_before_after=None,
        meta={
            "prev_version": prev_version,
            "new_version": new_version,
            "reason_changed": reason_changed,
            "confidence": round(confidence, 3),
            "cause": cause
        }
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
