"""Dual-anchor identity system.

Maintains two anchors:
- origin: Fixed reference to initial identity (never updates)
- live: Slow EMA that tracks genuine evolution (updates on resolution)

This prevents:
- False coherence (penalizing healthy growth)
- Presence-chasing (rewarding chaos)
- Identity lock-in (frozen to day-1 personality)
"""

import time
from dataclasses import dataclass
from typing import List

import numpy as np

from src.services.identity_ledger import anchor_update_event


@dataclass
class Anchors:
    """Dual-anchor system with origin (fixed) and live (EMA)."""
    origin: np.ndarray  # Fixed reference to initial identity
    live: np.ndarray  # Slow EMA tracking genuine evolution
    last_update_ts: float = 0.0  # Last anchor update timestamp


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    an = a / (np.linalg.norm(a) + 1e-9)
    bn = b / (np.linalg.norm(b) + 1e-9)
    return float(np.clip(an @ bn, -1, 1))


def update_live_anchor(
    anchors: Anchors,
    cur_vec: np.ndarray,
    strategy: str,
    beliefs_touched: List[str],
    cfg_beta_week_cap: float = 0.01
) -> Anchors:
    """Update live anchor with time-based EMA.

    Only updates on explicit resolution strategies (Commit, Reframe).
    Beta scales with time since last update, capped at weekly max.

    Args:
        anchors: Current anchor state
        cur_vec: Current presence vector
        strategy: Resolution strategy ("Commit", "Reframe", "Nuance", "Defer")
        beliefs_touched: List of belief statements involved
        cfg_beta_week_cap: Maximum beta per week (default 0.01)

    Returns:
        Updated anchors (origin unchanged, live potentially updated)
    """
    now = time.time()
    days = max(0.0, min((now - anchors.last_update_ts) / 86400.0, 14.0))

    # Time-based beta
    β_base = 0.01 * (days / 7.0)

    # Strategy weight
    β_strategy = {"Commit": 1.0, "Reframe": 0.5}.get(strategy, 0.0)

    # Final beta with cap
    β = min(β_base * β_strategy, cfg_beta_week_cap)

    if β == 0.0:
        return anchors

    # Compute similarities before update
    sim_live_before = cos(cur_vec, anchors.live)
    sim_origin = cos(cur_vec, anchors.origin)

    # Update live anchor with EMA
    new_live = (1.0 - β) * anchors.live + β * cur_vec
    new_live = new_live.astype(np.float32)

    # Compute similarity after update
    sim_live_after = cos(cur_vec, new_live)

    # Log to identity ledger
    anchor_update_event(sim_live_before, sim_live_after, sim_origin, strategy, beliefs_touched)

    return Anchors(origin=anchors.origin, live=new_live, last_update_ts=now)
