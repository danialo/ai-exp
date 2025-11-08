# GoalStore Phase 1 — Design Specification

This document specifies Phase 1 of Astra’s GoalStore: a value/effort/risk prioritization
layer with persistence and APIs, integrated with the TaskScheduler and Decision Framework,
with safety and observability guardrails. Scope matches .claude/tasks/goalstore-codex-prompt-final.md.

## Summary
- Persist goals with full CRUD, query by state, soft delete, optimistic locking.
- Deterministic prioritizer: value↑, effort↓, risk↓, urgency↑, alignment↑, contradictions↓.
- REST APIs to create/read/update/adopt/abandon and list prioritized goals.
- Scheduler co-existence via goal→task adaptor, idempotent submission, trace propagation.
- Observability via Prometheus metrics and IdentityLedger events.
- Flags: `GOAL_SYSTEM`, `GOAL_SHADOW` (shadow-first activation with rollback).
- Tests: unit, property, integration, performance, safety and idempotency.

---

## Data Model & Migration

### Python Model (dataclasses)
```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime


class GoalCategory(str, Enum):
    INTROSPECTION = "introspection"
    EXPLORATION = "exploration"
    MAINTENANCE = "maintenance"
    USER_REQUESTED = "user_requested"


class GoalState(str, Enum):
    PROPOSED = "proposed"
    ADOPTED = "adopted"
    EXECUTING = "executing"
    SATISFIED = "satisfied"
    ABANDONED = "abandoned"


@dataclass
class GoalDefinition:
    id: str
    text: str
    category: GoalCategory
    value: float                # [0,1]
    effort: float               # [0,1] (1 = very hard)
    risk: float                 # [0,1] (1 = very risky)
    horizon_min_min: int        # earliest start in minutes from created_at
    horizon_max_min: Optional[int] = None  # deadline window from created_at
    aligns_with: List[str] = field(default_factory=list)       # belief IDs
    contradicts: List[str] = field(default_factory=list)       # belief IDs
    success_metrics: Dict[str, float] = field(default_factory=dict)
    state: GoalState = GoalState.PROPOSED
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, object] = field(default_factory=dict)

    # Concurrency control (optimistic locking)
    version: int = 0
    deleted_at: Optional[datetime] = None  # soft delete tombstone
```

### SQLite Schema (forward-only migration)
Single `goals` table with JSON for arrays and dicts, plus indexes for state, category, and
prioritization. Use integer `version` for optimistic locking.

```sql
-- scripts/migrate_001_goal_store.sql
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS goals (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  category TEXT NOT NULL,
  value REAL NOT NULL CHECK (value BETWEEN 0.0 AND 1.0),
  effort REAL NOT NULL CHECK (effort BETWEEN 0.0 AND 1.0),
  risk REAL NOT NULL CHECK (risk BETWEEN 0.0 AND 1.0),
  horizon_min_min INTEGER NOT NULL,
  horizon_max_min INTEGER,
  aligns_with TEXT NOT NULL DEFAULT '[]',        -- JSON array of belief_ids
  contradicts TEXT NOT NULL DEFAULT '[]',        -- JSON array of belief_ids
  success_metrics TEXT NOT NULL DEFAULT '{}',    -- JSON object
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}',           -- JSON object
  version INTEGER NOT NULL DEFAULT 0,
  deleted_at TEXT
);

CREATE INDEX IF NOT EXISTS ix_goals_state ON goals(state);
CREATE INDEX IF NOT EXISTS ix_goals_category ON goals(category);
CREATE INDEX IF NOT EXISTS ix_goals_updated ON goals(updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_goals_deadline ON goals(horizon_max_min);
```

Migration entrypoint (forward-only):
- Add `scripts/migrate_001_goal_store.sql` and call from `scripts/init_db.py` if table missing.
- No destructive changes; future migrations append `migrate_00N_*.sql` files.

---

## Prioritizer

### Scoring Function
Let weights come from Decision Framework registration `goal_selected`:
`wv, we, wr, wu, wa ∈ [0,1]` with typical defaults `0.5, 0.25, 0.15, 0.05, 0.05`.

Components:
- Effort factor: `effort_term = 1 - effort`
- Risk factor: `risk_term = 1 - risk`
- Urgency factor (deadline proximity): see horizon scoring below
- Alignment bonus: fraction of aligns_with that are active beliefs (0..1)
- Contradiction penalty: negative term if any contradicts are active

Final score (clamped [0,1]):
```
raw = wv*value + we*effort_term + wr*risk_term + wu*urgency + wa*alignment - penalty
score = max(0.0, min(1.0, raw))
```

Where `penalty = 1.0` if adoption would contradict an active belief (hard block in selection),
otherwise `penalty = 0.0`.

### Horizon/Urgency
```python
from datetime import datetime

def compute_urgency(created_at: datetime, horizon_max_min: int | None, now: datetime) -> float:
    if not horizon_max_min:
        return 0.0
    elapsed_min = (now - created_at).total_seconds() / 60
    remaining_min = horizon_max_min - elapsed_min
    if remaining_min > 0:
        hours_remaining = remaining_min / 60
        return 0.0 if hours_remaining > 24 else (1.0 - hours_remaining/24)
    return -1.0  # missed deadline → negative pressure
```

### Deterministic Implementation
```python
def score_goal(goal: GoalDefinition, now: datetime, weights: dict[str, float],
               active_beliefs: set[str]) -> float:
    wv = weights.get("value_weight", 0.5)
    we = weights.get("effort_weight", 0.25)
    wr = weights.get("risk_weight", 0.15)
    wu = weights.get("urgency_weight", 0.05)
    wa = weights.get("alignment_weight", 0.05)

    effort_term = 1.0 - goal.effort
    risk_term = 1.0 - goal.risk
    urgency = compute_urgency(goal.created_at, goal.horizon_max_min, now)

    aligned = [b for b in goal.aligns_with if b in active_beliefs]
    alignment = (len(aligned) / max(1, len(goal.aligns_with))) if goal.aligns_with else 0.0
    contradict_active = any(b for b in goal.contradicts if b in active_beliefs)
    penalty = 1.0 if contradict_active else 0.0

    raw = (wv*goal.value + we*effort_term + wr*risk_term + wu*urgency + wa*alignment) - penalty
    return max(0.0, min(1.0, raw))
```

### Ranking Invariants (property tests)
1) Value monotonicity: higher `value` → higher rank (all else equal)
2) Effort sensitivity: higher `effort` → lower rank
3) Risk aversion: higher `risk` → lower rank
4) Urgency boost: closer to deadline → higher rank; negative after deadline
5) Alignment bonus: more active aligns_with → higher rank
6) Contradiction blocking: any active contradicts → score pushed to 0 by penalty
7) Determinism: same inputs → same order; stable sort by `id` as tiebreaker

---

## REST API Design (FastAPI, Pydantic v2)

Base path: `/v1/goals`

Headers:
- `X-Trace-Id` (optional): trace propagation
- `X-Span-Id` (optional): span propagation
- `Idempotency-Key` (POST/command endpoints): ensure idempotent create/adopt/abandon
- `If-Match: <version>` (PATCH): optimistic locking (or body `version` field)

### Pydantic Schemas
```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict
from datetime import datetime


class GoalCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str
    category: GoalCategory
    value: float = Field(ge=0.0, le=1.0)
    effort: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)
    horizon_min_min: int = Field(ge=0)
    horizon_max_min: Optional[int] = Field(default=None, ge=1)
    aligns_with: List[str] = []
    contradicts: List[str] = []
    success_metrics: Dict[str, float] = {}
    metadata: Dict[str, object] = {}


class GoalRead(BaseModel):
    id: str
    text: str
    category: GoalCategory
    value: float
    effort: float
    risk: float
    horizon_min_min: int
    horizon_max_min: Optional[int]
    aligns_with: List[str]
    contradicts: List[str]
    success_metrics: Dict[str, float]
    state: GoalState
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, object]
    version: int


class GoalPatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: Optional[str] = None
    category: Optional[GoalCategory] = None
    value: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    effort: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    horizon_min_min: Optional[int] = Field(default=None, ge=0)
    horizon_max_min: Optional[int] = Field(default=None, ge=1)
    aligns_with: Optional[List[str]] = None
    contradicts: Optional[List[str]] = None
    success_metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, object]] = None
    state: Optional[GoalState] = None
    version: int  # required for optimistic locking


class GoalAdoptResult(BaseModel):
    id: str
    adopted: bool
    blocked_by_belief: bool = False
    belief_ids: List[str] = []
    reason: Optional[str] = None
    version: int
```

### Endpoints
1) POST `/v1/goals`
   - Body: `GoalCreate`
   - Headers: `Idempotency-Key` (required)
   - Response: `GoalRead`
   - Emits ledger `goal_created`; increments `goalstore_goals_total{category,state="proposed"}`.

2) GET `/v1/goals`
   - Query: `state` (optional), `category` (optional), `limit`, `offset`
   - Response: `List[GoalRead]`

3) GET `/v1/goals/{id}`
   - Response: `GoalRead`

4) PATCH `/v1/goals/{id}`
   - Body: `GoalPatch` (must include current `version`)
   - Headers: `If-Match` optional (alternative to `version` field)
   - Behavior: 409 if `version` mismatch (optimistic locking)
   - Response: `GoalRead`; emits `goal_updated`.

5) POST `/v1/goals/{id}/adopt`
   - Headers: `Idempotency-Key`
   - Behavior: Validate with `BeliefConsistencyChecker`; block if severity ≥ 0.6 or any
     `contradicts` is active. If `GOAL_SHADOW=on`, do not execute, only log/metrics.
   - Response: `GoalAdoptResult`; emits `goal_adopted` or `goal_blocked_by_belief`.

6) POST `/v1/goals/{id}/abandon`
   - Headers: `Idempotency-Key`
   - Behavior: Transition to `abandoned`, set `deleted_at` for soft delete.
   - Response: `GoalRead`; emits `goal_abandoned`.

7) GET `/v1/goals/prioritized`
   - Query: `state` (default `proposed`), `limit`
   - Behavior: Rank using `score_goal`; stable sort by score desc, then `id`.
   - Response: `List[GoalRead]`; histogram `goalstore_prioritize_latency_ms`.

### Examples
Create goal:
```http
POST /v1/goals
Idempotency-Key: 8b8a9c1e-...
Content-Type: application/json

{
  "text": "Conduct weekly self-reflection on identity coherence",
  "category": "introspection",
  "value": 0.8,
  "effort": 0.3,
  "risk": 0.1,
  "horizon_min_min": 0,
  "horizon_max_min": 10080,
  "aligns_with": ["core.ontological.consciousness"],
  "contradicts": [],
  "success_metrics": {"coherence_delta": 0.05}
}
```

Adopt goal:
```http
POST /v1/goals/goal_001/adopt
Idempotency-Key: 3b1c...
```

Prioritized listing:
```http
GET /v1/goals/prioritized?state=proposed&limit=25
```

---

## Scheduler & Decision Framework Integration

### Decision Framework Registration
Use existing registry and weights under `decision_id="goal_selected"`:
```python
decision_registry.register_decision(
    decision_id="goal_selected",
    subsystem="goal_store",
    parameters={
        "value_weight": Parameter(current=0.5, min=0.0, max=1.0, step=0.05),
        "effort_weight": Parameter(current=0.25, min=0.0, max=1.0, step=0.05),
        "risk_weight": Parameter(current=0.15, min=0.0, max=1.0, step=0.05),
        "urgency_weight": Parameter(current=0.05, min=0.0, max=0.3, step=0.02),
        "alignment_weight": Parameter(current=0.05, min=0.0, max=0.3, step=0.02)
    },
    success_metrics=["coherence", "goal_satisfaction", "task_completion_rate"]
)
```

### TaskScheduler Co-existence (Phase 1)
- Add `GoalStore.select_goal(now, flags) -> Optional[GoalDefinition]`.
- In `TaskScheduler.get_next_task()` (new), if `GOAL_SYSTEM=on`, call `select_goal` first.
  - If a goal is returned and `GOAL_SHADOW=off`, adapt it to a `TaskDefinition`:
    - `id = f"goal_{goal.id}"`; `type = TaskType.CUSTOM`; `schedule = MANUAL`.
    - `prompt` derived from goal text and success metrics.
    - Submit idempotently; propagate `trace_id`.
- If `GOAL_SHADOW=on`, only log selection; do not emit a task.
- Fallback to existing scheduled tasks if no goal is ready or flag off.

### Safety Hooks
- Adoption vetting via `BeliefConsistencyChecker.check_consistency(query=goal.text, ...)`.
- Re-check before emitting a task (selection safety gate).
- Block when severity ≥ 0.6 or contradicts an active belief; emit ledger event.

---

## Observability

### Prometheus Metrics (namespaced `goalstore_`)
- Counters
  - `goalstore_goals_total{category,state}`
  - `goalstore_goals_adopted_total`
  - `goalstore_goal_selected_total`
  - `goalstore_goal_blocked_by_belief_total{belief_id}`
- Gauges
  - `goalstore_goals_by_state{state}` (current counts)
- Histograms
  - `goalstore_prioritize_latency_ms`

Implementation: use `prometheus_client` (`Counter`, `Gauge`, `Histogram`). Expose under the
existing `/metrics` endpoint (create if absent) and include goalstore metrics in registry.

### IdentityLedger Events
Emit structured events (via `identity_ledger.append_event` or helper functions):
- `goal_created`
- `goal_updated`
- `goal_adopted`
- `goal_selected_for_execution` (when chosen for TaskScheduler)
- `goal_satisfied`
- `goal_abandoned`
- `goal_blocked_by_belief` (include `belief_ids` and severity)

Fields: `{ goal_id, category, state, trace_id, span_id, meta }`.

---

## Error Handling
- Validate `aligns_with`/`contradicts` IDs format; warn if unknown `aligns_with`.
- Reject if `horizon_max_min` corresponds to time < `now`.
- On selection, re-run safety check; if blocked, emit ledger event and counter.
- All command endpoints are idempotent (via `Idempotency-Key`).
- Optimistic locking via `version`; 409 Conflict on mismatch.

---

## Test Plan

### Unit Tests
- Prioritizer property tests (Hypothesis): invariants 1–7.
- Urgency edge cases: >24h, <24h, past deadline.
- Alignment/contradiction effects with various belief sets.

### API Tests
- CRUD happy paths and validation failures.
- Optimistic locking: update with stale `version` → 409.
- Idempotency: repeated POST with same `Idempotency-Key` → single record.

### Integration Tests
- Adoption vetting with stubbed `BeliefConsistencyChecker` severities.
- Decision Framework weight retrieval applied to ranking.
- Scheduler co-existence in shadow mode (no tasks emitted) and active mode.

### Performance Tests
- Rank 1k goals: p95 ≤ 30 ms (in-memory data); with DB fetch p95 ≤ 80 ms.
- Concurrency smoke: parallel prioritize calls remain deterministic.

### Safety
- Zero safety violations in CI pack: blocked contradictions must never adopt/execute.

---

## Activation & Flags

### Flags (in `config/settings.py`)
- `GOAL_SYSTEM: bool` (default: off)
- `GOAL_SHADOW: bool` (default: on)

### Activation Plan
Phase 1 (Shadow default)
- `GOAL_SYSTEM=off`, `GOAL_SHADOW=on`; goals created/ranked; no execution impact.

Phase 2 (Canary, 1 week)
- `GOAL_SYSTEM=on` for 10% selections; monitor `goal_selected_total`, success, blocks.

Phase 3 (Full, 2 weeks)
- `GOAL_SYSTEM=on` for 100%; keep `GOAL_SHADOW` as kill switch.

Rollback
- Set `GOAL_SYSTEM=off`, restart server; scheduler uses existing tasks only.

### README Snippet (to add)
```md
GoalStore (Phase 1) adds goal CRUD + prioritization with safety vetting and
shadow-first activation. Enable selection by setting `GOAL_SYSTEM=on`. Keep
`GOAL_SHADOW=on` to log rankings without executing.

Key endpoints: POST /v1/goals, GET /v1/goals/prioritized, POST /v1/goals/{id}/adopt
```

---

## OpenAPI Examples
- Create, adopt, and prioritized examples provided above; FastAPI will expose under `/docs`.
  Include example responses for blocked adoption with `belief_ids` and `severity` note.

---

## Implementation Notes (Phase 1 scope)
- New module `src/services/goal_store.py` with:
  - Storage adapter (SQLite): CRUD, query by state, soft delete, optimistic locking.
  - Ranker using Decision Framework weights and safety lookups.
  - Idempotency support (command endpoints) backed by unique key table or in-memory TTL.
- API router `src/api/goal_endpoints.py` mounted in `app.py` under `/v1/goals`.
- Optional: cache prioritized listing for N seconds keyed by (state, limit).

---

## Checklist (Requirements Fit)
- [x] Data model + SQLite migration (forward-only)
- [x] GoalStore CRUD, list, query by state, soft delete, optimistic locking
- [x] Deterministic prioritizer with specified weights and penalties
- [x] APIs: create/read/update/adopt/abandon/prioritized with examples
- [x] Scheduler integration notes with idempotent submission and trace propagation
- [x] Observability: Prometheus metrics plan + IdentityLedger events
- [x] Tests: unit, property, integration, performance, safety
- [x] Flags and activation/rollback strategy; README snippet provided

