"""GoalStore API endpoints.

Endpoints (base: /v1/goals):
- POST   /v1/goals
- GET    /v1/goals
- GET    /v1/goals/{id}
- PATCH  /v1/goals/{id}
- POST   /v1/goals/{id}/adopt
- POST   /v1/goals/{id}/abandon
- GET    /v1/goals/prioritized
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict

from src.services.goal_store import (
    GoalCategory,
    GoalDefinition,
    GoalState,
)


router = APIRouter(prefix="/v1/goals", tags=["goals"])


class GoalCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    metadata: Dict[str, Any] = {}


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
    metadata: Dict[str, Any]
    version: int


class GoalPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    metadata: Optional[Dict[str, Any]] = None
    state: Optional[GoalState] = None
    version: int


class GoalAdoptResult(BaseModel):
    id: str
    adopted: bool
    blocked_by_belief: bool = False
    belief_ids: List[str] = []
    reason: Optional[str] = None
    version: int


def _to_read(g: GoalDefinition) -> GoalRead:
    return GoalRead(
        id=g.id,
        text=g.text,
        category=g.category,
        value=g.value,
        effort=g.effort,
        risk=g.risk,
        horizon_min_min=g.horizon_min_min,
        horizon_max_min=g.horizon_max_min,
        aligns_with=g.aligns_with,
        contradicts=g.contradicts,
        success_metrics=g.success_metrics,
        state=g.state,
        created_at=g.created_at,
        updated_at=g.updated_at,
        metadata=g.metadata,
        version=g.version,
    )


@router.post("")
async def create_goal(
    request: Request,
    payload: GoalCreate,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    trace_id: Optional[str] = Header(default=None, alias="X-Trace-Id"),
    span_id: Optional[str] = Header(default=None, alias="X-Span-Id"),
):
    """Create a new goal (idempotent if Idempotency-Key provided)."""
    goal_store = request.app.state.goal_store
    gid = f"goal_{uuid.uuid4().hex[:8]}"
    g = GoalDefinition(
        id=gid,
        text=payload.text,
        category=payload.category,
        value=payload.value,
        effort=payload.effort,
        risk=payload.risk,
        horizon_min_min=payload.horizon_min_min,
        horizon_max_min=payload.horizon_max_min,
        aligns_with=payload.aligns_with or [],
        contradicts=payload.contradicts or [],
        success_metrics=payload.success_metrics or {},
        metadata=payload.metadata or {},
    )

    # Basic validation: deadline cannot be in the past
    if g.horizon_max_min is not None and g.horizon_max_min < 0:
        raise HTTPException(status_code=400, detail="horizon_max_min must be >= 0")

    created = goal_store.create_goal(g, idempotency_key=idempotency_key)
    return _to_read(created)


@router.get("")
async def list_goals(
    request: Request,
    state: Optional[GoalState] = None,
    category: Optional[GoalCategory] = None,
    limit: int = 50,
    offset: int = 0,
):
    goal_store = request.app.state.goal_store
    items = goal_store.list_goals(state=state, category=category, limit=limit, offset=offset)
    return [_to_read(g) for g in items]


@router.get("/{goal_id}")
async def get_goal(request: Request, goal_id: str):
    goal_store = request.app.state.goal_store
    g = goal_store.get_goal(goal_id)
    if not g:
        raise HTTPException(status_code=404, detail="Goal not found")
    return _to_read(g)


@router.patch("/{goal_id}")
async def patch_goal(request: Request, goal_id: str, patch: GoalPatch):
    goal_store = request.app.state.goal_store
    updates = patch.model_dump(exclude_none=True)
    version = updates.pop("version")
    g = goal_store.update_goal(goal_id, updates, expected_version=version)
    if not g:
        raise HTTPException(status_code=409, detail="Version conflict or not found")
    return _to_read(g)


@router.post("/{goal_id}/adopt")
async def adopt_goal(
    request: Request,
    goal_id: str,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    goal_store = request.app.state.goal_store
    belief_checker = getattr(request.app.state, "belief_consistency_checker", None)
    belief_store = getattr(request.app.state, "belief_store", None)

    active_belief_ids: List[str] = []
    aligned_beliefs = []
    if belief_store:
        # Consider ASSERTED beliefs as active
        current = belief_store.get_current()
        for bid, bver in current.items():
            if getattr(bver, "state", "asserted") != "deprecated":
                active_belief_ids.append(bid)

    adopted, g, details = goal_store.adopt_goal(
        goal_id=goal_id,
        idempotency_key=idempotency_key,
        belief_checker=belief_checker,
        aligned_beliefs=aligned_beliefs,
        memories=[],
        active_belief_ids=active_belief_ids,
    )

    if not g:
        raise HTTPException(status_code=404, detail="Goal not found")

    return GoalAdoptResult(
        id=g.id,
        adopted=adopted,
        blocked_by_belief=bool(details.get("blocked_by_belief")),
        belief_ids=details.get("belief_ids", []),
        reason=details.get("reason"),
        version=g.version,
    )


@router.post("/{goal_id}/abandon")
async def abandon_goal(
    request: Request,
    goal_id: str,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    goal_store = request.app.state.goal_store
    g = goal_store.abandon_goal(goal_id, idempotency_key=idempotency_key)
    if not g:
        raise HTTPException(status_code=404, detail="Goal not found")
    return _to_read(g)


@router.get("/prioritized")
async def list_prioritized(
    request: Request,
    state: GoalState = GoalState.PROPOSED,
    limit: int = 50,
):
    goal_store = request.app.state.goal_store
    # Pull weights from decision framework if present
    weights: Dict[str, float] = {}
    try:
        registry = getattr(request.app.state, "decision_registry", None)
        if registry:
            params = registry.get_all_parameters("goal_selected") or {}
            for k, v in params.items():
                weights[k] = float(v)
    except Exception:
        pass

    # Active beliefs for alignment/penalty
    belief_store = getattr(request.app.state, "belief_store", None)
    active_beliefs: List[str] = []
    if belief_store:
        current = belief_store.get_current()
        for bid, bver in current.items():
            if getattr(bver, "state", "asserted") != "deprecated":
                active_beliefs.append(bid)

    scored = goal_store.prioritized(state=state, limit=limit, weights=weights, active_beliefs=active_beliefs)
    return [
        {
            "goal": _to_read(g).model_dump(),
            "score": score,
        }
        for g, score in scored
    ]

