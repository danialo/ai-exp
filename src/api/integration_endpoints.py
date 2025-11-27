"""
API endpoints for Integration Layer observability (Phase 4).

Provides:
- Global Workspace state inspection
- Focus stack monitoring
- Budget status queries
- Execution mode control
- Tick history and action log
- Prometheus metrics
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from prometheus_client import (
    Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)

# Define Prometheus metrics
TICK_DURATION = Histogram(
    'astra_tick_duration_seconds',
    'Duration of executive loop ticks',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

FOCUS_STACK_SIZE = Gauge(
    'astra_focus_stack_size',
    'Number of items in focus stack'
)

COGNITIVE_LOAD = Gauge(
    'astra_cognitive_load',
    'Current cognitive load (0.0-1.0)'
)

TOKENS_AVAILABLE = Gauge(
    'astra_budget_tokens_available',
    'Available token budget'
)

TOKENS_USED = Gauge(
    'astra_budget_tokens_used',
    'Tokens used this minute'
)

BELIEFS_FORMED = Gauge(
    'astra_beliefs_formed_today',
    'Beliefs formed today'
)

ANCHOR_DRIFT = Gauge(
    'astra_anchor_drift',
    'Identity drift from origin anchor'
)

AROUSAL_LEVEL = Gauge(
    'astra_arousal_level',
    'Current arousal level (0.0-1.0)'
)

TICK_COUNT = Counter(
    'astra_tick_count_total',
    'Total number of executive loop ticks'
)

ACTIONS_DISPATCHED = Counter(
    'astra_actions_dispatched_total',
    'Total actions dispatched',
    ['action_type']
)

router = APIRouter(prefix="/api/integration", tags=["integration"])


class ModeChangeRequest(BaseModel):
    """Request to change execution mode."""
    mode: str  # "interactive", "autonomous", "maintenance"
    reason: str = "manual_override"


class ModeChangeResponse(BaseModel):
    """Response from mode change."""
    previous_mode: str
    new_mode: str
    changed_at: str
    reason: str


@router.get("/state")
async def get_integration_state(request: Request):
    """Get current AstraState snapshot.

    Returns the full Global Workspace state including:
    - Self-model summary
    - Attention state
    - Active goals
    - Signal counts
    - Modulation state
    - Budget overview
    """
    integration_layer = getattr(request.app.state, 'integration_layer', None)

    if not integration_layer:
        raise HTTPException(status_code=503, detail="Integration Layer not initialized")

    state = integration_layer.state
    snapshot = state.to_snapshot()

    return {
        "tick_id": snapshot.tick_id,
        "timestamp": snapshot.timestamp,
        "mode": snapshot.mode,
        "session_id": snapshot.session_id,

        "identity": {
            "self_model_id": snapshot.self_model_id,
            "core_belief_count": snapshot.core_belief_count,
            "peripheral_belief_count": snapshot.peripheral_belief_count,
            "anchor_drift": snapshot.anchor_drift,
        },

        "attention": {
            "focus_stack": snapshot.focus_stack,
            "attention_capacity": snapshot.attention_capacity,
        },

        "goals": {
            "active_goal_ids": snapshot.active_goal_ids,
            "pending_goal_count": snapshot.pending_goal_count,
            "active_task_ids": snapshot.active_task_ids,
        },

        "signals": {
            "percept_buffer_size": snapshot.percept_buffer_size,
            "dissonance_alert_count": snapshot.dissonance_alert_count,
            "integration_event_count": snapshot.integration_event_count,
        },

        "modulation": {
            "emotional_state": snapshot.emotional_state,
            "arousal_level": snapshot.arousal_level,
            "cognitive_load": snapshot.cognitive_load,
        },

        "budgets": {
            "tokens_available": snapshot.tokens_available,
            "tokens_used_this_minute": snapshot.tokens_used_this_minute,
            "beliefs_formed_today": snapshot.beliefs_formed_today,
        },

        "last_introspection": snapshot.last_introspection,
    }


@router.get("/focus")
async def get_focus_stack(request: Request):
    """Get current focus stack with salience scores.

    Returns ordered list of attention items, most salient first.
    """
    integration_layer = getattr(request.app.state, 'integration_layer', None)

    if not integration_layer:
        raise HTTPException(status_code=503, detail="Integration Layer not initialized")

    state = integration_layer.state

    focus_items = []
    for item in state.focus_stack:
        focus_items.append({
            "item_type": item.item_type.value,
            "item_id": item.item_id,
            "content": item.content,
            "salience": item.salience,
            "entered_focus": item.entered_focus.isoformat(),
            "last_accessed": item.last_accessed.isoformat(),
            "access_count": item.access_count,
            "decay_rate": item.decay_rate,
            "min_salience_threshold": item.min_salience_threshold,
        })

    return {
        "focus_stack": focus_items,
        "stack_size": len(focus_items),
        "attention_capacity": state.attention_capacity,
        "cognitive_load": state.cognitive_load,
    }


@router.get("/budgets")
async def get_budget_status(request: Request):
    """Get current budget status for all resource types.

    Returns token budgets, belief operation limits, and resource usage.
    """
    integration_layer = getattr(request.app.state, 'integration_layer', None)

    if not integration_layer:
        raise HTTPException(status_code=503, detail="Integration Layer not initialized")

    budget = integration_layer.state.budget_status

    return {
        "tokens": {
            "per_minute_limit": budget.tokens_per_minute_limit,
            "used_this_minute": budget.tokens_used_this_minute,
            "available": budget.tokens_available,
            "utilization_pct": round(budget.tokens_used_this_minute / budget.tokens_per_minute_limit * 100, 1) if budget.tokens_per_minute_limit > 0 else 0,
        },

        "beliefs": {
            "form_limit": budget.beliefs_form_limit,
            "formed_today": budget.beliefs_formed_today,
            "form_remaining": budget.beliefs_form_limit - budget.beliefs_formed_today,

            "promote_limit": budget.beliefs_promote_limit,
            "promoted_today": budget.beliefs_promoted_today,
            "promote_remaining": budget.beliefs_promote_limit - budget.beliefs_promoted_today,

            "deprecate_limit": budget.beliefs_deprecate_limit,
            "deprecated_today": budget.beliefs_deprecated_today,
            "deprecate_remaining": budget.beliefs_deprecate_limit - budget.beliefs_deprecated_today,
        },

        "url_fetch": {
            "limit": budget.url_fetch_limit,
            "used_this_session": budget.url_fetches_this_session,
            "remaining": budget.url_fetch_limit - budget.url_fetches_this_session,
        },

        "introspection": {
            "min_interval_seconds": budget.min_introspection_interval_seconds,
            "last_introspection": budget.last_introspection.isoformat() if budget.last_introspection else None,
            "cooldown_active": _is_introspection_on_cooldown(budget),
        },

        "resources": {
            "cpu_usage": budget.cpu_usage,
            "memory_usage": budget.memory_usage,
            "gpu_usage": budget.gpu_usage,
        },
    }


def _is_introspection_on_cooldown(budget) -> bool:
    """Check if introspection is on cooldown."""
    if not budget.last_introspection:
        return False
    elapsed = (datetime.now() - budget.last_introspection).total_seconds()
    return elapsed < budget.min_introspection_interval_seconds


@router.post("/mode", response_model=ModeChangeResponse)
async def change_mode(request: Request, mode_request: ModeChangeRequest):
    """Change the Integration Layer execution mode.

    Valid modes:
    - interactive: User interaction active, prioritize responsiveness
    - autonomous: No user, self-directed goal pursuit
    - maintenance: Background consolidation and cleanup
    """
    integration_layer = getattr(request.app.state, 'integration_layer', None)

    if not integration_layer:
        raise HTTPException(status_code=503, detail="Integration Layer not initialized")

    # Import here to avoid circular imports
    from src.integration.state import ExecutionMode

    valid_modes = {m.value: m for m in ExecutionMode}

    if mode_request.mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode_request.mode}. Valid modes: {list(valid_modes.keys())}"
        )

    previous_mode = integration_layer.state.mode.value
    new_mode_enum = valid_modes[mode_request.mode]

    # Change the mode (both IL.mode and IL.state.mode need to be updated)
    integration_layer.mode = new_mode_enum
    integration_layer.state.mode = new_mode_enum

    return ModeChangeResponse(
        previous_mode=previous_mode,
        new_mode=mode_request.mode,
        changed_at=datetime.now().isoformat(),
        reason=mode_request.reason,
    )


@router.get("/history")
async def get_tick_history(request: Request, limit: int = 20):
    """Get recent tick history and dispatched actions.

    Returns the last N ticks with timing, actions taken, and outcomes.
    """
    integration_layer = getattr(request.app.state, 'integration_layer', None)

    if not integration_layer:
        raise HTTPException(status_code=503, detail="Integration Layer not initialized")

    # Get tick history if available (convert deque to list for slicing)
    tick_history = list(getattr(integration_layer, 'tick_history', []))
    action_log = list(getattr(integration_layer, 'action_log', []))

    # Get recent ticks (most recent first)
    recent_ticks = tick_history[-limit:][::-1] if tick_history else []
    recent_actions = action_log[-limit:][::-1] if action_log else []

    return {
        "current_tick": integration_layer.state.tick_id,
        "mode": integration_layer.state.mode.value,
        "tick_history": recent_ticks,
        "recent_actions": recent_actions,
        "history_size": len(tick_history),
        "action_log_size": len(action_log),
    }


def update_metrics(integration_layer) -> None:
    """Update Prometheus metrics from Integration Layer state.

    Call this periodically or after each tick to keep metrics current.
    """
    if not integration_layer:
        return

    state = integration_layer.state

    # Update gauges
    FOCUS_STACK_SIZE.set(len(state.focus_stack))
    COGNITIVE_LOAD.set(state.cognitive_load)
    AROUSAL_LEVEL.set(state.arousal_level)

    # Budget metrics
    TOKENS_AVAILABLE.set(state.budget_status.tokens_available)
    TOKENS_USED.set(state.budget_status.tokens_used_this_minute)
    BELIEFS_FORMED.set(state.budget_status.beliefs_formed_today)

    # Identity metrics
    if state.self_model:
        ANCHOR_DRIFT.set(state.self_model.anchor_drift)

    # Tick count
    TICK_COUNT._value.set(state.tick_id)


@router.get("/metrics")
async def get_metrics(request: Request):
    """Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    """
    integration_layer = getattr(request.app.state, 'integration_layer', None)

    # Update metrics before serving
    if integration_layer:
        update_metrics(integration_layer)

    return PlainTextResponse(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )
