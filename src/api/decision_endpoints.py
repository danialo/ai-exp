"""
API endpoints for the Adaptive Decision Framework.

Provides:
- Decision registry inspection
- Decision history queries
- Parameter management
- Success signal monitoring
- Abort condition status
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.decision_framework import get_decision_registry
from src.services.abort_condition_monitor import AbortConditionMonitor
from src.services.success_signal_evaluator import SuccessSignalEvaluator

router = APIRouter(prefix="/api/persona/decisions", tags=["decisions"])


class ParameterUpdateRequest(BaseModel):
    """Request to update a parameter value."""
    decision_id: str
    param_name: str
    new_value: float
    reason: str = "manual_adjustment"


class AdaptationRequest(BaseModel):
    """Request to trigger parameter adaptation."""
    decision_id: str
    force: bool = False


@router.get("/registry")
async def get_registry():
    """Get all registered decision points."""
    registry = get_decision_registry()
    stats = registry.get_registry_stats()

    return {
        "stats": stats,
        "message": f"Tracking {stats['total_decision_types']} decision types, "
                   f"{stats['total_decisions_made']} decisions made, "
                   f"{stats['evaluated_decisions']} evaluated"
    }


@router.get("/history")
async def get_decision_history(
    decision_id: Optional[str] = None,
    limit: int = 20,
    evaluated_only: bool = False
):
    """
    Get decision history.

    Args:
        decision_id: Filter by specific decision type (optional)
        limit: Maximum number of records to return
        evaluated_only: Only return evaluated decisions
    """
    registry = get_decision_registry()

    if decision_id:
        decisions = registry.get_recent_decisions(
            decision_id=decision_id,
            limit=limit,
            evaluated_only=evaluated_only
        )
        return {
            "decision_id": decision_id,
            "count": len(decisions),
            "decisions": decisions
        }
    else:
        # Get stats by type
        stats = registry.get_registry_stats()
        return {
            "total_decisions": stats["total_decisions_made"],
            "evaluated_decisions": stats["evaluated_decisions"],
            "decisions_by_type": stats["decisions_by_type"]
        }


@router.get("/parameters")
async def get_parameters(decision_id: str):
    """Get current parameter values for a decision type."""
    registry = get_decision_registry()
    params = registry.get_all_parameters(decision_id)

    if params is None:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

    return {
        "decision_id": decision_id,
        "parameters": params
    }


@router.post("/parameters")
async def update_parameter(request: ParameterUpdateRequest):
    """
    Update a parameter value (admin only).

    Args:
        request: Parameter update request
    """
    registry = get_decision_registry()

    success = registry.update_parameter(
        decision_id=request.decision_id,
        param_name=request.param_name,
        new_value=request.new_value,
        reason=request.reason
    )

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to update {request.param_name} for {request.decision_id}"
        )

    return {
        "success": True,
        "decision_id": request.decision_id,
        "param_name": request.param_name,
        "new_value": request.new_value
    }


@router.get("/success_signals")
async def get_success_signals(
    evaluator: SuccessSignalEvaluator = None
):
    """
    Get success signal baselines and targets.

    Note: Requires success_evaluator to be passed from app context
    """
    if not evaluator:
        # Return defaults
        from src.services.success_signal_evaluator import SuccessSignalBaselines, SuccessSignalTargets
        baselines = SuccessSignalBaselines()
        targets = SuccessSignalTargets()

        return {
            "baselines": {
                "coherence": baselines.coherence,
                "dissonance": baselines.dissonance,
                "satisfaction": baselines.satisfaction
            },
            "targets": {
                "coherence": targets.coherence,
                "dissonance": targets.dissonance,
                "satisfaction": targets.satisfaction
            },
            "current": None,
            "note": "Success evaluator not initialized"
        }

    return evaluator.get_telemetry()


@router.get("/abort_status")
async def get_abort_status(
    monitor: AbortConditionMonitor = None
):
    """
    Get abort condition monitoring status.

    Note: Requires abort_monitor to be passed from app context
    """
    if not monitor:
        return {
            "aborted": False,
            "message": "Abort monitor not initialized"
        }

    return monitor.get_telemetry()


@router.post("/abort_status/reset")
async def reset_abort_status(
    monitor: AbortConditionMonitor = None
):
    """
    Reset abort condition (admin override).

    Note: Requires abort_monitor to be passed from app context
    """
    if not monitor:
        raise HTTPException(status_code=503, detail="Abort monitor not initialized")

    monitor.reset()

    return {
        "success": True,
        "message": "Abort condition reset"
    }


@router.post("/adapt")
async def trigger_adaptation(request: AdaptationRequest):
    """
    Trigger parameter adaptation for a decision type.

    Note: This is a placeholder - actual adaptation logic would:
    1. Get unevaluated decisions
    2. Evaluate their outcomes
    3. Compute parameter adjustments
    4. Update registry

    Args:
        request: Adaptation request with decision_id and force flag
    """
    registry = get_decision_registry()

    # Get unevaluated decisions
    unevaluated = registry.get_unevaluated_decisions(
        decision_id=request.decision_id,
        older_than_hours=24
    )

    return {
        "decision_id": request.decision_id,
        "unevaluated_count": len(unevaluated),
        "message": f"Found {len(unevaluated)} decisions ready for evaluation",
        "note": "Full adaptation pipeline not yet implemented"
    }
