"""FastAPI routes for autonomous exploration jobs."""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException

from src.services.exploration import (
    ExplorationJobCreate,
    ExplorationJobStatus,
    JobState,
    JobManager,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/persona/explore", tags=["exploration"])

# Global job manager (will be set by app.py)
job_manager: Optional[JobManager] = None


def set_job_manager(manager: JobManager):
    """Set the global job manager instance.

    Args:
        manager: Job manager to use for this router
    """
    global job_manager
    job_manager = manager


@router.post("", status_code=201)
async def create_exploration_job(request: ExplorationJobCreate) -> dict:
    """Create a new exploration job.

    Args:
        request: Job creation parameters

    Returns:
        Job ID and initial status
    """
    if not job_manager:
        raise HTTPException(
            status_code=503,
            detail="Exploration system not initialized"
        )

    try:
        # Create job
        job = job_manager.create_job(request)

        # Start job immediately
        started = await job_manager.start_job(job.job_id)

        if not started:
            raise HTTPException(
                status_code=500,
                detail="Failed to start job"
            )

        return {
            "job_id": job.job_id,
            "state": job.state,
            "message": f"Exploration job created and started: {job.job_id}"
        }

    except Exception as e:
        logger.error(f"Failed to create exploration job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create job: {str(e)}"
        )


@router.get("/{job_id}")
async def get_exploration_job(job_id: str) -> ExplorationJobStatus:
    """Get status of an exploration job.

    Args:
        job_id: Job identifier

    Returns:
        Job status with details
    """
    if not job_manager:
        raise HTTPException(
            status_code=503,
            detail="Exploration system not initialized"
        )

    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )

    # Build status response
    return ExplorationJobStatus(
        job_id=job.job_id,
        state=job.state,
        iteration=job.iteration,
        tokens_spent=job.tokens_spent,
        oracle={
            "last_result": job.last_oracle_result.dict() if job.last_oracle_result else None,
            "checks": job.oracle_checks,
        },
        logs_tail=job.logs_tail,
        artifacts=job.artifacts,
        metrics=job.metrics,
        acceptance_pass_at_iteration=job.iteration if job.state == JobState.SUCCEEDED else None,
        budget_schedule_used=[],  # TODO: Track budget schedule
    )


@router.delete("/{job_id}")
async def cancel_exploration_job(job_id: str) -> dict:
    """Cancel a running exploration job.

    Args:
        job_id: Job identifier

    Returns:
        Cancellation status
    """
    if not job_manager:
        raise HTTPException(
            status_code=503,
            detail="Exploration system not initialized"
        )

    cancelled = await job_manager.cancel_job(job_id)

    if not cancelled:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Job cannot be cancelled (state: {job.state})"
            )

    return {
        "job_id": job_id,
        "message": "Job cancelled successfully"
    }


@router.get("")
async def list_exploration_jobs(
    state: Optional[JobState] = None,
    limit: int = 100
) -> dict:
    """List exploration jobs.

    Args:
        state: Filter by job state (optional)
        limit: Maximum number of jobs to return

    Returns:
        List of job summaries
    """
    if not job_manager:
        raise HTTPException(
            status_code=503,
            detail="Exploration system not initialized"
        )

    jobs = job_manager.list_jobs(state=state, limit=limit)

    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "task": j.task[:100],  # Truncate for list view
                "state": j.state,
                "iteration": j.iteration,
                "tokens_spent": j.tokens_spent,
                "created_at": j.created_at.isoformat(),
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in jobs
        ],
        "total": len(jobs)
    }
