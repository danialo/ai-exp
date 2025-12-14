"""Job queue and state management for exploration jobs."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional, List

from .models import ExplorationJob, ExplorationJobCreate, JobState
from .orchestrator import ExplorationOrchestrator

logger = logging.getLogger(__name__)


class JobManager:
    """Manages exploration job queue and state."""

    def __init__(self, orchestrator: ExplorationOrchestrator):
        """Initialize job manager.

        Args:
            orchestrator: Orchestrator for running jobs
        """
        self.orchestrator = orchestrator
        self.jobs: Dict[str, ExplorationJob] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}

    def create_job(self, request: ExplorationJobCreate) -> ExplorationJob:
        """Create a new exploration job.

        Args:
            request: Job creation request

        Returns:
            Created job with assigned job_id
        """
        job_id = str(uuid.uuid4())

        job = ExplorationJob(
            job_id=job_id,
            task=request.task,
            oracle=request.oracle,
            limits=request.limits,
            workspace_policy=request.workspace_policy,
            capabilities=request.capabilities,
            state=JobState.QUEUED,
            created_at=datetime.utcnow()
        )

        self.jobs[job_id] = job
        logger.info(f"Created job {job_id}: {request.task[:50]}...")

        return job

    async def start_job(self, job_id: str) -> bool:
        """Start running a queued job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was started, False if not found or already running
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return False

        if job.state != JobState.QUEUED:
            logger.warning(f"Job {job_id} not in QUEUED state: {job.state}")
            return False

        if job_id in self.running_tasks:
            logger.warning(f"Job {job_id} already has running task")
            return False

        # Create async task for the job
        task = asyncio.create_task(self._run_job_wrapper(job_id))
        self.running_tasks[job_id] = task

        logger.info(f"Started job {job_id}")
        return True

    async def _run_job_wrapper(self, job_id: str):
        """Wrapper to run job and update state.

        Args:
            job_id: Job identifier
        """
        try:
            job = self.jobs[job_id]
            updated_job = await self.orchestrator.run_job(job)
            self.jobs[job_id] = updated_job
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            job = self.jobs[job_id]
            job.state = JobState.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
        finally:
            if job_id in self.running_tasks:
                del self.running_tasks[job_id]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled, False if not found or not running
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return False

        if job_id not in self.running_tasks:
            logger.warning(f"Job {job_id} is not running")
            return False

        # Cancel the task
        task = self.running_tasks[job_id]
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Update job state
        job.state = JobState.CANCELED
        job.completed_at = datetime.utcnow()

        # Remove from running tasks
        del self.running_tasks[job_id]

        logger.info(f"Cancelled job {job_id}")
        return True

    def get_job(self, job_id: str) -> Optional[ExplorationJob]:
        """Get a job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job if found, None otherwise
        """
        return self.jobs.get(job_id)

    def list_jobs(
        self,
        state: Optional[JobState] = None,
        limit: int = 100
    ) -> List[ExplorationJob]:
        """List jobs, optionally filtered by state.

        Args:
            state: Filter by job state (optional)
            limit: Maximum number of jobs to return

        Returns:
            List of jobs
        """
        jobs = list(self.jobs.values())

        if state:
            jobs = [j for j in jobs if j.state == state]

        # Sort by creation time, most recent first
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was deleted, False if not found
        """
        if job_id not in self.jobs:
            return False

        # Can't delete running jobs
        if job_id in self.running_tasks:
            logger.warning(f"Cannot delete running job {job_id}")
            return False

        del self.jobs[job_id]
        logger.info(f"Deleted job {job_id}")
        return True

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs.

        Args:
            max_age_hours: Maximum age in hours for completed jobs
        """
        now = datetime.utcnow()
        to_delete = []

        for job_id, job in self.jobs.items():
            if job.state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED):
                if job.completed_at:
                    age_hours = (now - job.completed_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        to_delete.append(job_id)

        for job_id in to_delete:
            self.delete_job(job_id)

        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old jobs")


def create_job_manager(orchestrator: ExplorationOrchestrator) -> JobManager:
    """Factory function to create a JobManager.

    Args:
        orchestrator: Orchestrator instance

    Returns:
        Configured job manager
    """
    return JobManager(orchestrator=orchestrator)
