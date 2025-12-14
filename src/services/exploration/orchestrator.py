"""Orchestrator for autonomous exploration loops."""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import (
    ExplorationJob,
    JobState,
    JobMetrics,
    OracleResult,
)
from .workspace import WorkspaceManager
from .oracle import Oracle

logger = logging.getLogger(__name__)


class ExplorationOrchestrator:
    """Orchestrates the PLAN→ACT→CHECK→REFLECT→PATCH exploration loop."""

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        persona_service,
        llm_service,
    ):
        """Initialize orchestrator.

        Args:
            workspace_manager: Workspace manager for file operations
            persona_service: Astra's persona service for planning
            llm_service: LLM service for generation
        """
        self.workspace_manager = workspace_manager
        self.persona_service = persona_service
        self.llm_service = llm_service
        self.running_jobs: Dict[str, asyncio.Task] = {}

    async def run_job(self, job: ExplorationJob) -> ExplorationJob:
        """Run an exploration job through the PLAN→ACT→CHECK→REFLECT→PATCH loop.

        Args:
            job: Job to execute

        Returns:
            Updated job with final state
        """
        logger.info(f"Starting exploration job {job.job_id}: {job.task}")

        job.state = JobState.RUNNING
        job.started_at = datetime.utcnow()

        # Create workspace
        workspace_dir = self.workspace_manager.create_workspace(
            job.job_id,
            job.workspace_policy
        )

        # Create oracle
        oracle = Oracle(job.oracle, workspace_dir)
        oracle.set_llm_service(self.llm_service)

        # Conversation context for the loop
        conversation_history = []

        # Token budget tracking
        tokens_used_this_iteration = 0
        budget_schedule = []

        try:
            for iteration in range(1, job.limits.max_iterations + 1):
                job.iteration = iteration
                logger.info(f"Job {job.job_id} - Iteration {iteration}/{job.limits.max_iterations}")

                # Calculate token budget for this iteration
                token_budget = self._calculate_token_budget(iteration, job.limits)
                budget_schedule.append(token_budget)

                # Check global token cap
                if job.tokens_spent >= job.limits.token_global_cap:
                    logger.warning(f"Job {job.job_id} hit global token cap")
                    job.state = JobState.FAILED
                    job.error_message = f"Global token budget exceeded: {job.tokens_spent}/{job.limits.token_global_cap}"
                    break

                iteration_start = time.time()
                metrics = JobMetrics()

                # PLAN: Generate next actions from Astra
                plan_start = time.time()
                plan_result = await self._plan_step(
                    job,
                    workspace_dir,
                    conversation_history,
                    token_budget,
                    iteration
                )
                metrics.plan_time_ms = int((time.time() - plan_start) * 1000)

                if not plan_result["success"]:
                    logger.error(f"Planning failed: {plan_result.get('error')}")
                    job.state = JobState.FAILED
                    job.error_message = plan_result.get('error', 'Planning failed')
                    break

                tokens_used_this_iteration = plan_result["tokens_used"]
                job.tokens_spent += tokens_used_this_iteration
                actions = plan_result["actions"]

                # ACT: Execute actions (write files, run scripts)
                exec_start = time.time()
                exec_result = await self._act_step(
                    job,
                    workspace_dir,
                    actions
                )
                metrics.exec_time_ms = int((time.time() - exec_start) * 1000)

                # Update conversation history with execution results
                conversation_history.append({
                    "role": "assistant",
                    "content": plan_result["response"]
                })
                conversation_history.append({
                    "role": "user",
                    "content": f"Execution results:\n{exec_result['summary']}"
                })

                # CHECK: Run oracle
                check_start = time.time()
                oracle_result = await oracle.check()
                job.oracle_checks += 1
                job.last_oracle_result = oracle_result
                metrics.check_time_ms = int((time.time() - check_start) * 1000)

                logger.info(f"Oracle check: {'PASS' if oracle_result.ok else 'FAIL'}")

                # PASS: If oracle passed, we're done!
                if oracle_result.ok:
                    logger.info(f"Job {job.job_id} succeeded at iteration {iteration}")
                    job.state = JobState.SUCCEEDED
                    job.artifacts = self.workspace_manager.list_artifacts(job.job_id)
                    break

                # REFLECT: Analyze failure
                reflect_start = time.time()
                reflection = await self._reflect_step(
                    job,
                    oracle_result,
                    exec_result,
                    iteration
                )
                metrics.reflect_time_ms = int((time.time() - reflect_start) * 1000)

                # Add reflection to conversation
                conversation_history.append({
                    "role": "user",
                    "content": reflection
                })

                # Update metrics
                job.metrics = metrics

                # Update error tracking
                if oracle_result.stderr:
                    for line in oracle_result.stderr.split('\n')[:10]:
                        error_key = line[:100]  # Truncate for histogram
                        job.error_histogram[error_key] = job.error_histogram.get(error_key, 0) + 1

                # PATCH: Next iteration will plan fixes based on reflection

            # If we exited loop without success
            if job.state == JobState.RUNNING:
                job.state = JobState.FAILED
                job.error_message = f"Max iterations ({job.limits.max_iterations}) reached without success"

        except asyncio.CancelledError:
            logger.info(f"Job {job.job_id} cancelled")
            job.state = JobState.CANCELED
            raise
        except Exception as e:
            logger.error(f"Job {job.job_id} error: {e}", exc_info=True)
            job.state = JobState.FAILED
            job.error_message = str(e)
        finally:
            job.completed_at = datetime.utcnow()

            # Cleanup workspace (keep artifacts)
            self.workspace_manager.cleanup_workspace(job.job_id, keep_artifacts=True)

        logger.info(f"Job {job.job_id} finished: {job.state}")
        return job

    def _calculate_token_budget(self, iteration: int, limits) -> int:
        """Calculate token budget for this iteration.

        Args:
            iteration: Current iteration number (1-indexed)
            limits: Job limits configuration

        Returns:
            Token budget for this iteration
        """
        # Start budget
        budget = limits.token_start

        # Apply growth for each iteration after the first
        if iteration > 1:
            growth_factor = limits.token_growth ** (iteration - 1)
            budget = int(limits.token_start * growth_factor)

        # Cap at per-attempt ceiling
        budget = min(budget, limits.token_attempt_cap)

        return budget

    async def _plan_step(
        self,
        job: ExplorationJob,
        workspace_dir: Path,
        conversation_history: List[Dict[str, str]],
        token_budget: int,
        iteration: int
    ) -> Dict[str, Any]:
        """PLAN: Generate next actions from Astra.

        Args:
            job: Current job
            workspace_dir: Workspace directory
            conversation_history: Conversation context
            token_budget: Token budget for this iteration
            iteration: Current iteration number

        Returns:
            Dict with success, response, tokens_used, actions
        """
        try:
            # Build planning prompt
            prompt = self._build_planning_prompt(
                job,
                workspace_dir,
                iteration,
                token_budget
            )

            # Generate response from Astra
            result = self.persona_service.generate_response(
                user_message=prompt,
                conversation_history=conversation_history,
                retrieve_memories=False  # Don't retrieve memories for task execution
            )

            # Parse response for tool calls (write_file, exec)
            response_text, _ = result if isinstance(result, tuple) else (result, None)

            # Extract actions from response
            # For now, return simple structure - will enhance with actual tool parsing
            actions = self._parse_actions_from_response(response_text)

            return {
                "success": True,
                "response": response_text,
                "tokens_used": len(response_text.split()) * 2,  # Rough estimate
                "actions": actions
            }

        except Exception as e:
            logger.error(f"Planning error: {e}")
            return {
                "success": False,
                "error": str(e),
                "tokens_used": 0,
                "actions": []
            }

    def _build_planning_prompt(
        self,
        job: ExplorationJob,
        workspace_dir: Path,
        iteration: int,
        token_budget: int
    ) -> str:
        """Build the planning prompt for Astra.

        Args:
            job: Current job
            workspace_dir: Workspace directory
            iteration: Current iteration
            token_budget: Token budget

        Returns:
            Planning prompt
        """
        # List current workspace files
        workspace_files = []
        for file_path in workspace_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                rel_path = file_path.relative_to(workspace_dir)
                workspace_files.append(str(rel_path))

        prompt = f"""Task: {job.task}

You are working in an isolated workspace to complete this task.
Workspace: {workspace_dir}
Current files: {', '.join(workspace_files) if workspace_files else 'none'}

Iteration: {iteration}/{job.limits.max_iterations}
Token budget: {token_budget}
Tokens spent so far: {job.tokens_spent}/{job.limits.token_global_cap}

Oracle checks: {job.oracle_checks}
Last oracle result: {'PASS' if job.last_oracle_result and job.last_oracle_result.ok else 'FAIL'}

Available tools:
- write_file(path, content): Write a file in the workspace
- exec(command): Execute a command in the workspace

Plan your next actions to complete the task. The oracle will verify your work."""

        return prompt

    def _parse_actions_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse actions from Astra's response.

        Args:
            response: Response text

        Returns:
            List of action dicts
        """
        # Placeholder - will implement proper tool call parsing
        # For now, return empty list (actions will be extracted from tool calls)
        return []

    async def _act_step(
        self,
        job: ExplorationJob,
        workspace_dir: Path,
        actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """ACT: Execute planned actions.

        Args:
            job: Current job
            workspace_dir: Workspace directory
            actions: List of actions to execute

        Returns:
            Execution results
        """
        results = []

        for action in actions:
            if action["type"] == "write_file":
                success = self.workspace_manager.safe_write_file(
                    action["path"],
                    action["content"],
                    workspace_dir,
                    job.workspace_policy
                )
                results.append({
                    "action": "write_file",
                    "path": action["path"],
                    "success": success
                })
            elif action["type"] == "exec":
                # Execute command - placeholder for now
                results.append({
                    "action": "exec",
                    "command": action["command"],
                    "success": True,
                    "output": "Command execution pending implementation"
                })

        summary = "\n".join([
            f"- {r['action']}: {'✓' if r['success'] else '✗'} {r.get('path', r.get('command', ''))}"
            for r in results
        ])

        return {
            "success": True,
            "results": results,
            "summary": summary or "No actions executed"
        }

    async def _reflect_step(
        self,
        job: ExplorationJob,
        oracle_result: OracleResult,
        exec_result: Dict[str, Any],
        iteration: int
    ) -> str:
        """REFLECT: Analyze failure and guide next iteration.

        Args:
            job: Current job
            oracle_result: Oracle check result
            exec_result: Execution results
            iteration: Current iteration

        Returns:
            Reflection prompt for next iteration
        """
        reflection = f"""Oracle check FAILED (iteration {iteration}):

Exit code: {oracle_result.exit_code}
Stderr: {oracle_result.stderr[:500] if oracle_result.stderr else 'none'}
Stdout: {oracle_result.stdout[:500] if oracle_result.stdout else 'none'}

Previous actions:
{exec_result['summary']}

Analyze what went wrong and plan how to fix it. You have {job.limits.max_iterations - iteration} iterations remaining."""

        return reflection

    async def cancel_job(self, job_id: str):
        """Cancel a running job.

        Args:
            job_id: Job identifier
        """
        if job_id in self.running_jobs:
            task = self.running_jobs[job_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.running_jobs[job_id]
            logger.info(f"Cancelled job {job_id}")


def create_orchestrator(
    workspace_manager: WorkspaceManager,
    persona_service,
    llm_service,
) -> ExplorationOrchestrator:
    """Factory function to create an ExplorationOrchestrator.

    Args:
        workspace_manager: Workspace manager
        persona_service: Persona service
        llm_service: LLM service

    Returns:
        Configured orchestrator
    """
    return ExplorationOrchestrator(
        workspace_manager=workspace_manager,
        persona_service=persona_service,
        llm_service=llm_service,
    )
