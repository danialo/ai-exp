"""HTN task executor with budget enforcement and context injection."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import traceback
import logging

from src.services.task_queue import TaskStore, Task, new_task
from src.services.research_session import ResearchSessionStore
from src.services.research_htn_methods import METHODS
from src.utils.logging_config import get_multi_logger

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context passed to HTN methods."""
    llm_service: Any
    web_search_service: Any
    url_fetcher_service: Any
    session_store: ResearchSessionStore
    task_store: TaskStore


class HTNTaskExecutor:
    """Executes HTN tasks with session budget enforcement."""

    def __init__(self, task_store: TaskStore, session_store: ResearchSessionStore, ctx: ExecutionContext):
        self.tasks = task_store
        self.sessions = session_store
        self.ctx = ctx
        # Inject backrefs so SynthesizeFindings can access stores
        self.ctx.session_store = session_store
        self.ctx.task_store = task_store

    def _enforce_budgets_and_enqueue(self, parent: Task, proposals: List[Dict[str, Any]]) -> int:
        """
        Enforce session budgets and enqueue child tasks.

        Args:
            parent: Parent task that generated these proposals
            proposals: List of dicts with keys: htn_task_type, args, depth (optional), dedup_key (optional)

        Returns:
            Number of tasks actually enqueued
        """
        sess = self.sessions.get_session(parent.session_id)
        if not sess:
            logger.warning(f"Session {parent.session_id} not found")
            return 0

        # Check budget
        remaining = max(0, sess.max_tasks - int(sess.tasks_created or 0))
        if remaining <= 0:
            logger.info(f"Session {parent.session_id} budget exhausted")
            return 0

        # Enforce max_children_per_task
        per_parent = min(sess.max_children_per_task, remaining)
        accepted = []
        seen = set()  # Simple dedup within this batch

        for p in proposals:
            if len(accepted) >= per_parent:
                break

            # Depth check
            depth = int(p.get("depth", parent.depth + 1))
            if depth > sess.max_depth:
                logger.debug(f"Skipping task (depth {depth} > max {sess.max_depth})")
                continue

            # Optional deduplication
            dedup_key = p.get("dedup_key")
            if dedup_key:
                if dedup_key in seen:
                    logger.debug(f"Skipping duplicate task: {dedup_key}")
                    continue
                seen.add(dedup_key)

            t = new_task(
                session_id=parent.session_id,
                htn_task_type=p["htn_task_type"],
                args=p.get("args", {}),
                depth=depth,
                parent_id=parent.id,
            )
            accepted.append(t)

        if not accepted:
            return 0

        # Atomic: enqueue tasks and update session counter
        self.tasks.create_many(accepted)
        self.sessions.increment_tasks_created(parent.session_id, len(accepted))

        logger.info(f"Enqueued {len(accepted)} child tasks for parent {parent.id}")
        return len(accepted)

    def run_until_empty(self, session_id: Optional[str] = None) -> None:
        """
        Execute tasks until queue is empty.

        Args:
            session_id: If set, only execute tasks for this session
        """
        while True:
            task = self.tasks.pop_next_queued()
            if not task:
                logger.info("No more queued tasks")
                break

            # Filter by session if requested
            if session_id and task.session_id != session_id:
                logger.debug(f"Skipping task {task.id} (wrong session)")
                self.tasks.mark_error(task.id, "Skipped by filtered executor")
                continue

            try:
                logger.info(f"Executing task {task.id}: {task.htn_task_type}")

                # Look up HTN method
                handler = METHODS.get(task.htn_task_type)
                if not handler:
                    logger.warning(f"No HTN method for {task.htn_task_type}, marking done")
                    self.tasks.mark_done(task.id)
                    continue

                # Execute HTN method (returns list of child task proposals)
                proposals = handler(task=task, ctx=self.ctx)

                # Enforce budgets and enqueue children
                num_children = self._enforce_budgets_and_enqueue(task, proposals or [])

                # Mark parent task complete
                self.tasks.mark_done(task.id)

                # Log task completion
                get_multi_logger().log_research_event(
                    event_type="task_done",
                    session_id=task.session_id,
                    data={
                        "task_id": task.id,
                        "type": task.htn_task_type,
                        "depth": task.depth,
                        "children": num_children
                    }
                )

                # Check if session should be marked complete
                self._maybe_complete_session(task.session_id)

            except Exception as e:
                logger.error(f"Task {task.id} failed: {e}\n{traceback.format_exc()}")
                self.tasks.mark_error(task.id, str(e))

    def _maybe_complete_session(self, session_id: str) -> None:
        """Mark session complete if budget exhausted or no tasks remain."""
        sess = self.sessions.get_session(session_id)
        if not sess or sess.status == "completed":
            return

        remaining_budget = sess.max_tasks - int(sess.tasks_created or 0)
        queued_left = self.tasks.queued_count(session_id)

        # Session is logically "done" when budget exhausted or no queued tasks
        if remaining_budget > 0 and queued_left > 0:
            return

        logger.info(f"Session {session_id} done (budget={remaining_budget}, queued={queued_left}), running synthesis...")

        # Run synthesis once if available
        synth_handler = METHODS.get("SynthesizeFindings")
        if synth_handler:
            from uuid import uuid4
            synth_task = Task(
                id=str(uuid4()),
                session_id=session_id,
                htn_task_type="SynthesizeFindings",
                args={},
                status="running",
                depth=0,
                parent_id=None,
            )
            try:
                proposals = synth_handler(task=synth_task, ctx=self.ctx)
                if proposals:
                    # SynthesizeFindings should not create children, but guard anyway
                    logger.warning(f"SynthesizeFindings unexpectedly returned {len(proposals)} proposals")
                    self._enforce_budgets_and_enqueue(synth_task, proposals)
            except Exception as e:
                logger.error(f"SynthesizeFindings failed for session {session_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("No SynthesizeFindings method registered; skipping synthesis")

        # Finally mark session complete
        self.sessions.mark_complete(session_id)
        logger.info(f"Session {session_id} marked complete")

        # Log session completion
        get_multi_logger().log_research_event(
            event_type="session_complete",
            session_id=session_id,
            data={
                "tasks_created": sess.tasks_created,
                "max_tasks": sess.max_tasks,
                "budget_remaining": remaining_budget
            }
        )
