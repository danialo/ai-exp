#!/usr/bin/env python3
"""Backfill existing task execution results as TASK_EXECUTION experiences.

This script scans persona_space/tasks/results/*.json files and creates
TASK_EXECUTION experiences for historical task executions.

Features:
- Idempotent execution (safe to re-run)
- Dry-run mode for preview
- Progress reporting
- Error handling with detailed logging
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.raw_store import RawStore
from src.pipeline.task_experience import create_task_execution_experience
from src.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_task_result(result_path: Path) -> Optional[Dict]:
    """Parse a task result JSON file.

    Args:
        result_path: Path to task result JSON file

    Returns:
        Parsed task result dict, or None if parsing fails
    """
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to parse {result_path}: {e}")
        return None


def infer_task_type_from_id(task_id: str) -> str:
    """Infer task type from task_id.

    Args:
        task_id: Task identifier

    Returns:
        Inferred task type
    """
    # Map known task IDs to types
    type_map = {
        "daily_reflection": "reflection",
        "weekly_goal_assessment": "assessment",
        "memory_consolidation": "consolidation",
        "capability_exploration": "exploration",
        "emotional_check_in": "reflection",
        "self_inquiry": "reflection",
        "authentic_response": "reflection",
        "belief_consolidation": "consolidation",
    }

    return type_map.get(task_id, "custom")


def backfill_task_execution(
    result_data: Dict,
    result_path: Path,
    raw_store: RawStore,
    dry_run: bool = False
) -> Optional[str]:
    """Create a TASK_EXECUTION experience from a task result JSON.

    Args:
        result_data: Parsed task result data
        result_path: Path to result file (for logging)
        raw_store: RawStore instance
        dry_run: If True, don't actually create the experience

    Returns:
        Experience ID if created, None otherwise
    """
    try:
        # Extract task metadata
        task_id = result_data.get("task_id", "unknown")
        task_name = result_data.get("task_name", "Unknown Task")
        started_at_iso = result_data.get("started_at")
        completed_at_iso = result_data.get("completed_at")
        success = result_data.get("success", False)
        response_text = result_data.get("response")
        error_str = result_data.get("error")
        metadata = result_data.get("metadata", {})

        # Parse timestamps
        if not started_at_iso or not completed_at_iso:
            logger.warning(f"{result_path}: Missing timestamps, skipping")
            return None

        started_at = datetime.fromisoformat(started_at_iso)
        completed_at = datetime.fromisoformat(completed_at_iso)

        # Ensure UTC-aware
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)

        # Determine status and error
        status = "success" if success else "failed"
        error_details = None
        if not success and error_str:
            error_details = {
                "type": "UnknownError",  # We don't have the original exception type
                "message": error_str,
                "stack_hash": str(hash(error_str))[:16],
            }

        # Infer task type
        task_type = metadata.get("task_type", infer_task_type_from_id(task_id))

        # Build task config (limited info from result file)
        task_config = {
            "type": task_type,
            "backfilled": True,
        }

        # Retrieval metadata (unknown for backfilled tasks)
        retrieval_metadata = {
            "memory_count": 0,
            "source": [],
            "backfilled": True,  # Flag to indicate this is estimated
        }

        # Create experience
        experience = create_task_execution_experience(
            task_id=task_id,
            task_slug=task_id,
            task_name=task_name,
            task_type=task_type,
            scheduled_vs_manual="scheduled",  # Assume scheduled
            started_at=started_at,
            ended_at=completed_at,
            status=status,
            response_text=response_text,
            error=error_details,
            parent_experience_ids=[],  # Unknown for backfilled
            retrieval_metadata=retrieval_metadata,
            files_written=[],  # Unknown for backfilled
            task_config=task_config,
            trace_id=None,  # Will be auto-generated
            span_id=None,  # Will be auto-generated
            attempt=1,
            retry_of=None,
        )

        # Mark as backfilled in structured content
        experience.content.structured["backfilled"] = True
        experience.content.structured["legacy_result_file"] = str(result_path)

        # Extract idempotency key
        idempotency_key = experience.content.structured["idempotency_key"]

        if dry_run:
            logger.info(f"[DRY RUN] Would create experience for {task_id} from {result_path.name}")
            return None

        # Store experience idempotently
        experience_id = raw_store.append_experience_idempotent(experience, idempotency_key)

        logger.info(f"Created experience {experience_id} for {task_id} from {result_path.name}")
        return experience_id

    except Exception as e:
        logger.error(f"Failed to backfill {result_path}: {e}", exc_info=True)
        return None


def main():
    """Main backfill script."""
    parser = argparse.ArgumentParser(
        description="Backfill task execution results as TASK_EXECUTION experiences"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="persona_space/tasks/results",
        help="Directory containing task result JSON files (default: persona_space/tasks/results)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(settings.RAW_STORE_DB_PATH),
        help=f"Path to raw store database (default: {settings.RAW_STORE_DB_PATH})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be backfilled without actually creating experiences"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1

    # Initialize raw store
    logger.info(f"Opening raw store: {args.db_path}")
    raw_store = RawStore(args.db_path)

    # Find all result JSON files
    result_files = sorted(results_dir.glob("*.json"))
    logger.info(f"Found {len(result_files)} task result files in {results_dir}")

    if args.dry_run:
        logger.info("DRY RUN MODE: No experiences will be created")

    # Process each result file
    created_count = 0
    skipped_count = 0
    error_count = 0

    for result_path in result_files:
        # Parse result
        result_data = parse_task_result(result_path)
        if result_data is None:
            error_count += 1
            continue

        # Backfill
        experience_id = backfill_task_execution(
            result_data=result_data,
            result_path=result_path,
            raw_store=raw_store,
            dry_run=args.dry_run
        )

        if experience_id:
            created_count += 1
        else:
            skipped_count += 1

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Backfill Summary")
    logger.info("=" * 60)
    logger.info(f"Total files processed:  {len(result_files)}")
    logger.info(f"Experiences created:    {created_count}")
    logger.info(f"Skipped:                {skipped_count}")
    logger.info(f"Errors:                 {error_count}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN completed. Re-run without --dry-run to actually create experiences.")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
