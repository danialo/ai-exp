#!/usr/bin/env python3
"""Backfill last_reinforced field for existing self-definition experiences.

Self-definitions created before the reinforcement tracking was added don't have
a last_reinforced timestamp. This script sets last_reinforced = created_at for
any self-definitions missing this field.

Features:
- Idempotent execution (safe to re-run)
- Dry-run mode for preview
- Progress reporting
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlmodel import Session as DBSession, select
from src.memory.raw_store import RawStore
from src.memory.models import Experience, ExperienceType
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_last_reinforced(db_path: str, dry_run: bool = True) -> dict:
    """Backfill last_reinforced for self-definitions missing it.

    Args:
        db_path: Path to the raw store database
        dry_run: If True, only report what would be changed

    Returns:
        Summary of changes made
    """
    raw_store = RawStore(db_path)

    results = {
        "total_self_defs": 0,
        "missing_last_reinforced": 0,
        "updated": 0,
        "already_set": 0,
        "errors": 0,
        "dry_run": dry_run,
    }

    with DBSession(raw_store.engine) as db:
        # Get all SELF_DEFINITION experiences
        stmt = select(Experience).where(
            Experience.type == ExperienceType.SELF_DEFINITION.value
        )
        self_defs = db.exec(stmt).all()
        results["total_self_defs"] = len(self_defs)

        logger.info(f"Found {len(self_defs)} self-definition experiences")

        for exp in self_defs:
            try:
                content = exp.content or {}
                structured = content.get("structured", {})

                # Check if last_reinforced already exists
                if structured.get("last_reinforced"):
                    results["already_set"] += 1
                    continue

                results["missing_last_reinforced"] += 1

                # Get created_at as ISO string
                created_at = exp.created_at
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                last_reinforced_value = created_at.isoformat()

                if dry_run:
                    text_preview = content.get("text", "")[:50]
                    logger.info(f"Would set last_reinforced={last_reinforced_value} for: {text_preview}...")
                else:
                    # Update the structured data
                    if "structured" not in content:
                        content["structured"] = {}
                    content["structured"]["last_reinforced"] = last_reinforced_value

                    # Also set reinforcement_count to 1 if missing
                    if "reinforcement_count" not in content["structured"]:
                        content["structured"]["reinforcement_count"] = 1

                    exp.content = content
                    db.add(exp)
                    results["updated"] += 1

                    text_preview = content.get("text", "")[:50]
                    logger.debug(f"Updated: {text_preview}...")

            except Exception as e:
                logger.error(f"Error processing {exp.id}: {e}")
                results["errors"] += 1

        if not dry_run:
            db.commit()
            logger.info(f"Committed {results['updated']} updates")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Backfill last_reinforced for self-definition experiences"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(settings.RAW_STORE_DB_PATH),
        help=f"Path to raw store database (default: {settings.RAW_STORE_DB_PATH})"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the backfill (default is dry-run)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dry_run = not args.execute

    if dry_run:
        logger.info("=== DRY RUN MODE (use --execute to apply changes) ===")
    else:
        logger.info("=== EXECUTING BACKFILL ===")

    logger.info(f"Database: {args.db_path}")
    results = backfill_last_reinforced(db_path=args.db_path, dry_run=dry_run)

    print("\n" + "="*50)
    print("BACKFILL SUMMARY")
    print("="*50)
    print(f"Total self-definitions:    {results['total_self_defs']}")
    print(f"Already had last_reinforced: {results['already_set']}")
    print(f"Missing last_reinforced:   {results['missing_last_reinforced']}")
    if dry_run:
        print(f"Would update:              {results['missing_last_reinforced']}")
    else:
        print(f"Updated:                   {results['updated']}")
    print(f"Errors:                    {results['errors']}")
    print("="*50)

    if dry_run and results['missing_last_reinforced'] > 0:
        print("\nRun with --execute to apply these changes")


if __name__ == "__main__":
    main()
