#!/usr/bin/env python3
"""
Backfill Script for HTN Self-Belief Decomposer.

Processes existing Experience(type='self_definition') rows and extracts
beliefs using the HTN pipeline.

Usage:
    python scripts/backfill_self_definitions_to_beliefs.py [options]

Options:
    --limit N       Process at most N experiences (default: all)
    --offset N      Skip first N experiences
    --batch-size N  Commit every N experiences (default: 10)
    --dry-run       Don't write to database, just report what would be done
    --verbose       Enable verbose logging
    --experience-id ID  Process a specific experience ID only

Non-destructive: Does not modify Experience rows, only adds to belief tables.
Idempotent: Can be safely re-run (uses extractor_version in unique constraint).
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sqlmodel import Session, create_engine, select, text

from src.utils.belief_config import get_belief_config
from src.utils.extractor_version import get_extractor_version
from src.services.htn_belief_methods import HTNBeliefExtractor, ExtractionResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class BackfillProcessor:
    """
    Process Experience rows and extract beliefs.

    This class handles:
    - Querying Experience(type='self_definition') rows
    - Batched processing with progress tracking
    - Error handling and recovery
    - Statistics collection
    """

    def __init__(
        self,
        db_url: str,
        limit: int = None,
        offset: int = 0,
        batch_size: int = 10,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the backfill processor.

        Args:
            db_url: Database connection URL
            limit: Maximum experiences to process (None = all)
            offset: Number of experiences to skip
            batch_size: Commit every N experiences
            dry_run: Don't persist changes
            verbose: Enable verbose logging
        """
        self.db_url = db_url
        self.limit = limit
        self.offset = offset
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.verbose = verbose

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Load config and get version
        self.config = get_belief_config()
        self.extractor_version = get_extractor_version(self.config)

        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_atoms': 0,
            'new_nodes': 0,
            'matched_nodes': 0,
            'tentative_links': 0,
            'conflicts': 0,
            'errors': [],
        }

    def run(self, experience_id: str = None):
        """
        Run the backfill process.

        Args:
            experience_id: Optional specific experience to process
        """
        logger.info(f"Starting backfill with extractor version: {self.extractor_version}")

        if self.dry_run:
            logger.info("DRY RUN MODE - no changes will be persisted")

        engine = create_engine(self.db_url)

        with Session(engine) as session:
            # Get experiences to process
            experiences = self._get_experiences(session, experience_id)

            if not experiences:
                logger.info("No experiences to process")
                return

            logger.info(f"Found {len(experiences)} experiences to process")

            # Create extractor
            # Note: In production, you would inject an LLM client here
            extractor = HTNBeliefExtractor(
                config=self.config,
                llm_client=None,  # Will use simple fallback extraction
                db_session=session if not self.dry_run else None,
            )

            # Process in batches
            for i, exp in enumerate(experiences):
                try:
                    self._process_experience(extractor, exp, session)
                    self.stats['successful'] += 1
                except Exception as e:
                    logger.error(f"Error processing experience {exp['id']}: {e}")
                    self.stats['failed'] += 1
                    self.stats['errors'].append({
                        'experience_id': exp['id'],
                        'error': str(e),
                    })
                    if not self.dry_run:
                        session.rollback()

                self.stats['total_processed'] += 1

                # Commit batch
                if (i + 1) % self.batch_size == 0 and not self.dry_run:
                    session.commit()
                    logger.info(f"Processed {i + 1}/{len(experiences)} experiences")

            # Final commit
            if not self.dry_run:
                session.commit()

        self._report_stats()

    def _get_experiences(self, session: Session, experience_id: str = None) -> list:
        """
        Get Experience rows to process.

        Returns list of dicts with id, content, affect, session_id.
        """
        # Build query for Experience(type='self_definition')
        # Assuming experiences table exists with these columns
        query_parts = [
            "SELECT id, content, affect, session_id",
            "FROM experiences",
            "WHERE type = 'self_definition'",
        ]

        if experience_id:
            query_parts.append(f"AND id = '{experience_id}'")

        query_parts.append("ORDER BY created_at ASC")

        if self.limit:
            query_parts.append(f"LIMIT {self.limit}")

        if self.offset:
            query_parts.append(f"OFFSET {self.offset}")

        query = " ".join(query_parts)

        try:
            result = session.exec(text(query))
            rows = result.fetchall()

            experiences = []
            for row in rows:
                exp = {
                    'id': row[0],
                    'content': row[1],
                    'affect': row[2],
                    'session_id': row[3],
                }
                experiences.append(exp)

            return experiences

        except Exception as e:
            logger.error(f"Error querying experiences: {e}")
            # Try fallback for SQLite
            return self._get_experiences_sqlite(session, experience_id)

    def _get_experiences_sqlite(self, session: Session, experience_id: str = None) -> list:
        """
        Fallback for SQLite databases.
        """
        import sqlite3
        import json

        # Extract database path from URL
        db_path = self.db_url.replace('sqlite:///', '')

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
            SELECT id, content, affect, session_id
            FROM experiences
            WHERE type = 'self_definition'
        """

        params = []
        if experience_id:
            query += " AND id = ?"
            params.append(experience_id)

        query += " ORDER BY created_at ASC"

        if self.limit:
            query += f" LIMIT {self.limit}"
        if self.offset:
            query += f" OFFSET {self.offset}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        experiences = []
        for row in rows:
            content = row[1]
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = {'text': content}

            affect = row[2]
            if isinstance(affect, str):
                try:
                    affect = json.loads(affect)
                except json.JSONDecodeError:
                    affect = {}

            exp = {
                'id': row[0],
                'content': content,
                'affect': affect or {},
                'session_id': row[3],
            }
            experiences.append(exp)

        return experiences

    def _process_experience(
        self,
        extractor: HTNBeliefExtractor,
        exp_dict: dict,
        session: Session
    ):
        """
        Process a single experience.

        Args:
            extractor: HTNBeliefExtractor instance
            exp_dict: Experience data as dict
            session: Database session
        """
        # Create experience-like object
        class ExperienceProxy:
            def __init__(self, data):
                self.id = data['id']
                self.content = data['content']
                self.affect = data['affect']
                self.session_id = data['session_id']

        exp = ExperienceProxy(exp_dict)

        if self.verbose:
            content = exp.content
            text = content.get('text', str(content)) if isinstance(content, dict) else str(content)
            logger.debug(f"Processing experience {exp.id}: {text[:100]}...")

        # Extract beliefs
        result: ExtractionResult = extractor.extract_and_update_self_knowledge(exp)

        # Update stats
        self.stats['total_atoms'] += len(result.dedup_result.deduped_atoms)
        self.stats['new_nodes'] += result.stats.get('nodes_created', 0)
        self.stats['matched_nodes'] += result.stats.get('nodes_matched', 0)
        self.stats['tentative_links'] += result.stats.get('tentative_links_created', 0)
        self.stats['conflicts'] += result.stats.get('conflicts_detected', 0)

        if self.verbose:
            logger.debug(
                f"  Extracted {len(result.dedup_result.deduped_atoms)} atoms, "
                f"{result.stats.get('nodes_created', 0)} new nodes, "
                f"{result.stats.get('nodes_matched', 0)} matched"
            )

    def _report_stats(self):
        """Print final statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("BACKFILL COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Extractor version: {self.extractor_version}")
        logger.info(f"Total experiences processed: {self.stats['total_processed']}")
        logger.info(f"  Successful: {self.stats['successful']}")
        logger.info(f"  Failed: {self.stats['failed']}")
        logger.info(f"  Skipped: {self.stats['skipped']}")
        logger.info(f"Total atoms extracted: {self.stats['total_atoms']}")
        logger.info(f"  New belief nodes: {self.stats['new_nodes']}")
        logger.info(f"  Matched to existing: {self.stats['matched_nodes']}")
        logger.info(f"Tentative links created: {self.stats['tentative_links']}")
        logger.info(f"Conflicts detected: {self.stats['conflicts']}")

        if self.stats['errors']:
            logger.info(f"\nErrors ({len(self.stats['errors'])}):")
            for err in self.stats['errors'][:10]:  # Show first 10
                logger.info(f"  {err['experience_id']}: {err['error']}")
            if len(self.stats['errors']) > 10:
                logger.info(f"  ... and {len(self.stats['errors']) - 10} more")

        if self.dry_run:
            logger.info("\nDRY RUN - no changes were persisted")


def main():
    parser = argparse.ArgumentParser(
        description='Backfill beliefs from existing self_definition experiences'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Maximum experiences to process'
    )
    parser.add_argument(
        '--offset', type=int, default=0,
        help='Skip first N experiences'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10,
        help='Commit every N experiences'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Don\'t write to database'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--experience-id', type=str, default=None,
        help='Process specific experience ID'
    )
    parser.add_argument(
        '--db-url', type=str, default=None,
        help='Database URL (default: from config)'
    )

    args = parser.parse_args()

    # Determine database URL
    db_url = args.db_url
    if not db_url:
        # Try to get from environment or default
        import os
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            # Default SQLite path
            db_path = project_root / 'data' / 'astra.db'
            db_url = f'sqlite:///{db_path}'

    logger.info(f"Using database: {db_url}")

    processor = BackfillProcessor(
        db_url=db_url,
        limit=args.limit,
        offset=args.offset,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    processor.run(experience_id=args.experience_id)


if __name__ == '__main__':
    main()
