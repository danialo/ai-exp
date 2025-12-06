#!/usr/bin/env python3
"""
Rollback Script for HTN Self-Belief Decomposer.

Soft-deletes all BeliefOccurrences created by a specific extractor version.
Uses soft delete (sets deleted_at timestamp) rather than hard delete.

Usage:
    python scripts/rollback_extractor_version.py --version <version_hash> [options]

Options:
    --version HASH  Extractor version hash to rollback (required)
    --hard-delete   Actually delete rows instead of soft delete (destructive!)
    --dry-run       Show what would be deleted without making changes
    --verbose       Enable verbose logging

Non-destructive by default: Sets deleted_at timestamp, data can be recovered.
After rollback, affected BeliefNodes may become orphaned (no occurrences).

To find version hashes:
    SELECT DISTINCT extractor_version, COUNT(*)
    FROM belief_occurrences
    GROUP BY extractor_version;
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
from src.memory.models.belief_occurrence import BeliefOccurrence
from src.memory.models.belief_node import BeliefNode
from src.memory.models.tentative_link import TentativeLink
from src.services.self_knowledge_index import SelfKnowledgeIndex

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class RollbackProcessor:
    """
    Rollback occurrences for a specific extractor version.

    This class handles:
    - Soft-deleting BeliefOccurrences by version
    - Optionally hard-deleting (destructive)
    - Identifying orphaned BeliefNodes after rollback
    - Statistics collection
    """

    def __init__(
        self,
        db_url: str,
        version: str,
        hard_delete: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the rollback processor.

        Args:
            db_url: Database connection URL
            version: Extractor version hash to rollback
            hard_delete: Delete rows instead of soft delete
            dry_run: Don't persist changes
            verbose: Enable verbose logging
        """
        self.db_url = db_url
        self.version = version
        self.hard_delete = hard_delete
        self.dry_run = dry_run
        self.verbose = verbose

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Get current extractor version for comparison
        self.config = get_belief_config()
        self.current_version = get_extractor_version(self.config)

        # Statistics
        self.stats = {
            'occurrences_affected': 0,
            'nodes_orphaned': 0,
            'tentative_links_affected': 0,
            'self_knowledge_claims_removed': 0,
        }

        # SelfKnowledgeIndex for remove_claim (TASK 10.4)
        self.self_knowledge_index = SelfKnowledgeIndex()

    def run(self):
        """Run the rollback process."""
        logger.info(f"Rolling back extractor version: {self.version}")
        logger.info(f"Current extractor version: {self.current_version}")

        if self.version == self.current_version:
            logger.warning("WARNING: You are rolling back the CURRENT extractor version!")

        if self.hard_delete:
            logger.warning("HARD DELETE MODE - data will be permanently removed!")

        if self.dry_run:
            logger.info("DRY RUN MODE - no changes will be persisted")

        engine = create_engine(self.db_url)

        with Session(engine) as session:
            # Count affected occurrences
            occurrences = self._get_affected_occurrences(session)

            if not occurrences:
                logger.info(f"No occurrences found for version {self.version}")
                return

            logger.info(f"Found {len(occurrences)} occurrences to roll back")

            # Soft delete or hard delete occurrences
            if self.hard_delete:
                self._hard_delete_occurrences(session, occurrences)
            else:
                self._soft_delete_occurrences(session, occurrences)

            self.stats['occurrences_affected'] = len(occurrences)

            # Remove claims from SelfKnowledgeIndex (TASK 10.4)
            experience_ids = set(occ.source_experience_id for occ in occurrences)
            for experience_id in experience_ids:
                if not self.dry_run:
                    removed = self.self_knowledge_index.remove_claim(experience_id)
                    self.stats['self_knowledge_claims_removed'] += removed
                    if self.verbose and removed > 0:
                        logger.debug(f"Removed {removed} claims for experience {experience_id}")

            # Find orphaned nodes (nodes with no remaining occurrences)
            orphaned_nodes = self._find_orphaned_nodes(session)
            self.stats['nodes_orphaned'] = len(orphaned_nodes)

            if orphaned_nodes:
                logger.info(f"Found {len(orphaned_nodes)} potentially orphaned nodes")
                self._handle_orphaned_nodes(session, orphaned_nodes)

            # Handle tentative links that reference this version
            affected_links = self._get_affected_tentative_links(session)
            self.stats['tentative_links_affected'] = len(affected_links)

            if affected_links:
                logger.info(f"Found {len(affected_links)} affected tentative links")
                self._handle_affected_links(session, affected_links)

            if not self.dry_run:
                session.commit()

        self._report_stats()

    def _get_affected_occurrences(self, session: Session) -> list:
        """Get occurrences to roll back."""
        occurrences = session.exec(
            select(BeliefOccurrence).where(
                BeliefOccurrence.extractor_version == self.version,
                BeliefOccurrence.deleted_at.is_(None)  # Not already deleted
            )
        ).all()

        return list(occurrences)

    def _soft_delete_occurrences(self, session: Session, occurrences: list):
        """
        Soft delete occurrences by setting deleted_at.

        This preserves the data for potential recovery.
        """
        now = datetime.now(timezone.utc)

        for occ in occurrences:
            if self.verbose:
                logger.debug(f"Soft deleting occurrence {occ.occurrence_id}")

            if not self.dry_run:
                occ.deleted_at = now
                session.add(occ)

        logger.info(f"Soft deleted {len(occurrences)} occurrences")

    def _hard_delete_occurrences(self, session: Session, occurrences: list):
        """
        Hard delete occurrences (permanent).

        WARNING: This cannot be undone!
        """
        for occ in occurrences:
            if self.verbose:
                logger.debug(f"Hard deleting occurrence {occ.occurrence_id}")

            if not self.dry_run:
                session.delete(occ)

        logger.info(f"Hard deleted {len(occurrences)} occurrences")

    def _find_orphaned_nodes(self, session: Session) -> list:
        """
        Find BeliefNodes that have no remaining occurrences.

        A node is orphaned if:
        - All its occurrences are either deleted or from the rolled-back version
        """
        # Get all unique belief_ids from rolled back occurrences
        rolled_back_occurrences = session.exec(
            select(BeliefOccurrence.belief_id).where(
                BeliefOccurrence.extractor_version == self.version
            ).distinct()
        ).all()

        orphaned = []

        for (belief_id,) in rolled_back_occurrences:
            # Check if any non-deleted occurrences remain
            remaining = session.exec(
                select(BeliefOccurrence).where(
                    BeliefOccurrence.belief_id == belief_id,
                    BeliefOccurrence.extractor_version != self.version,
                    BeliefOccurrence.deleted_at.is_(None)
                )
            ).first()

            if not remaining:
                node = session.get(BeliefNode, belief_id)
                if node:
                    orphaned.append(node)

        return orphaned

    def _handle_orphaned_nodes(self, session: Session, nodes: list):
        """
        Handle orphaned nodes by updating their status.

        Does NOT delete nodes - just marks them as orphaned.
        """
        for node in nodes:
            if self.verbose:
                logger.debug(f"Marking node {node.belief_id} as orphaned")

            if not self.dry_run:
                node.status = 'orphaned'
                node.activation = 0.0  # Reset activation since no evidence
                session.add(node)

        logger.info(f"Marked {len(nodes)} nodes as orphaned")

    def _get_affected_tentative_links(self, session: Session) -> list:
        """Get tentative links created by this version."""
        links = session.exec(
            select(TentativeLink).where(
                TentativeLink.extractor_version == self.version
            )
        ).all()

        return list(links)

    def _handle_affected_links(self, session: Session, links: list):
        """
        Handle tentative links from rolled-back version.

        Updates status to 'rolled_back' rather than deleting.
        """
        for link in links:
            if self.verbose:
                logger.debug(f"Marking link {link.link_id} as rolled_back")

            if not self.dry_run:
                link.status = 'rolled_back'
                session.add(link)

        logger.info(f"Marked {len(links)} tentative links as rolled_back")

    def _report_stats(self):
        """Print final statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("ROLLBACK COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Version rolled back: {self.version}")
        logger.info(f"Occurrences affected: {self.stats['occurrences_affected']}")
        logger.info(f"Nodes orphaned: {self.stats['nodes_orphaned']}")
        logger.info(f"Tentative links affected: {self.stats['tentative_links_affected']}")
        logger.info(f"SelfKnowledgeIndex claims removed: {self.stats['self_knowledge_claims_removed']}")

        if self.hard_delete:
            logger.info("\nHARD DELETE was used - data is permanently removed")
        else:
            logger.info("\nSoft delete was used - data can be recovered")

        if self.dry_run:
            logger.info("\nDRY RUN - no changes were persisted")


def list_versions(db_url: str):
    """List all extractor versions in the database with counts."""
    engine = create_engine(db_url)

    with Session(engine) as session:
        # Query version counts
        query = text("""
            SELECT
                extractor_version,
                COUNT(*) as occurrence_count,
                COUNT(DISTINCT belief_id) as node_count,
                MIN(created_at) as first_seen,
                MAX(created_at) as last_seen
            FROM belief_occurrences
            WHERE deleted_at IS NULL
            GROUP BY extractor_version
            ORDER BY first_seen DESC
        """)

        try:
            result = session.exec(query)
            rows = result.fetchall()

            if not rows:
                print("No extractor versions found in database")
                return

            print("\nExtractor versions in database:")
            print("-" * 80)
            print(f"{'Version':<18} {'Occurrences':>12} {'Nodes':>8} {'First Seen':<20} {'Last Seen':<20}")
            print("-" * 80)

            for row in rows:
                version, occ_count, node_count, first_seen, last_seen = row
                print(f"{version:<18} {occ_count:>12} {node_count:>8} {str(first_seen)[:19]:<20} {str(last_seen)[:19]:<20}")

            print("-" * 80)

            # Show current version
            config = get_belief_config()
            current = get_extractor_version(config)
            print(f"\nCurrent extractor version: {current}")

        except Exception as e:
            print(f"Error querying versions: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Rollback beliefs from a specific extractor version'
    )
    parser.add_argument(
        '--version', type=str, default=None,
        help='Extractor version hash to rollback'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all extractor versions in database'
    )
    parser.add_argument(
        '--hard-delete', action='store_true',
        help='Permanently delete instead of soft delete'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Don\'t make changes, just show what would happen'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--db-url', type=str, default=None,
        help='Database URL (default: from config)'
    )

    args = parser.parse_args()

    # Determine database URL
    db_url = args.db_url
    if not db_url:
        import os
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            db_path = project_root / 'data' / 'astra.db'
            db_url = f'sqlite:///{db_path}'

    if args.list:
        list_versions(db_url)
        return

    if not args.version:
        parser.error("--version is required (or use --list to see available versions)")

    # Confirm hard delete
    if args.hard_delete and not args.dry_run:
        confirm = input(
            f"WARNING: You are about to PERMANENTLY DELETE all data for version {args.version}.\n"
            "This cannot be undone. Type 'yes' to confirm: "
        )
        if confirm.lower() != 'yes':
            print("Aborted.")
            return

    processor = RollbackProcessor(
        db_url=db_url,
        version=args.version,
        hard_delete=args.hard_delete,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    processor.run()


if __name__ == '__main__':
    main()
