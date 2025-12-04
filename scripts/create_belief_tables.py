#!/usr/bin/env python3
"""
Create belief extraction tables in the database.

This script creates the tables for the HTN Self-Belief Decomposer:
- belief_nodes
- belief_occurrences
- tentative_links
- conflict_edges
- stream_assignments

Usage:
    python scripts/create_belief_tables.py [--db-path PATH]

By default, uses data/raw_store.db (same database as experiences).
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlmodel import SQLModel, create_engine

# Import all models to register them with SQLModel
from src.memory.models.belief_node import BeliefNode
from src.memory.models.belief_occurrence import BeliefOccurrence
from src.memory.models.tentative_link import TentativeLink
from src.memory.models.conflict_edge import ConflictEdge
from src.memory.models.stream_assignment import StreamAssignment


def create_tables(db_path: str, echo: bool = False) -> None:
    """
    Create all belief-related tables in the database.

    Args:
        db_path: Path to SQLite database file
        echo: If True, print SQL statements
    """
    # Ensure parent directory exists
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Create engine
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)

    # Create only our tables (won't recreate existing tables)
    # SQLModel.metadata.create_all only creates tables that don't exist
    SQLModel.metadata.create_all(engine)

    print(f"Tables created successfully in {db_path}")

    # Verify tables exist
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    expected_tables = [
        'belief_nodes',
        'belief_occurrences',
        'tentative_links',
        'conflict_edges',
        'stream_assignments',
    ]

    print("\nTable status:")
    for table in expected_tables:
        status = "OK" if table in tables else "MISSING"
        print(f"  {table}: {status}")


def verify_schema(db_path: str) -> bool:
    """
    Verify that all required columns exist in the tables.

    Returns True if schema is valid, False otherwise.
    """
    from sqlalchemy import create_engine, inspect

    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)

    schema_valid = True

    # Expected columns for each table
    expected_schema = {
        'belief_nodes': [
            'belief_id', 'canonical_text', 'canonical_hash', 'belief_type',
            'polarity', 'created_at', 'last_reinforced_at', 'activation',
            'core_score', 'status', 'embedding'
        ],
        'belief_occurrences': [
            'occurrence_id', 'belief_id', 'source_experience_id', 'extractor_version',
            'raw_text', 'raw_span', 'source_weight', 'atom_confidence',
            'epistemic_frame', 'epistemic_confidence', 'match_confidence',
            'context_id', 'created_at', 'deleted_at'
        ],
        'tentative_links': [
            'link_id', 'from_belief_id', 'to_belief_id', 'confidence',
            'status', 'support_both', 'support_one', 'last_support_at',
            'signals', 'extractor_version', 'created_at', 'updated_at'
        ],
        'conflict_edges': [
            'edge_id', 'from_belief_id', 'to_belief_id', 'conflict_type',
            'status', 'evidence_occurrence_ids', 'created_at', 'updated_at'
        ],
        'stream_assignments': [
            'belief_id', 'primary_stream', 'secondary_stream', 'confidence',
            'migrated_from', 'updated_at'
        ],
    }

    print("\nSchema verification:")
    for table_name, expected_cols in expected_schema.items():
        if table_name not in inspector.get_table_names():
            print(f"  {table_name}: TABLE MISSING")
            schema_valid = False
            continue

        actual_cols = [col['name'] for col in inspector.get_columns(table_name)]
        missing = set(expected_cols) - set(actual_cols)
        extra = set(actual_cols) - set(expected_cols)

        if missing:
            print(f"  {table_name}: MISSING COLUMNS: {missing}")
            schema_valid = False
        elif extra:
            print(f"  {table_name}: OK (extra columns: {extra})")
        else:
            print(f"  {table_name}: OK")

    return schema_valid


def main():
    parser = argparse.ArgumentParser(
        description="Create belief extraction tables in the database"
    )
    parser.add_argument(
        '--db-path',
        default='data/raw_store.db',
        help='Path to SQLite database (default: data/raw_store.db)'
    )
    parser.add_argument(
        '--echo',
        action='store_true',
        help='Print SQL statements'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify schema, do not create tables'
    )

    args = parser.parse_args()

    if args.verify_only:
        if not Path(args.db_path).exists():
            print(f"Database not found: {args.db_path}")
            sys.exit(1)
        valid = verify_schema(args.db_path)
        sys.exit(0 if valid else 1)

    create_tables(args.db_path, echo=args.echo)
    verify_schema(args.db_path)


if __name__ == '__main__':
    main()
