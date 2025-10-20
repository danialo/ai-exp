#!/usr/bin/env python3
"""Initialize the experience memory databases.

Creates the SQLite raw store and ChromaDB vector index with proper schema.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.memory.raw_store import create_raw_store
from src.memory.vector_store import create_vector_store
from src.memory.embedding import create_embedding_provider


def init_databases(reset: bool = False) -> None:
    """Initialize or reset the databases.

    Args:
        reset: If True, delete existing databases and start fresh
    """
    print("Initializing AI Experience Memory System databases...")
    print(f"Raw store path: {settings.RAW_STORE_DB_PATH}")
    print(f"Vector index path: {settings.VECTOR_INDEX_PATH}")

    # Ensure data directories exist
    settings.ensure_data_directories()

    # Handle reset for raw store
    if reset:
        raw_store_path = Path(settings.RAW_STORE_DB_PATH)
        if raw_store_path.exists():
            print(f"  Deleting existing raw store: {raw_store_path}")
            raw_store_path.unlink()

    # Initialize raw store (SQLite)
    print("\n[1/3] Initializing raw store (SQLite)...")
    raw_store = create_raw_store(db_path=settings.RAW_STORE_DB_PATH)
    exp_count = raw_store.count_experiences()
    print(f"  ✓ Raw store initialized with {exp_count} experiences")

    # Initialize vector store (ChromaDB)
    print("\n[2/3] Initializing vector store (ChromaDB)...")
    vector_store = create_vector_store(
        persist_directory=settings.VECTOR_INDEX_PATH,
        collection_name="experiences",
        reset=reset,
    )
    print(f"  ✓ Vector store initialized with {vector_store.count()} vectors")

    # Test embedding provider
    print(f"\n[3/3] Testing embedding provider ({settings.EMBEDDING_MODEL})...")
    embedding_provider = create_embedding_provider(
        model_name=settings.EMBEDDING_MODEL,
        use_mock=False,
    )
    test_embedding = embedding_provider.embed("Test sentence")
    print(f"  ✓ Embedding provider ready (dimension: {test_embedding.shape[0]})")

    print("\n✅ Database initialization complete!")
    print("\nYou can now:")
    print("  - Ingest experiences using the IngestionPipeline")
    print("  - Query experiences using the RetrievalService")
    print("  - Run the CLI harness (when implemented)")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize AI Experience Memory System databases"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing databases (WARNING: deletes all data)",
    )
    args = parser.parse_args()

    if args.reset:
        response = input(
            "⚠️  WARNING: This will delete all existing data. Continue? (yes/no): "
        )
        if response.lower() != "yes":
            print("Aborted.")
            return

    try:
        init_databases(reset=args.reset)
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
