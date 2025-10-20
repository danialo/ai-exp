#!/usr/bin/env python3
"""Add ownership column to experience table."""

import sqlite3
import sys

def migrate():
    """Add ownership column to existing database."""
    db_path = "data/raw_store.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if column already exists
        cursor.execute("PRAGMA table_info(experience)")
        columns = [row[1] for row in cursor.fetchall()]

        if "ownership" in columns:
            print("✓ Column 'ownership' already exists")
            return

        # Add the column
        print("Adding 'ownership' column...")
        cursor.execute("ALTER TABLE experience ADD COLUMN ownership TEXT DEFAULT 'user'")
        conn.commit()

        print("✓ Migration complete: ownership column added")

    except Exception as e:
        print(f"✗ Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
