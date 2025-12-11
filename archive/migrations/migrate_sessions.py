"""Migration script to add session tracking columns to existing database."""

import sqlite3
from pathlib import Path

def migrate():
    db_path = Path("data/raw_store.db")

    if not db_path.exists():
        print("Database doesn't exist yet, skipping migration")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if columns already exist
    cursor.execute("PRAGMA table_info(experience)")
    columns = [row[1] for row in cursor.fetchall()]

    print(f"Current experience columns: {columns}")

    # Add session_id if missing
    if "session_id" not in columns:
        print("Adding session_id column...")
        cursor.execute("ALTER TABLE experience ADD COLUMN session_id TEXT")
        print("✓ Added session_id column")
    else:
        print("✓ session_id column already exists")

    # Add consolidated if missing
    if "consolidated" not in columns:
        print("Adding consolidated column...")
        cursor.execute("ALTER TABLE experience ADD COLUMN consolidated INTEGER DEFAULT 0")
        print("✓ Added consolidated column")
    else:
        print("✓ consolidated column already exists")

    conn.commit()

    # Create index on session_id if it doesn't exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='ix_experience_session_id'")
    if not cursor.fetchone():
        print("Creating index on session_id...")
        cursor.execute("CREATE INDEX ix_experience_session_id ON experience(session_id)")
        print("✓ Created index on session_id")
    else:
        print("✓ Index on session_id already exists")

    conn.commit()
    conn.close()

    print("\n✓ Migration completed successfully!")

if __name__ == "__main__":
    migrate()
